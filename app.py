import streamlit as st
import pandas as pd
import numpy as np
import io

# ==========================================
# 1. 핵심 로직: 데이터 클리닝 및 정규화
# ==========================================
def normalize_series(series):
    """
    모든 데이터를 '깔끔한 문자열'로 변환합니다.
    예: 1.0 -> '1', ' 1 ' -> '1', 1 -> '1'
    """
    return series.astype(str).str.strip().replace(r'\.0$', '', regex=True)

def parse_mapping(text):
    """
    입력된 텍스트를 {코드: 라벨} 딕셔너리로 변환 (코드는 무조건 문자열로 통일)
    """
    mapping = {}
    if not text: return mapping
    for item in text.replace('\n', ',').split(','):
        if ':' in item:
            k, v = item.split(':', 1)
            # 키를 정규화(공백제거, .0제거)하여 저장
            clean_k = str(k).strip().replace('.0', '') 
            mapping[clean_k] = v.strip()
    return mapping

# ==========================================
# 2. 사용자 분석 함수 (수정됨: 데이터타입 이슈 방지)
# ==========================================
def get_hierarchical_response_stats_v2(df, target_cols, response_mapping, 
                                       parent_col, parent_mapping, 
                                       sub_col, sub_mapping):
    results = {}
    
    df = df[~(df[target_cols[0]].isna())] # 값이 있는 경우만 가져오기

    # 내부 함수: 비율 계산
    def calc_ratios(subset_df):
        denom = len(subset_df)
        ratios = {}
        for code, label in response_mapping.items():
            # 데이터와 코드 모두 문자열로 비교
            count = (subset_df[target_cols] == code).sum().sum()
            ratios[label] = round(count / denom, 3) if denom > 0 else 0.0
        return pd.Series(ratios)

    # 1. [전체] 그룹
    results[('전체', '계')] = calc_ratios(df)
    for s_code, s_label in sub_mapping.items():
        sub_subset = df[df[sub_col] == s_code]
        results[('전체', s_label)] = calc_ratios(sub_subset)

    # 2. [상위] 그룹
    for p_code, p_label in parent_mapping.items():
        parent_subset = df[df[parent_col] == p_code]
        results[(p_label, '계')] = calc_ratios(parent_subset)
        for s_code, s_label in sub_mapping.items():
            sub_subset = parent_subset[parent_subset[sub_col] == s_code]
            results[(p_label, s_label)] = calc_ratios(sub_subset)

    return pd.DataFrame(results)

def get_hierarchical_mean_point(df, target_cols, 
                                parent_col, parent_mapping, 
                                sub_col, sub_mapping):
    results = {}
    
    df = df[~(df[target_cols[0]].isna())] # 값이 있는 경우만 가져오기

    # 내부 함수: 평균 계산 (이미 숫자형으로 변환된 데이터 사용)
    def calc_means(subset_df):
        if subset_df.empty:
            return pd.Series([np.nan] * len(target_cols), index=target_cols)
        return subset_df[target_cols].mean(axis=0).round(3)

    # 1. [전체] 그룹
    results[('전체', '계')] = calc_means(df)
    for s_code, s_label in sub_mapping.items():
        sub_subset = df[df[sub_col] == s_code]
        results[('전체', s_label)] = calc_means(sub_subset)

    # 2. [상위] 그룹
    for p_code, p_label in parent_mapping.items():
        parent_subset = df[df[parent_col] == p_code]
        results[(p_label, '계')] = calc_means(parent_subset)
        for s_code, s_label in sub_mapping.items():
            sub_subset = parent_subset[parent_subset[sub_col] == s_code]
            results[(p_label, s_label)] = calc_means(sub_subset)

    result_df = pd.DataFrame(results)
    result_df.index.name = '문항'
    result_df.columns.names = ['그룹', '세부그룹']
    return result_df

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="간편 데이터 분석기", layout="wide")
st.title("📊 간편 계층적 데이터 분석기")

with st.sidebar:
    uploaded_file = st.file_uploader("파일 업로드 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
    analysis_mode = st.radio("분석 모드", ("비율 분석 (Response Stats)", "평균 분석 (Mean Point)"))
    st.info("💡 팁: 데이터 내의 1.0, ' 1 ' 등은 자동으로 '1'로 처리됩니다.")

if uploaded_file:
    # 데이터 로드
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # 콤팩트한 설정을 위해 expander 사용
    with st.expander("⚙️ 분석 설정 (클릭하여 열기)", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            all_cols = df_raw.columns.tolist()
            parent_col = st.selectbox("상위 그룹 (Parent)", all_cols, index=0)
            sub_col = st.selectbox("하위 그룹 (Sub)", all_cols, index=min(1, len(all_cols)-1))
            target_cols = st.multiselect("분석 문항 (Targets)", all_cols)

        with col2:
            # 그룹 매핑 자동 생성 (정규화된 값 기준)
            p_vals = sorted(normalize_series(df_raw[parent_col].dropna()).unique())
            s_vals = sorted(normalize_series(df_raw[sub_col].dropna()).unique())
            
            p_map_txt = st.text_area("상위 그룹 매핑", value=", ".join([f"{v}:{v}" for v in p_vals]), height=68)
            s_map_txt = st.text_area("하위 그룹 매핑", value=", ".join([f"{v}:{v}" for v in s_vals]), height=68)
            
            r_map_txt = ""
            if "비율" in analysis_mode:
                r_map_txt = st.text_area("응답 매핑 (예: 1:만족, 2:보통)", placeholder="1:그렇다, 0:아니다", height=68)

    if st.button("분석 실행", type="primary", use_container_width=True):
        if not target_cols:
            st.error("분석할 문항을 선택하세요.")
        else:
            try:
                # 1. 데이터 전처리 (복사본 사용)
                df = df_raw.copy()
                
                # 그룹 컬럼 정규화 (무조건 문자열 '1' 형태로 통일)
                df[parent_col] = normalize_series(df[parent_col])
                df[sub_col] = normalize_series(df[sub_col])
                
                # 매핑 파싱
                p_map = parse_mapping(p_map_txt)
                s_map = parse_mapping(s_map_txt)

                result_df = None

                if "비율" in analysis_mode:
                    if not r_map_txt:
                        st.warning("비율 분석을 위해 응답 매핑을 입력하세요.")
                    else:
                        r_map = parse_mapping(r_map_txt)
                        # 타겟 컬럼도 정규화 (문자열 매칭)
                        for col in target_cols:
                            df[col] = normalize_series(df[col])
                            
                        result_df = get_hierarchical_response_stats_v2(
                            df, target_cols, r_map, parent_col, p_map, sub_col, s_map
                        )

                else: # 평균 분석
                    # 타겟 컬럼을 숫자로 변환 (에러는 NaN 처리)
                    for col in target_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                    result_df = get_hierarchical_mean_point(
                        df, target_cols, parent_col, p_map, sub_col, s_map
                    )

                # 결과 출력
                if result_df is not None:
                    st.success("✅ 분석 완료")
                    st.dataframe(result_df)
                    st.download_button("CSV 다운로드", result_df.to_csv(encoding='utf-8-sig'), "result.csv", "text/csv")

            except Exception as e:
                st.error(f"오류 발생: {e}")
else:
    st.write("👈 파일을 업로드해주세요.")