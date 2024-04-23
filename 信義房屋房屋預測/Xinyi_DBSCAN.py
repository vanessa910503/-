import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan_cluster_analysis(data_path, fraction, eps_range, min_samples_range):
    # 讀取Excel文件
    data = pd.read_csv(data_path)
    df = data.sample(frac=fraction)
    df = df.dropna(axis=0)

    rename_dict = {
        'A': '政府機構(A)',
        'B': '各級學校(B)',
        'C': '醫療院所(C)',
        'D': '飯店旅館(D)',
        'E': '金融機構(E)',
        'F': '觀光旅遊(F)',
        'G': '休閒娛樂(G)',
        'H': '逛街購物(H)',
        'I': '餐飲小吃(I)',
        'J': '行車服務(J)',
        'K': '交通設施(K)',
        'L': '民間機構(L)',
        'X': '鄰避設施(X)'
    }

    df.rename(columns=rename_dict, inplace=True)

    # Prepare features for clustering
    clustering_features = df.drop(columns=['總價元', '車位總價元'])

    corr_features = set()

    # 建立相關矩陣
    corr_matrix = clustering_features.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    # Feature Selection
    clustering_features.drop(labels=corr_features, axis=1, inplace=True)

    results = []

    for eps in range(eps_range[0], eps_range[1] + 1):
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            # 資料標準化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_features)

            # 擬合 DBSCAN 模型
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(scaled_features)

            # 複製原始 DataFrame
            df_copy = df.copy()

            # 添加分群結果欄位
            cluster_label = f'分群結果_DBSCAN_eps_{eps}_min_samples_{min_samples}'
            df_copy[cluster_label] = dbscan.labels_
            results.append(df_copy)

    return results