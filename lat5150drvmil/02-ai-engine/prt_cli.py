#!/usr/bin/env python3
"""
PRT Visualization & ML CLI - Natural Language Interface
Interactive data exploration, pattern recognition, ML visualization

Usage:
    python3 prt_cli.py "load dataset /path/to/data.csv"
    python3 prt_cli.py "visualize correlations in ds_1"
    python3 prt_cli.py "train classifier on ds_1"
    python3 prt_cli.py "cluster data in ds_1"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.prt_visualization_wrapper import PRTVisualizationAgent

class PRTCLI:
    def __init__(self):
        self.agent = PRTVisualizationAgent()

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Load dataset patterns
        if 'load' in query_lower or 'import' in query_lower:
            path_match = re.search(r'(?:load |import |from )([^\s]+\.(?:csv|xlsx|xls))', query)
            if path_match:
                file_path = path_match.group(1)

                name_match = re.search(r'(?:as |named |called )"?([A-Za-z0-9_\s]+)"?', query)
                name = name_match.group(1).strip() if name_match else None

                target_match = re.search(r'(?:target |label |predict )([A-Za-z0-9_]+)', query)
                target_column = target_match.group(1) if target_match else None

                return {
                    'action': 'load_dataset',
                    'file_path': file_path,
                    'name': name,
                    'target_column': target_column
                }

        # Explore dataset
        elif 'explore' in query_lower or 'describe' in query_lower or 'statistics' in query_lower:
            ds_match = re.search(r'ds_(\d+)', query)
            dataset_id = f"ds_{ds_match.group(1)}" if ds_match else None

            return {
                'action': 'explore',
                'dataset_id': dataset_id
            }

        # Visualize patterns
        elif 'visualize' in query_lower or 'plot' in query_lower or 'show' in query_lower:
            ds_match = re.search(r'ds_(\d+)', query)
            dataset_id = f"ds_{ds_match.group(1)}" if ds_match else None

            viz_type = 'correlation'  # default
            if 'distribution' in query_lower or 'histogram' in query_lower:
                viz_type = 'distribution'
            elif 'scatter' in query_lower:
                viz_type = 'scatter'
            elif 'pairplot' in query_lower or 'pairs' in query_lower:
                viz_type = 'pairplot'
            elif 'box' in query_lower or 'boxplot' in query_lower:
                viz_type = 'box'

            return {
                'action': 'visualize',
                'dataset_id': dataset_id,
                'viz_type': viz_type
            }

        # Train classifier
        elif 'train' in query_lower or 'classify' in query_lower or 'classifier' in query_lower:
            ds_match = re.search(r'ds_(\d+)', query)
            dataset_id = f"ds_{ds_match.group(1)}" if ds_match else None

            algorithm = 'random_forest'  # default
            if 'svm' in query_lower or 'support vector' in query_lower:
                algorithm = 'svm'
            elif 'logistic' in query_lower:
                algorithm = 'logistic'
            elif 'knn' in query_lower or 'k-nearest' in query_lower:
                algorithm = 'knn'
            elif 'decision tree' in query_lower:
                algorithm = 'decision_tree'

            test_match = re.search(r'test[\s=]+(0\.\d+)', query)
            test_size = float(test_match.group(1)) if test_match else 0.2

            return {
                'action': 'classify',
                'dataset_id': dataset_id,
                'algorithm': algorithm,
                'test_size': test_size
            }

        # Cluster analysis
        elif 'cluster' in query_lower:
            ds_match = re.search(r'ds_(\d+)', query)
            dataset_id = f"ds_{ds_match.group(1)}" if ds_match else None

            algorithm = 'kmeans'  # default
            if 'dbscan' in query_lower:
                algorithm = 'dbscan'
            elif 'hierarchical' in query_lower or 'agglomerative' in query_lower:
                algorithm = 'hierarchical'

            cluster_match = re.search(r'(\d+)\s+cluster', query)
            n_clusters = int(cluster_match.group(1)) if cluster_match else 3

            return {
                'action': 'cluster',
                'dataset_id': dataset_id,
                'algorithm': algorithm,
                'n_clusters': n_clusters
            }

        # Dimensionality reduction
        elif 'pca' in query_lower or 'tsne' in query_lower or 'reduce' in query_lower or 'dimension' in query_lower:
            ds_match = re.search(r'ds_(\d+)', query)
            dataset_id = f"ds_{ds_match.group(1)}" if ds_match else None

            method = 'pca'
            if 'tsne' in query_lower or 't-sne' in query_lower:
                method = 'tsne'

            comp_match = re.search(r'(\d+)\s+(?:component|dimension)', query)
            n_components = int(comp_match.group(1)) if comp_match else 2

            return {
                'action': 'reduce',
                'dataset_id': dataset_id,
                'method': method,
                'n_components': n_components
            }

        # List datasets
        elif 'list' in query_lower and 'dataset' in query_lower:
            return {'action': 'list_datasets'}

        # List models
        elif 'list' in query_lower and 'model' in query_lower:
            return {'action': 'list_models'}

        # Status
        elif 'status' in query_lower or 'info' in query_lower:
            return {'action': 'status'}

        else:
            return {'action': 'help'}

    def execute(self, query: str):
        """Execute natural language command"""
        parsed = self.parse_command(query)
        action = parsed.get('action')

        if action == 'load_dataset':
            result = self.agent.load_dataset(
                file_path=parsed['file_path'],
                name=parsed.get('name'),
                target_column=parsed.get('target_column')
            )

            if result.get('success'):
                print(f"✅ Dataset loaded successfully!")
                print(f"   ID: {result['dataset_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Rows: {result['rows']}")
                print(f"   Columns: {result['columns']}")
                print(f"   Numeric: {', '.join(result['numeric_columns'][:5])}")
                if len(result['numeric_columns']) > 5:
                    print(f"            ... and {len(result['numeric_columns']) - 5} more")
                if result.get('categorical_columns'):
                    print(f"   Categorical: {', '.join(result['categorical_columns'][:3])}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'explore':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., ds_1)")
                return

            result = self.agent.explore_dataset(dataset_id=parsed['dataset_id'])

            if result.get('success'):
                print(f"✅ Dataset exploration complete!")
                print(f"   Dataset: {parsed['dataset_id']}")
                print(f"\n   Missing values:")
                for col, count in list(result['summary']['missing_values'].items())[:10]:
                    if count > 0:
                        print(f"      {col}: {count}")
                print(f"\n   Data types:")
                for col, dtype in list(result['summary']['data_types'].items())[:10]:
                    print(f"      {col}: {dtype}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'visualize':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., ds_1)")
                return

            result = self.agent.visualize_dataset(
                dataset_id=parsed['dataset_id'],
                viz_type=parsed['viz_type']
            )

            if result.get('success'):
                print(f"✅ Visualization created!")
                print(f"   Type: {result['viz_type']}")
                print(f"   File: {result['file_path']}")
                print(f"\n   Open: file://{result['file_path']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'classify':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., ds_1)")
                return

            result = self.agent.train_classifier(
                dataset_id=parsed['dataset_id'],
                algorithm=parsed['algorithm'],
                test_size=parsed['test_size']
            )

            if result.get('success'):
                print(f"✅ Classifier trained successfully!")
                print(f"   Model ID: {result['model_id']}")
                print(f"   Algorithm: {result['algorithm']}")
                print(f"   Accuracy: {result['accuracy']:.4f}")

                print(f"\n   Confusion Matrix:")
                cm = result['confusion_matrix']
                for row in cm:
                    print(f"      {row}")

                if 'classification_report' in result:
                    print(f"\n   Classification Report:")
                    report = result['classification_report']
                    print(f"      Precision: {report.get('weighted avg', {}).get('precision', 0):.3f}")
                    print(f"      Recall: {report.get('weighted avg', {}).get('recall', 0):.3f}")
                    print(f"      F1-Score: {report.get('weighted avg', {}).get('f1-score', 0):.3f}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'cluster':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., ds_1)")
                return

            result = self.agent.cluster_analysis(
                dataset_id=parsed['dataset_id'],
                n_clusters=parsed['n_clusters'],
                algorithm=parsed['algorithm']
            )

            if result.get('success'):
                print(f"✅ Clustering complete!")
                print(f"   Algorithm: {result['algorithm']}")
                print(f"   Clusters: {result['n_clusters']}")
                print(f"   Silhouette score: {result['silhouette_score']:.4f}")

                print(f"\n   Cluster sizes:")
                for label, size in result['cluster_sizes'].items():
                    print(f"      Cluster {label}: {size} samples")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'reduce':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., ds_1)")
                return

            result = self.agent.dimensionality_reduction(
                dataset_id=parsed['dataset_id'],
                method=parsed['method'],
                n_components=parsed['n_components']
            )

            if result.get('success'):
                print(f"✅ Dimensionality reduction complete!")
                print(f"   Method: {result['method'].upper()}")
                print(f"   Components: {result['n_components']}")

                if result.get('explained_variance'):
                    print(f"\n   Explained variance:")
                    for i, var in enumerate(result['explained_variance'], 1):
                        print(f"      PC{i}: {var:.4f} ({var*100:.2f}%)")

                if result.get('visualization'):
                    print(f"\n   Visualization: {result['visualization']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'list_datasets':
            result = self.agent.list_datasets()

            if result.get('success'):
                print(f"📊 Loaded datasets: {result['count']}")
                for ds in result['datasets']:
                    print(f"\n   [{ds['id']}] {ds['name']}")
                    print(f"      Rows: {ds['rows']}")
                    print(f"      Columns: {ds['columns']}")
                    if ds.get('target_column'):
                        print(f"      Target: {ds['target_column']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'list_models':
            result = self.agent.list_models()

            if result.get('success'):
                print(f"🤖 Trained models: {result['count']}")
                for model in result['models']:
                    print(f"\n   [{model['id']}]")
                    print(f"      Dataset: {model['dataset_id']}")
                    print(f"      Algorithm: {model['algorithm']}")
                    print(f"      Accuracy: {model['accuracy']:.4f}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'status':
            status = self.agent.get_status()
            print(f"📊 PRT Visualization Agent Status")
            print(f"   Available: {status['available']}")
            print(f"   Dependencies:")
            for dep, avail in status['dependencies'].items():
                print(f"      {dep}: {'✅' if avail else '❌'}")
            print(f"   Datasets: {status['datasets_loaded']}")
            print(f"   Models: {status['models_trained']}")
            print(f"   Visualizations: {status['visualizations_created']}")
            print(f"   Storage: {status['storage_path']}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
📊 PRT Visualization & ML CLI - Natural Language Interface

Interactive Data Exploration, Pattern Recognition, ML Visualization

USAGE:
    python3 prt_cli.py "your natural language command"

EXAMPLES:
    # Load data
    python3 prt_cli.py "load /data/customers.csv as Customer Data target purchased"
    python3 prt_cli.py "import /data/sales.xlsx"

    # Explore
    python3 prt_cli.py "explore dataset ds_1"
    python3 prt_cli.py "describe ds_1"

    # Visualize
    python3 prt_cli.py "visualize correlations in ds_1"
    python3 prt_cli.py "plot distributions for ds_1"
    python3 prt_cli.py "show boxplot for ds_1"

    # Classification
    python3 prt_cli.py "train random forest classifier on ds_1"
    python3 prt_cli.py "train svm classifier on ds_1 test=0.3"

    # Clustering
    python3 prt_cli.py "cluster ds_1 into 5 clusters"
    python3 prt_cli.py "cluster ds_1 using dbscan"

    # Dimensionality reduction
    python3 prt_cli.py "reduce ds_1 to 2 dimensions using pca"
    python3 prt_cli.py "apply tsne to ds_1"

    # Management
    python3 prt_cli.py "list datasets"
    python3 prt_cli.py "list models"
    python3 prt_cli.py "status"

DEPENDENCIES:
    pip install pandas numpy scikit-learn matplotlib seaborn plotly joblib

TEMPEST COMPLIANCE:
    - 100% local processing (air-gapped compatible)
    - No cloud services
    - Suitable for classified data analysis
    - Standard electromagnetic profile
        """)

def main():
    if len(sys.argv) < 2:
        cli = PRTCLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = PRTCLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
