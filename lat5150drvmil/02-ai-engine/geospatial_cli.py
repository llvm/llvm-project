#!/usr/bin/env python3
"""
Geospatial Analytics CLI - Natural Language Interface
OSINT, threat-intel mapping, infrastructure visualization

Usage:
    python3 geospatial_cli.py "load threat data from /path/to/data.geojson"
    python3 geospatial_cli.py "create map of threat intelligence"
    python3 geospatial_cli.py "analyze hotspots in dataset geo_1"
    python3 geospatial_cli.py "list all datasets"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.geospatial_wrapper import GeospatialAgent

class GeospatialCLI:
    def __init__(self):
        self.agent = GeospatialAgent()

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Load data patterns
        if 'load' in query_lower or 'import' in query_lower:
            # Extract file path
            path_match = re.search(r'(?:from |load |import )([^\s]+\.(?:geojson|kml|shp|gpx|csv))', query)
            if path_match:
                file_path = path_match.group(1)
                # Extract dataset name if provided
                name_match = re.search(r'(?:as |named |called )"([^"]+)"', query)
                dataset_name = name_match.group(1) if name_match else None

                return {
                    'action': 'load_data',
                    'file_path': file_path,
                    'dataset_name': dataset_name
                }

        # Create map patterns
        elif 'map' in query_lower or 'visualize' in query_lower or 'create map' in query_lower:
            # Extract dataset IDs
            dataset_match = re.findall(r'(?:dataset|geo)_(\d+)', query)
            dataset_ids = [f"geo_{d}" for d in dataset_match] if dataset_match else None

            # Extract map type
            map_type = 'folium'  # default
            if 'pydeck' in query_lower or '3d' in query_lower:
                map_type = 'pydeck'
            elif 'plotly' in query_lower or 'interactive' in query_lower:
                map_type = 'plotly'

            # Extract style
            style = 'default'
            if 'dark' in query_lower:
                style = 'dark'
            elif 'satellite' in query_lower:
                style = 'satellite'
            elif 'light' in query_lower:
                style = 'light'

            # Extract title
            title_match = re.search(r'(?:title |titled |called )"([^"]+)"', query)
            title = title_match.group(1) if title_match else "Geospatial Map"

            return {
                'action': 'create_map',
                'dataset_ids': dataset_ids,
                'map_type': map_type,
                'style': style,
                'title': title
            }

        # Analyze patterns
        elif 'analyze' in query_lower or 'analysis' in query_lower:
            dataset_match = re.search(r'(?:dataset|geo)_(\d+)', query)
            dataset_id = f"geo_{dataset_match.group(1)}" if dataset_match else None

            analysis_type = 'hotspot'  # default
            if 'density' in query_lower:
                analysis_type = 'density'

            return {
                'action': 'analyze',
                'dataset_id': dataset_id,
                'analysis_type': analysis_type
            }

        # List datasets
        elif 'list' in query_lower or 'show' in query_lower:
            return {'action': 'list_datasets'}

        # Status
        elif 'status' in query_lower or 'info' in query_lower:
            return {'action': 'status'}

        else:
            return {'action': 'help'}

    def execute(self, query: str):
        """Execute natural language command"""
        parsed = self.parse_command(query)
        action = parsed.get('action')

        if action == 'load_data':
            result = self.agent.load_data(
                file_path=parsed['file_path'],
                dataset_name=parsed.get('dataset_name')
            )

            if result.get('success'):
                print(f"✅ Dataset loaded successfully!")
                print(f"   ID: {result['dataset_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Features: {result['feature_count']}")
                print(f"   Columns: {', '.join(result['numeric_columns'])}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'create_map':
            result = self.agent.create_map(
                dataset_ids=parsed.get('dataset_ids'),
                map_type=parsed['map_type'],
                title=parsed['title'],
                style=parsed['style']
            )

            if result.get('success'):
                print(f"✅ Map created successfully!")
                print(f"   Type: {result['map_type']}")
                print(f"   Features: {result['feature_count']}")
                print(f"   File: {result['file_path']}")
                print(f"\n   Open in browser: file://{result['file_path']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'analyze':
            if not parsed.get('dataset_id'):
                print("❌ Please specify a dataset ID (e.g., geo_1)")
                return

            result = self.agent.threat_intel_analysis(
                dataset_id=parsed['dataset_id'],
                analysis_type=parsed['analysis_type']
            )

            if result.get('success'):
                print(f"✅ Analysis complete!")
                print(f"   Type: {result['analysis_type']}")
                print(f"   Total features: {result['total_features']}")

                if result['analysis_type'] == 'hotspot' and 'hotspots' in result:
                    print(f"\n   Top hotspots:")
                    for i, hotspot in enumerate(result['hotspots'][:5], 1):
                        print(f"   {i}. Grid ({hotspot['grid_x']}, {hotspot['grid_y']}): {hotspot['count']} features")
                elif result['analysis_type'] == 'density':
                    print(f"   Density: {result['density']:.6f} features/area")
                    print(f"   Area: {result['area']:.2f} sq units")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'list_datasets':
            result = self.agent.list_datasets()

            if result.get('success'):
                print(f"📊 Loaded datasets: {result['count']}")
                for dataset in result['datasets']:
                    print(f"\n   [{dataset['id']}] {dataset['name']}")
                    print(f"      Features: {dataset['feature_count']}")
                    print(f"      Type: {dataset['data_type']}")
                    print(f"      CRS: {dataset['crs']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'status':
            status = self.agent.get_status()
            print(f"🌍 Geospatial Agent Status")
            print(f"   Available: {status['available']}")
            print(f"   Dependencies:")
            for dep, avail in status['dependencies'].items():
                print(f"      {dep}: {'✅' if avail else '❌'}")
            print(f"   Datasets: {status['datasets_loaded']}")
            print(f"   Maps: {status['maps_created']}")
            print(f"   Storage: {status['storage_path']}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
🌍 Geospatial Analytics CLI - Natural Language Interface

OSINT, Threat Intelligence, Infrastructure Mapping

USAGE:
    python3 geospatial_cli.py "your natural language command"

EXAMPLES:
    # Load data
    python3 geospatial_cli.py "load /data/threat.geojson as APT Campaign"
    python3 geospatial_cli.py "import /data/osint.kml"

    # Create maps
    python3 geospatial_cli.py "create map of geo_1"
    python3 geospatial_cli.py "visualize geo_1 in dark style"
    python3 geospatial_cli.py "create 3d map of geo_1 titled Threat Map"

    # Analysis
    python3 geospatial_cli.py "analyze hotspots in geo_1"
    python3 geospatial_cli.py "analyze density in geo_1"

    # Management
    python3 geospatial_cli.py "list datasets"
    python3 geospatial_cli.py "status"

DEPENDENCIES:
    pip install geopandas folium pydeck plotly shapely pandas

TEMPEST COMPLIANCE:
    - All data processed locally (air-gapped compatible)
    - No cloud services required
    - Electromagnetic emissions: Standard workstation levels
    - Recommend: Shielded environment for classified data
        """)

def main():
    if len(sys.argv) < 2:
        cli = GeospatialCLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = GeospatialCLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
