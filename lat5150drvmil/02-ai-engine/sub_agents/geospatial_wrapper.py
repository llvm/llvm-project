#!/usr/bin/env python3
"""
Geospatial Analytics Agent - OpenSphere-Inspired
OSINT, Threat Intel, Infrastructure Mapping, Geo-tagged Analytics

Capabilities:
- Load geospatial data (KML, GeoJSON, Shapefiles, GPX, CSV)
- Threat intelligence mapping
- Cyber-physical infrastructure visualization
- Temporal geo-analytics
- Interactive maps (2D/3D)
- OSINT data correlation

Python-native using: GeoPandas, Folium, PyDeck, Plotly
"""

import os
import json
import tempfile
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GeospatialAgent:
    def __init__(self):
        """Initialize geospatial analytics agent"""
        self.data_dir = Path.home() / ".dsmil" / "geospatial"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.maps_dir = self.data_dir / "maps"
        self.maps_dir.mkdir(exist_ok=True)

        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)

        # Track loaded datasets
        self.datasets = {}
        self._load_dataset_registry()

        # Check dependencies
        self.dependencies = self._check_dependencies()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Check which geospatial libraries are available"""
        deps = {}

        try:
            import geopandas
            deps['geopandas'] = True
        except ImportError:
            deps['geopandas'] = False

        try:
            import folium
            deps['folium'] = True
        except ImportError:
            deps['folium'] = False

        try:
            import pydeck
            deps['pydeck'] = True
        except ImportError:
            deps['pydeck'] = False

        try:
            import plotly
            deps['plotly'] = True
        except ImportError:
            deps['plotly'] = False

        try:
            import shapely
            deps['shapely'] = True
        except ImportError:
            deps['shapely'] = False

        return deps

    def is_available(self) -> bool:
        """Check if agent has minimum required dependencies"""
        # Minimum: geopandas OR folium
        return self.dependencies.get('geopandas', False) or self.dependencies.get('folium', False)

    def _load_dataset_registry(self):
        """Load dataset registry from disk"""
        registry_file = self.data_dir / "datasets.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.datasets = json.load(f)

    def _save_dataset_registry(self):
        """Save dataset registry to disk"""
        registry_file = self.data_dir / "datasets.json"
        with open(registry_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)

    def load_data(self, file_path: str, dataset_name: Optional[str] = None,
                  data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load geospatial data from various formats

        Args:
            file_path: Path to geospatial file
            dataset_name: Name to register dataset as
            data_type: Override auto-detection (geojson, kml, shp, gpx, csv)

        Returns:
            Dict with status and dataset info
        """
        if not self.dependencies.get('geopandas', False):
            return {
                "success": False,
                "error": "GeoPandas not installed. Install: pip install geopandas"
            }

        import geopandas as gpd

        file_path = Path(file_path)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Auto-detect data type
        if not data_type:
            suffix = file_path.suffix.lower()
            type_map = {
                '.geojson': 'geojson',
                '.json': 'geojson',
                '.kml': 'kml',
                '.shp': 'shapefile',
                '.gpx': 'gpx',
                '.csv': 'csv'
            }
            data_type = type_map.get(suffix, 'unknown')

        try:
            # Load based on type
            if data_type in ['geojson', 'json', 'shapefile', 'gpx']:
                gdf = gpd.read_file(file_path)
            elif data_type == 'kml':
                # KML requires fiona with KML driver
                import fiona
                fiona.drvsupport.supported_drivers['KML'] = 'rw'
                gdf = gpd.read_file(file_path, driver='KML')
            elif data_type == 'csv':
                # CSV requires lat/lon columns
                import pandas as pd
                df = pd.read_csv(file_path)

                # Auto-detect lat/lon columns
                lat_cols = [c for c in df.columns if 'lat' in c.lower()]
                lon_cols = [c for c in df.columns if 'lon' in c.lower()]

                if not lat_cols or not lon_cols:
                    return {
                        "success": False,
                        "error": "CSV must have latitude and longitude columns"
                    }

                lat_col = lat_cols[0]
                lon_col = lon_cols[0]

                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                    crs="EPSG:4326"
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported data type: {data_type}"
                }

            # Register dataset
            if not dataset_name:
                dataset_name = file_path.stem

            dataset_id = f"geo_{len(self.datasets) + 1}"

            self.datasets[dataset_id] = {
                "id": dataset_id,
                "name": dataset_name,
                "file_path": str(file_path),
                "data_type": data_type,
                "feature_count": len(gdf),
                "crs": str(gdf.crs),
                "bounds": gdf.total_bounds.tolist(),
                "columns": list(gdf.columns)
            }

            # Save GeoDataFrame to disk
            dataset_file = self.datasets_dir / f"{dataset_id}.geojson"
            gdf.to_file(dataset_file, driver='GeoJSON')

            self._save_dataset_registry()

            return {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "feature_count": len(gdf),
                "crs": str(gdf.crs),
                "bounds": gdf.total_bounds.tolist(),
                "columns": list(gdf.columns)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load data: {str(e)}"
            }

    def create_map(self, dataset_ids: Optional[List[str]] = None,
                   map_type: str = "folium",
                   output_file: Optional[str] = None,
                   title: str = "Geospatial Analytics",
                   style: str = "default") -> Dict[str, Any]:
        """
        Create interactive map visualization

        Args:
            dataset_ids: List of dataset IDs to visualize (None = all)
            map_type: 'folium' (2D), 'pydeck' (3D), 'plotly' (interactive)
            output_file: Save location (auto-generate if None)
            title: Map title
            style: Map style/theme

        Returns:
            Dict with map info and file path
        """
        if dataset_ids is None:
            dataset_ids = list(self.datasets.keys())

        if not dataset_ids:
            return {
                "success": False,
                "error": "No datasets loaded. Use load_data() first."
            }

        # Load datasets
        gdfs = []
        for dataset_id in dataset_ids:
            if dataset_id not in self.datasets:
                continue
            dataset_file = self.datasets_dir / f"{dataset_id}.geojson"
            if dataset_file.exists():
                import geopandas as gpd
                gdf = gpd.read_file(dataset_file)
                gdf['_dataset_name'] = self.datasets[dataset_id]['name']
                gdfs.append(gdf)

        if not gdfs:
            return {
                "success": False,
                "error": "No valid datasets found"
            }

        # Combine all datasets
        import geopandas as gpd
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        # Create map based on type
        try:
            if map_type == "folium":
                result = self._create_folium_map(combined_gdf, title, style)
            elif map_type == "pydeck":
                result = self._create_pydeck_map(combined_gdf, title, style)
            elif map_type == "plotly":
                result = self._create_plotly_map(combined_gdf, title, style)
            else:
                return {
                    "success": False,
                    "error": f"Unknown map type: {map_type}"
                }

            if result.get('success'):
                # Save map
                if not output_file:
                    output_file = self.maps_dir / f"map_{len(list(self.maps_dir.glob('*.html'))) + 1}.html"
                else:
                    output_file = Path(output_file)

                result['map_object'].save(str(output_file))
                result['file_path'] = str(output_file)
                del result['map_object']  # Remove from return

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create map: {str(e)}"
            }

    def _create_folium_map(self, gdf, title: str, style: str) -> Dict[str, Any]:
        """Create 2D interactive map with Folium"""
        if not self.dependencies.get('folium'):
            return {
                "success": False,
                "error": "Folium not installed. Install: pip install folium"
            }

        import folium
        import pandas as pd

        # Calculate center
        centroid = gdf.geometry.unary_union.centroid
        center = [centroid.y, centroid.x]

        # Create map
        tiles_map = {
            'default': 'OpenStreetMap',
            'satellite': 'Esri.WorldImagery',
            'dark': 'CartoDB dark_matter',
            'light': 'CartoDB positron'
        }

        m = folium.Map(
            location=center,
            zoom_start=10,
            tiles=tiles_map.get(style, 'OpenStreetMap')
        )

        # Add title
        title_html = f'''
            <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Add features
        for idx, row in gdf.iterrows():
            # Create popup
            popup_html = f"<b>{row.get('_dataset_name', 'Unknown')}</b><br>"
            for col in gdf.columns:
                if col not in ['geometry', '_dataset_name']:
                    popup_html += f"{col}: {row[col]}<br>"

            if row.geometry.geom_type == 'Point':
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            else:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)

        return {
            "success": True,
            "map_type": "folium",
            "feature_count": len(gdf),
            "map_object": m
        }

    def _create_pydeck_map(self, gdf, title: str, style: str) -> Dict[str, Any]:
        """Create 3D map with PyDeck"""
        if not self.dependencies.get('pydeck'):
            return {
                "success": False,
                "error": "PyDeck not installed. Install: pip install pydeck"
            }

        import pydeck as pdk
        import pandas as pd

        # Convert to points for 3D visualization
        gdf_points = gdf.copy()
        gdf_points['lon'] = gdf_points.geometry.centroid.x
        gdf_points['lat'] = gdf_points.geometry.centroid.y

        # Calculate center
        center_lon = gdf_points['lon'].mean()
        center_lat = gdf_points['lat'].mean()

        # Create layer
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=gdf_points,
            get_position='[lon, lat]',
            get_radius=100,
            get_fill_color=[255, 0, 0, 140],
            pickable=True
        )

        # Create view
        view_state = pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=10,
            pitch=45
        )

        # Create deck
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=f'mapbox://styles/mapbox/{style}-v9' if style != 'default' else 'mapbox://styles/mapbox/light-v9'
        )

        # Save as HTML
        html = deck.to_html(as_string=True)

        # Create mock object with save method
        class DeckWrapper:
            def __init__(self, html_content):
                self.html = html_content
            def save(self, path):
                with open(path, 'w') as f:
                    f.write(self.html)

        return {
            "success": True,
            "map_type": "pydeck",
            "feature_count": len(gdf),
            "map_object": DeckWrapper(html)
        }

    def _create_plotly_map(self, gdf, title: str, style: str) -> Dict[str, Any]:
        """Create interactive map with Plotly"""
        if not self.dependencies.get('plotly'):
            return {
                "success": False,
                "error": "Plotly not installed. Install: pip install plotly"
            }

        import plotly.graph_objects as go
        import pandas as pd

        # Convert to points
        gdf_points = gdf.copy()
        gdf_points['lon'] = gdf_points.geometry.centroid.x
        gdf_points['lat'] = gdf_points.geometry.centroid.y

        # Create figure
        fig = go.Figure()

        # Add scatter mapbox
        fig.add_trace(go.Scattermapbox(
            lat=gdf_points['lat'],
            lon=gdf_points['lon'],
            mode='markers',
            marker=dict(size=10, color='red'),
            text=gdf_points.get('_dataset_name', 'Unknown'),
            hoverinfo='text'
        ))

        # Update layout
        center_lat = gdf_points['lat'].mean()
        center_lon = gdf_points['lon'].mean()

        style_map = {
            'default': 'open-street-map',
            'satellite': 'satellite',
            'dark': 'carto-darkmatter',
            'light': 'carto-positron'
        }

        fig.update_layout(
            title=title,
            mapbox=dict(
                style=style_map.get(style, 'open-street-map'),
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10
            ),
            showlegend=False
        )

        # Create wrapper for saving
        class PlotlyWrapper:
            def __init__(self, figure):
                self.fig = figure
            def save(self, path):
                self.fig.write_html(path)

        return {
            "success": True,
            "map_type": "plotly",
            "feature_count": len(gdf),
            "map_object": PlotlyWrapper(fig)
        }

    def threat_intel_analysis(self, dataset_id: str,
                              analysis_type: str = "hotspot") -> Dict[str, Any]:
        """
        Perform threat intelligence analysis on geospatial data

        Args:
            dataset_id: Dataset to analyze
            analysis_type: 'hotspot', 'density', 'cluster', 'temporal'

        Returns:
            Dict with analysis results
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        dataset_file = self.datasets_dir / f"{dataset_id}.geojson"
        if not dataset_file.exists():
            return {
                "success": False,
                "error": "Dataset file not found"
            }

        import geopandas as gpd
        import pandas as pd

        gdf = gpd.read_file(dataset_file)

        if analysis_type == "hotspot":
            # Identify geographic hotspots
            # Group by spatial grid
            gdf['grid_x'] = (gdf.geometry.x * 100).astype(int)
            gdf['grid_y'] = (gdf.geometry.y * 100).astype(int)

            hotspots = gdf.groupby(['grid_x', 'grid_y']).size().reset_index(name='count')
            hotspots = hotspots.sort_values('count', ascending=False)

            return {
                "success": True,
                "analysis_type": "hotspot",
                "hotspots": hotspots.head(10).to_dict('records'),
                "total_features": len(gdf)
            }

        elif analysis_type == "density":
            # Calculate spatial density
            total_area = gdf.unary_union.convex_hull.area
            density = len(gdf) / total_area if total_area > 0 else 0

            return {
                "success": True,
                "analysis_type": "density",
                "density": density,
                "feature_count": len(gdf),
                "area": total_area
            }

        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}"
            }

    def list_datasets(self) -> Dict[str, Any]:
        """List all loaded datasets"""
        return {
            "success": True,
            "datasets": list(self.datasets.values()),
            "count": len(self.datasets)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "available": self.is_available(),
            "dependencies": self.dependencies,
            "datasets_loaded": len(self.datasets),
            "maps_created": len(list(self.maps_dir.glob('*.html'))),
            "storage_path": str(self.data_dir)
        }

# Export
__all__ = ['GeospatialAgent']
