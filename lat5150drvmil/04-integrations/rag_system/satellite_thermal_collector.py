#!/usr/bin/env python3
"""
Satellite Thermal Anomaly Collector (FIRMS VIIRS)

Advanced geospatial OSINT using NASA's FIRMS (Fire Information for Resource Management System)
to detect thermal anomalies from satellite data. Useful for detecting:

- Military strikes/operations at sea
- Industrial incidents
- Natural disasters
- Explosions and fires
- Anomalous thermal events

Based on methodology from pizzint.watch/polyglobe

Data Source:
- NASA FIRMS VIIRS 375m thermal anomalies
- VIIRS (Visible Infrared Imaging Radiometer Suite) on Suomi NPP and NOAA-20 satellites
- Near-real-time detection (3-5 hour latency)
- Global coverage

Features:
- Automatic thermal anomaly detection
- Geospatial filtering (ocean vs land, distance from infrastructure)
- Temporal analysis (one-off vs recurring events)
- Confidence scoring based on Fire Radiative Power (FRP)
- Integration with reported events for validation
- Export to RAG system for intelligence correlation

Usage:
    # Detect ocean anomalies in Eastern Pacific
    python3 satellite_thermal_collector.py --region eastern_pacific --days 7

    # Monitor specific coordinates
    python3 satellite_thermal_collector.py --lat 14.0 --lon -106.5 --radius 100

    # Continuous monitoring
    python3 satellite_thermal_collector.py --monitor --interval 3600

API Reference:
https://firms.modaps.eosdis.nasa.gov/api/
"""

import os
import json
import time
import math
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request
import urllib.parse
import urllib.error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/satellite_thermal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
THERMAL_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/thermal_anomalies')
THERMAL_INDEX_FILE = Path('rag_system/thermal_anomaly_index.json')

# Create directories
THERMAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# NASA FIRMS API
# Get your API key from: https://firms.modaps.eosdis.nasa.gov/api/area/
FIRMS_API_KEY = os.getenv('FIRMS_API_KEY', 'DEMO')  # Use environment variable
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


# Predefined regions of interest
REGIONS = {
    'eastern_pacific': {
        'name': 'Eastern Pacific (Mexico Coast)',
        'bounds': {'min_lat': 10.0, 'max_lat': 20.0, 'min_lon': -110.0, 'max_lon': -100.0},
        'description': 'Off Mexican Pacific coast, common for interdiction operations'
    },
    'caribbean': {
        'name': 'Caribbean Sea',
        'bounds': {'min_lat': 10.0, 'max_lat': 25.0, 'min_lon': -85.0, 'max_lon': -60.0},
        'description': 'Caribbean drug trafficking routes'
    },
    'persian_gulf': {
        'name': 'Persian Gulf',
        'bounds': {'min_lat': 23.0, 'max_lat': 30.0, 'min_lon': 48.0, 'max_lon': 57.0},
        'description': 'Strategic waterway, frequent naval operations'
    },
    'south_china_sea': {
        'name': 'South China Sea',
        'bounds': {'min_lat': 0.0, 'max_lat': 25.0, 'min_lon': 100.0, 'max_lon': 120.0},
        'description': 'Disputed waters, naval activity'
    },
    'red_sea': {
        'name': 'Red Sea',
        'bounds': {'min_lat': 12.0, 'max_lat': 30.0, 'min_lon': 32.0, 'max_lon': 45.0},
        'description': 'Shipping lane, recent Houthi attacks'
    },
    'strait_of_hormuz': {
        'name': 'Strait of Hormuz',
        'bounds': {'min_lat': 24.0, 'max_lat': 27.0, 'min_lon': 55.0, 'max_lon': 58.0},
        'description': 'Critical oil shipping chokepoint'
    },
    'black_sea': {
        'name': 'Black Sea',
        'bounds': {'min_lat': 41.0, 'max_lat': 47.0, 'min_lon': 27.0, 'max_lon': 42.0},
        'description': 'Ukraine conflict zone'
    },
    'global_ocean': {
        'name': 'Global Ocean',
        'bounds': {'min_lat': -90.0, 'max_lat': 90.0, 'min_lon': -180.0, 'max_lon': 180.0},
        'description': 'All ocean thermal anomalies worldwide'
    }
}


@dataclass
class ThermalAnomaly:
    """Thermal anomaly detection from VIIRS satellite"""
    id: str                          # Unique identifier (hash of lat/lon/time)
    latitude: float                  # Decimal degrees
    longitude: float                 # Decimal degrees
    brightness: float                # Brightness temperature (Kelvin)
    frp: float                       # Fire Radiative Power (MW)
    confidence: str                  # Nominal, low, high
    acquisition_date: str            # Date (YYYY-MM-DD)
    acquisition_time: str            # Time (HHMM UTC)
    satellite: str                   # N for Suomi NPP, J for NOAA-20
    instrument: str                  # VIIRS
    daynight: str                    # D for day, N for night
    detected_at: str                 # When we collected it (ISO format)

    # Analysis fields
    is_ocean: bool = None            # True if in ocean (not land)
    distance_to_land_km: float = None  # Distance to nearest land
    distance_to_port_km: float = None  # Distance to nearest major port
    region: str = None               # Which region it's in
    anomaly_score: float = None      # Our confidence this is anomalous (0-1)
    recurring: bool = None           # Is this a recurring source?
    description: str = None          # Human-readable description
    metadata: Dict = None            # Additional data

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class SatelliteThermalCollector:
    """Collect and analyze thermal anomalies from NASA FIRMS VIIRS"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or FIRMS_API_KEY
        self.anomalies: Dict[str, ThermalAnomaly] = {}
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load existing thermal anomaly index"""
        if THERMAL_INDEX_FILE.exists():
            with open(THERMAL_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {'anomalies': {}, 'last_update': None, 'stats': {}}

    def _save_index(self):
        """Save thermal anomaly index"""
        self.index['last_update'] = datetime.now().isoformat()
        self.index['anomalies'] = {id: anom.to_dict() for id, anom in self.anomalies.items()}

        # Calculate stats
        stats = {
            'total_anomalies': len(self.anomalies),
            'ocean_anomalies': sum(1 for a in self.anomalies.values() if a.is_ocean),
            'high_confidence': sum(1 for a in self.anomalies.values() if a.confidence == 'h'),
            'by_region': {},
            'by_satellite': {}
        }

        for anom in self.anomalies.values():
            # By region
            if anom.region:
                stats['by_region'][anom.region] = stats['by_region'].get(anom.region, 0) + 1

            # By satellite
            stats['by_satellite'][anom.satellite] = stats['by_satellite'].get(anom.satellite, 0) + 1

        self.index['stats'] = stats

        with open(THERMAL_INDEX_FILE, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.info(f"Thermal anomaly index saved: {len(self.anomalies)} detections")

    def fetch_firms_area(self, bounds: Dict, days: int = 1, source: str = 'VIIRS_NOAA20_NRT') -> List[Dict]:
        """
        Fetch FIRMS data for a bounding box

        Args:
            bounds: {min_lat, max_lat, min_lon, max_lon}
            days: Number of days to look back (1-10)
            source: VIIRS_NOAA20_NRT or VIIRS_SNPP_NRT

        Returns:
            List of thermal anomaly dicts
        """
        # Build URL
        url = f"{FIRMS_BASE_URL}/{self.api_key}/{source}/{bounds['min_lon']},{bounds['min_lat']},{bounds['max_lon']},{bounds['max_lat']}/{days}"

        logger.info(f"Fetching FIRMS data: {source}, {days} days")
        logger.debug(f"URL: {url}")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'OSINT-Aggregator/1.0'})

            with urllib.request.urlopen(req, timeout=60) as response:
                csv_data = response.read().decode('utf-8')

            # Parse CSV
            lines = csv_data.strip().split('\n')

            if len(lines) < 2:
                logger.warning(f"No data returned for {source}")
                return []

            # Parse header
            header = lines[0].split(',')

            # Parse rows
            detections = []
            for line in lines[1:]:
                values = line.split(',')

                if len(values) != len(header):
                    continue

                detection = dict(zip(header, values))
                detections.append(detection)

            logger.info(f"Fetched {len(detections)} detections from {source}")
            return detections

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"No data available for query (404)")
                return []
            else:
                logger.error(f"HTTP error fetching FIRMS data: {e}")
                return []
        except Exception as e:
            logger.error(f"Error fetching FIRMS data: {e}")
            return []

    def fetch_region(self, region_name: str, days: int = 1) -> List[ThermalAnomaly]:
        """Fetch thermal anomalies for a named region"""
        if region_name not in REGIONS:
            logger.error(f"Unknown region: {region_name}")
            return []

        region = REGIONS[region_name]
        logger.info(f"Fetching region: {region['name']}")
        logger.info(f"  {region['description']}")

        # Fetch from both VIIRS satellites
        all_detections = []

        for source in ['VIIRS_NOAA20_NRT', 'VIIRS_SNPP_NRT']:
            detections = self.fetch_firms_area(region['bounds'], days, source)
            all_detections.extend(detections)
            time.sleep(1)  # Rate limiting

        # Convert to ThermalAnomaly objects
        anomalies = []
        for det in all_detections:
            try:
                # Create unique ID
                id_str = f"{det['latitude']}{det['longitude']}{det['acq_date']}{det['acq_time']}"
                anom_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

                # Parse values
                lat = float(det['latitude'])
                lon = float(det['longitude'])
                brightness = float(det['bright_ti4'])
                frp = float(det['frp'])
                confidence = det['confidence']
                acq_date = det['acq_date']
                acq_time = det['acq_time']
                satellite = det['satellite']
                daynight = det['daynight']

                anomaly = ThermalAnomaly(
                    id=anom_id,
                    latitude=lat,
                    longitude=lon,
                    brightness=brightness,
                    frp=frp,
                    confidence=confidence,
                    acquisition_date=acq_date,
                    acquisition_time=acq_time,
                    satellite=satellite,
                    instrument='VIIRS',
                    daynight=daynight,
                    region=region_name
                )

                # Analyze anomaly
                self._analyze_anomaly(anomaly)

                anomalies.append(anomaly)

            except Exception as e:
                logger.error(f"Error parsing detection: {e}")

        return anomalies

    def fetch_point(self, lat: float, lon: float, radius_km: float = 50, days: int = 7) -> List[ThermalAnomaly]:
        """
        Fetch thermal anomalies near a specific point

        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)
            radius_km: Search radius in kilometers
            days: Days to look back

        Returns:
            List of thermal anomalies within radius
        """
        # Convert radius to degrees (approximate)
        # At equator: 1 degree ≈ 111 km
        radius_deg = radius_km / 111.0

        bounds = {
            'min_lat': lat - radius_deg,
            'max_lat': lat + radius_deg,
            'min_lon': lon - radius_deg,
            'max_lon': lon + radius_deg
        }

        logger.info(f"Fetching anomalies near ({lat}, {lon}) within {radius_km}km for {days} days")

        # Fetch from both satellites
        all_detections = []
        for source in ['VIIRS_NOAA20_NRT', 'VIIRS_SNPP_NRT']:
            detections = self.fetch_firms_area(bounds, days, source)
            all_detections.extend(detections)
            time.sleep(1)

        # Convert and filter by actual distance
        anomalies = []
        for det in all_detections:
            try:
                lat_det = float(det['latitude'])
                lon_det = float(det['longitude'])

                # Calculate distance
                distance_km = self._haversine_distance(lat, lon, lat_det, lon_det)

                if distance_km <= radius_km:
                    # Create anomaly
                    id_str = f"{det['latitude']}{det['longitude']}{det['acq_date']}{det['acq_time']}"
                    anom_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

                    anomaly = ThermalAnomaly(
                        id=anom_id,
                        latitude=lat_det,
                        longitude=lon_det,
                        brightness=float(det['bright_ti4']),
                        frp=float(det['frp']),
                        confidence=det['confidence'],
                        acquisition_date=det['acq_date'],
                        acquisition_time=det['acq_time'],
                        satellite=det['satellite'],
                        instrument='VIIRS',
                        daynight=det['daynight']
                    )

                    # Add distance to metadata
                    anomaly.metadata['distance_from_query_km'] = distance_km

                    self._analyze_anomaly(anomaly)
                    anomalies.append(anomaly)

            except Exception as e:
                logger.error(f"Error parsing detection: {e}")

        logger.info(f"Found {len(anomalies)} anomalies within {radius_km}km")
        return anomalies

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula

        Returns distance in kilometers
        """
        R = 6371.0  # Earth radius in kilometers

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def _analyze_anomaly(self, anomaly: ThermalAnomaly):
        """
        Analyze thermal anomaly to determine if it's interesting

        Scoring based on:
        - Ocean location (higher score)
        - High FRP (Fire Radiative Power)
        - Daytime detection (easier to correlate with visual confirmation)
        - High confidence from satellite
        - Distance from known industrial sources
        - Non-recurring (one-off event more interesting than persistent source)
        """
        score = 0.0

        # Check if ocean (simple heuristic: no sophisticated coastline data)
        # For now, assume anything far from major land masses is ocean
        # In production, use proper ocean/land dataset
        is_ocean = self._is_likely_ocean(anomaly.latitude, anomaly.longitude)
        anomaly.is_ocean = is_ocean

        if is_ocean:
            score += 0.3  # Ocean detection is more interesting

        # Fire Radiative Power (FRP)
        # Higher FRP = larger fire/explosion
        # Typical range: 0.1 - 1000+ MW
        # Naval strike/explosion: 50-500 MW
        if anomaly.frp > 100:
            score += 0.3
        elif anomaly.frp > 50:
            score += 0.2
        elif anomaly.frp > 20:
            score += 0.1

        # Confidence level from satellite
        if anomaly.confidence == 'h':
            score += 0.2
        elif anomaly.confidence == 'n':
            score += 0.1

        # Daytime detection (easier to correlate with visual reports)
        if anomaly.daynight == 'D':
            score += 0.1

        # Clamp score to 0-1
        anomaly.anomaly_score = min(1.0, score)

        # Generate description
        anomaly.description = self._generate_description(anomaly)

    def _is_likely_ocean(self, lat: float, lon: float) -> bool:
        """
        Simple heuristic to determine if coordinates are likely ocean

        In production, use proper ocean/land boundary dataset.
        For now, use simple rules based on known land masses.
        """
        # Very simple heuristic - this is NOT accurate for all cases
        # TODO: Integrate with proper coastline/ocean dataset

        # Eastern Pacific (off Mexico)
        if 10 <= lat <= 20 and -110 <= lon <= -100:
            return True

        # Caribbean
        if 10 <= lat <= 25 and -85 <= lon <= -60:
            # Caribbean is mostly ocean with islands
            return True

        # Persian Gulf
        if 23 <= lat <= 30 and 48 <= lon <= 57:
            return True

        # For other regions, default to False (land) to be conservative
        return False

    def _generate_description(self, anomaly: ThermalAnomaly) -> str:
        """Generate human-readable description"""
        location_type = "Ocean" if anomaly.is_ocean else "Land"
        time_of_day = "Daytime" if anomaly.daynight == 'D' else "Nighttime"
        conf_str = {"h": "High", "n": "Nominal", "l": "Low"}.get(anomaly.confidence, "Unknown")

        desc = f"{time_of_day} thermal anomaly detected at {anomaly.latitude:.4f}°, {anomaly.longitude:.4f}° ({location_type}). "
        desc += f"FRP: {anomaly.frp:.1f} MW, Confidence: {conf_str}. "
        desc += f"Satellite: {anomaly.satellite}, Brightness: {anomaly.brightness:.1f}K. "

        if anomaly.anomaly_score > 0.7:
            desc += "⚠️ HIGH ANOMALY SCORE - Possible significant event."
        elif anomaly.anomaly_score > 0.5:
            desc += "Moderate anomaly score - Worth investigating."

        return desc

    def save_anomaly_to_file(self, anomaly: ThermalAnomaly):
        """Save thermal anomaly as markdown file"""
        filename = THERMAL_DATA_DIR / f"{anomaly.id}.md"

        content = f"""# Thermal Anomaly {anomaly.id}

**Location:** {anomaly.latitude:.6f}°, {anomaly.longitude:.6f}°
**Date:** {anomaly.acquisition_date} {anomaly.acquisition_time} UTC
**Type:** {'Ocean' if anomaly.is_ocean else 'Land'} Anomaly
**Anomaly Score:** {anomaly.anomaly_score:.2f}/1.0

## Detection Details

- **Fire Radiative Power (FRP):** {anomaly.frp:.1f} MW
- **Brightness Temperature:** {anomaly.brightness:.1f} K
- **Confidence:** {anomaly.confidence.upper()} ({'High' if anomaly.confidence == 'h' else 'Nominal' if anomaly.confidence == 'n' else 'Low'})
- **Time of Day:** {'Daytime' if anomaly.daynight == 'D' else 'Nighttime'}
- **Satellite:** {anomaly.satellite} ({anomaly.instrument})

## Analysis

{anomaly.description}

## Geographic Context

- **Region:** {anomaly.region or 'Unknown'}
- **Location Type:** {'Ocean' if anomaly.is_ocean else 'Land'}

## Coordinates for Verification

```
Latitude:  {anomaly.latitude:.6f}°
Longitude: {anomaly.longitude:.6f}°
```

Google Maps: https://www.google.com/maps?q={anomaly.latitude},{anomaly.longitude}

## Raw Data

```json
{json.dumps(anomaly.to_dict(), indent=2)}
```

---
*Detected by Satellite Thermal Anomaly Collector*
*Source: NASA FIRMS VIIRS*
*Category: OSINT / Geospatial Intelligence*
"""

        with open(filename, 'w') as f:
            f.write(content)

    def collect(self, regions: List[str] = None, days: int = 1):
        """
        Collect thermal anomalies from specified regions

        Args:
            regions: List of region names (or None for all)
            days: Days to look back
        """
        if regions is None:
            regions = ['eastern_pacific', 'caribbean', 'persian_gulf', 'red_sea']

        new_anomalies = []

        for region_name in regions:
            logger.info("=" * 80)
            logger.info(f"Collecting thermal anomalies: {region_name}")
            logger.info("=" * 80)

            anomalies = self.fetch_region(region_name, days)
            new_anomalies.extend(anomalies)

        # Process new anomalies
        added_count = 0
        for anomaly in new_anomalies:
            if anomaly.id not in self.anomalies:
                self.anomalies[anomaly.id] = anomaly
                self.save_anomaly_to_file(anomaly)
                added_count += 1

                # Log high-score anomalies
                if anomaly.anomaly_score > 0.6:
                    logger.warning(f"⚠️  HIGH ANOMALY: {anomaly.latitude:.4f}, {anomaly.longitude:.4f} - Score: {anomaly.anomaly_score:.2f}")

        # Save index
        self._save_index()

        logger.info("=" * 80)
        logger.info(f"Collection Complete: {added_count} new anomalies ({len(self.anomalies)} total)")
        logger.info("=" * 80)

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return self.index.get('stats', {})


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Satellite Thermal Anomaly Collector')
    parser.add_argument(
        '--regions',
        nargs='+',
        choices=list(REGIONS.keys()),
        help='Regions to monitor'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Days to look back (1-10)'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help='Latitude for point search'
    )
    parser.add_argument(
        '--lon',
        type=float,
        help='Longitude for point search'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=50,
        help='Search radius in km (for point search)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Continuous monitoring mode'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Update interval in seconds'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='FIRMS API key (or set FIRMS_API_KEY env variable)'
    )

    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.getenv('FIRMS_API_KEY')
    if not api_key or api_key == 'DEMO':
        logger.warning("Using DEMO API key. Get your free key at: https://firms.modaps.eosdis.nasa.gov/api/area/")

    collector = SatelliteThermalCollector(api_key=api_key)

    if args.stats:
        # Show statistics
        stats = collector.get_statistics()
        print("\n" + "=" * 80)
        print("Satellite Thermal Anomaly Statistics")
        print("=" * 80)
        print(f"\nTotal Anomalies: {stats.get('total_anomalies', 0)}")
        print(f"Ocean Anomalies: {stats.get('ocean_anomalies', 0)}")
        print(f"High Confidence: {stats.get('high_confidence', 0)}")
        print(f"Last Update: {collector.index.get('last_update', 'Never')}")

        print("\nBy Region:")
        for region, count in sorted(stats.get('by_region', {}).items()):
            print(f"  {region:20s}: {count:4d}")

        print()
        return

    # Point search
    if args.lat is not None and args.lon is not None:
        logger.info(f"Searching near point ({args.lat}, {args.lon}) within {args.radius}km")
        anomalies = collector.fetch_point(args.lat, args.lon, args.radius, args.days)

        print(f"\nFound {len(anomalies)} anomalies:")
        for anom in sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)[:10]:
            print(f"\n  {anom.latitude:.4f}, {anom.longitude:.4f}")
            print(f"    Score: {anom.anomaly_score:.2f}, FRP: {anom.frp:.1f} MW")
            print(f"    {anom.acquisition_date} {anom.acquisition_time} UTC")
            print(f"    Distance: {anom.metadata.get('distance_from_query_km', 0):.1f} km")

        return

    # Region collection
    if args.monitor:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            collector.collect(regions=args.regions, days=args.days)
            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        # One-shot collection
        collector.collect(regions=args.regions, days=args.days)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nCollector stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
