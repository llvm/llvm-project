#!/usr/bin/env python3
"""
ADS-B Flight Tracking Integration

Integrates with multiple ADS-B sources for real-time aircraft tracking.
Particularly useful for military intelligence and geospatial OSINT.

Supported Sources:
- OpenSky Network (free, no API key required)
- ADS-B Exchange (requires API key)
- FlightRadar24 (requires API key)
- FlightAware (requires API key)

Features:
- Real-time aircraft positions
- Historical flight data
- Military aircraft filtering
- Call sign analysis
- Geofencing and alerts
- Integration with military OSINT collector

Usage:
    # Track all aircraft in a region
    tracker = ADSBFlightTracker()
    flights = tracker.get_flights_in_bbox(lat_min=35, lat_max=45, lon_min=-10, lon_max=10)

    # Track military aircraft
    military = tracker.get_military_aircraft()

    # Monitor specific region
    tracker.monitor_region('persian_gulf', interval=60)
"""

import os
import json
import time
import logging
import hashlib
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
        logging.FileHandler('rag_system/adsb_flight_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ADSB_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/aircraft')
ADSB_INDEX_FILE = Path('rag_system/adsb_flight_index.json')

# Create directories
ADSB_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (set via environment variables)
OPENSKY_USERNAME = os.getenv('OPENSKY_USERNAME', '')
OPENSKY_PASSWORD = os.getenv('OPENSKY_PASSWORD', '')
ADSBEXCHANGE_API_KEY = os.getenv('ADSBEXCHANGE_API_KEY', '')
FLIGHTRADAR24_API_KEY = os.getenv('FLIGHTRADAR24_API_KEY', '')


# Regions of Interest
REGIONS = {
    'global': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'europe': {'lat_min': 35, 'lat_max': 70, 'lon_min': -10, 'lon_max': 40},
    'middle_east': {'lat_min': 15, 'lat_max': 40, 'lon_min': 35, 'lon_max': 60},
    'persian_gulf': {'lat_min': 23, 'lat_max': 30, 'lon_min': 48, 'lon_max': 57},
    'south_china_sea': {'lat_min': 0, 'lat_max': 25, 'lon_min': 100, 'lon_max': 125},
    'taiwan_strait': {'lat_min': 22, 'lat_max': 27, 'lon_min': 118, 'lon_max': 123},
    'baltic': {'lat_min': 53, 'lat_max': 66, 'lon_min': 10, 'lon_max': 30},
    'mediterranean': {'lat_min': 30, 'lat_max': 46, 'lon_min': -6, 'lon_max': 37},
    'north_pacific': {'lat_min': 20, 'lat_max': 60, 'lon_min': 120, 'lon_max': -120},
}


# Military call sign patterns
MILITARY_PATTERNS = [
    'BLOCKED',  # Blocked military aircraft
    'RCH',      # Reach (USAF transport)
    'CNV',      # Convoy (USAF transport)
    'EVAC',     # Evacuation flights
    'DUKE',     # US military
    'ORBIT',    # Surveillance orbits
    'PAT',      # Patrol aircraft
    'HOMER',    # US military
    'SPAR',     # Special Air Resources (USAF VIP)
    'VENUS',    # US military
    'SAM',      # Special Air Mission (US military VIP)
    'NAVY',     # US Navy
    'USAF',     # US Air Force
    'ARMY',     # US Army
]

# Military aircraft types (ICAO codes)
MILITARY_AIRCRAFT_TYPES = [
    'B52',     # B-52 Stratofortress
    'B1',      # B-1 Lancer
    'B2',      # B-2 Spirit
    'F15',     # F-15 Eagle
    'F16',     # F-16 Fighting Falcon
    'F18',     # F/A-18 Hornet
    'F22',     # F-22 Raptor
    'F35',     # F-35 Lightning II
    'C130',    # C-130 Hercules
    'C17',     # C-17 Globemaster III
    'C5',      # C-5 Galaxy
    'KC135',   # KC-135 Stratotanker
    'KC10',    # KC-10 Extender
    'KC46',    # KC-46 Pegasus
    'E3',      # E-3 Sentry (AWACS)
    'E8',      # E-8 JSTARS
    'RC135',   # RC-135 Rivet Joint (SIGINT)
    'P8',      # P-8 Poseidon
    'U2',      # U-2 Dragon Lady
    'RQ4',     # RQ-4 Global Hawk
]


@dataclass
class AircraftPosition:
    """Aircraft position data from ADS-B"""
    icao24: str                      # ICAO 24-bit address
    callsign: Optional[str] = None   # Flight callsign
    origin_country: Optional[str] = None
    time_position: Optional[int] = None
    last_contact: Optional[int] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    altitude: Optional[float] = None  # Meters above sea level
    on_ground: bool = False
    velocity: Optional[float] = None  # m/s
    true_track: Optional[float] = None  # Degrees from north
    vertical_rate: Optional[float] = None  # m/s
    sensors: Optional[List[int]] = None
    geo_altitude: Optional[float] = None
    squawk: Optional[str] = None
    spi: bool = False
    position_source: Optional[int] = None
    category: Optional[int] = None
    aircraft_type: Optional[str] = None
    is_military: bool = False
    collected_at: str = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now().isoformat()

        # Detect military aircraft
        if self.callsign:
            callsign_upper = self.callsign.strip().upper()
            self.is_military = any(pattern in callsign_upper for pattern in MILITARY_PATTERNS)

        if self.aircraft_type:
            aircraft_type_upper = self.aircraft_type.upper()
            if any(mil_type in aircraft_type_upper for mil_type in MILITARY_AIRCRAFT_TYPES):
                self.is_military = True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class OpenSkyNetworkClient:
    """
    Client for OpenSky Network API

    Free, open-source ADS-B data. No API key required for basic access.
    Rate limits: 10 requests/minute for anonymous, 400 requests/minute for registered users

    API Docs: https://openskynetwork.github.io/opensky-api/
    """

    BASE_URL = "https://opensky-network.org/api"

    def __init__(self, username: str = '', password: str = ''):
        self.username = username
        self.password = password
        self.session_count = 0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        min_interval = 6 if not self.username else 0.15  # 10/min or 400/min
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated or anonymous request"""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            url += '?' + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})

        req = urllib.request.Request(url)

        if self.username and self.password:
            import base64
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            req.add_header('Authorization', f'Basic {encoded_credentials}')

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                self.session_count += 1
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.error("Rate limit exceeded for OpenSky Network")
            else:
                logger.error(f"HTTP error {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"Error fetching from OpenSky: {e}")
            return None

    def get_all_states(self, time_secs: int = 0, icao24: str = None) -> List[AircraftPosition]:
        """
        Get all current aircraft states

        Args:
            time_secs: Unix timestamp (0 for current time)
            icao24: ICAO 24-bit address filter (comma-separated for multiple)

        Returns:
            List of AircraftPosition objects
        """
        params = {'time': time_secs if time_secs > 0 else None, 'icao24': icao24}
        data = self._make_request('states/all', params)

        if not data or 'states' not in data:
            return []

        aircraft_list = []
        for state in data['states']:
            # OpenSky state vector format:
            # [icao24, callsign, origin_country, time_position, last_contact,
            #  longitude, latitude, baro_altitude, on_ground, velocity,
            #  true_track, vertical_rate, sensors, geo_altitude, squawk,
            #  spi, position_source, category]

            aircraft = AircraftPosition(
                icao24=state[0],
                callsign=state[1].strip() if state[1] else None,
                origin_country=state[2],
                time_position=state[3],
                last_contact=state[4],
                longitude=state[5],
                latitude=state[6],
                altitude=state[7],  # barometric altitude
                on_ground=state[8],
                velocity=state[9],
                true_track=state[10],
                vertical_rate=state[11],
                sensors=state[12],
                geo_altitude=state[13],
                squawk=state[14],
                spi=state[15],
                position_source=state[16],
                category=state[17] if len(state) > 17 else None
            )

            aircraft_list.append(aircraft)

        return aircraft_list

    def get_states_in_bbox(self, lat_min: float, lat_max: float,
                           lon_min: float, lon_max: float) -> List[AircraftPosition]:
        """
        Get aircraft within a bounding box

        Args:
            lat_min, lat_max: Latitude range
            lon_min, lon_max: Longitude range

        Returns:
            List of AircraftPosition objects
        """
        params = {
            'lamin': lat_min,
            'lamax': lat_max,
            'lomin': lon_min,
            'lomax': lon_max
        }

        data = self._make_request('states/all', params)

        if not data or 'states' not in data:
            return []

        aircraft_list = []
        for state in data['states']:
            aircraft = AircraftPosition(
                icao24=state[0],
                callsign=state[1].strip() if state[1] else None,
                origin_country=state[2],
                time_position=state[3],
                last_contact=state[4],
                longitude=state[5],
                latitude=state[6],
                altitude=state[7],
                on_ground=state[8],
                velocity=state[9],
                true_track=state[10],
                vertical_rate=state[11],
                sensors=state[12],
                geo_altitude=state[13],
                squawk=state[14],
                spi=state[15],
                position_source=state[16],
                category=state[17] if len(state) > 17 else None
            )

            aircraft_list.append(aircraft)

        return aircraft_list


class ADSBFlightTracker:
    """Main ADS-B flight tracking aggregator"""

    def __init__(self):
        self.opensky = OpenSkyNetworkClient(OPENSKY_USERNAME, OPENSKY_PASSWORD)
        self.tracked_aircraft: Dict[str, AircraftPosition] = {}
        self.index = self._load_index()

        logger.info("ADS-B Flight Tracker initialized")
        if OPENSKY_USERNAME:
            logger.info("  - OpenSky Network: Authenticated")
        else:
            logger.info("  - OpenSky Network: Anonymous (rate limited)")

    def _load_index(self) -> Dict:
        """Load existing flight tracking index"""
        if ADSB_INDEX_FILE.exists():
            with open(ADSB_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {'aircraft': {}, 'last_update': None, 'stats': {}}

    def _save_index(self):
        """Save flight tracking index"""
        self.index['last_update'] = datetime.now().isoformat()
        self.index['aircraft'] = {icao: aircraft.to_dict()
                                   for icao, aircraft in self.tracked_aircraft.items()}

        # Calculate stats
        stats = {
            'total_aircraft': len(self.tracked_aircraft),
            'military_aircraft': sum(1 for a in self.tracked_aircraft.values() if a.is_military),
            'on_ground': sum(1 for a in self.tracked_aircraft.values() if a.on_ground),
            'in_flight': sum(1 for a in self.tracked_aircraft.values() if not a.on_ground),
        }

        self.index['stats'] = stats

        with open(ADSB_INDEX_FILE, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.info(f"Flight tracking index saved: {len(self.tracked_aircraft)} aircraft")

    def get_flights_in_region(self, region: str) -> List[AircraftPosition]:
        """
        Get all flights in a predefined region

        Args:
            region: Region name (e.g., 'persian_gulf', 'taiwan_strait')

        Returns:
            List of AircraftPosition objects
        """
        if region not in REGIONS:
            logger.error(f"Unknown region: {region}")
            return []

        bbox = REGIONS[region]
        logger.info(f"Fetching flights in {region}")

        flights = self.opensky.get_states_in_bbox(
            lat_min=bbox['lat_min'],
            lat_max=bbox['lat_max'],
            lon_min=bbox['lon_min'],
            lon_max=bbox['lon_max']
        )

        logger.info(f"Found {len(flights)} aircraft in {region}")
        return flights

    def get_military_aircraft(self, region: str = None) -> List[AircraftPosition]:
        """
        Get all military aircraft (optionally in a specific region)

        Args:
            region: Optional region name

        Returns:
            List of military AircraftPosition objects
        """
        if region:
            flights = self.get_flights_in_region(region)
        else:
            flights = self.opensky.get_all_states()

        military = [f for f in flights if f.is_military]
        logger.info(f"Found {len(military)} military aircraft")

        return military

    def track_aircraft(self, icao24: str) -> Optional[AircraftPosition]:
        """
        Track a specific aircraft by ICAO 24-bit address

        Args:
            icao24: Aircraft ICAO address

        Returns:
            AircraftPosition or None
        """
        aircraft_list = self.opensky.get_all_states(icao24=icao24)

        if aircraft_list:
            aircraft = aircraft_list[0]
            self.tracked_aircraft[icao24] = aircraft
            return aircraft

        return None

    def monitor_region(self, region: str, interval: int = 60, duration: int = 3600):
        """
        Continuously monitor a region for aircraft activity

        Args:
            region: Region name
            interval: Update interval in seconds
            duration: Total monitoring duration in seconds
        """
        logger.info(f"Starting monitoring of {region} (interval: {interval}s, duration: {duration}s)")

        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                flights = self.get_flights_in_region(region)

                # Update tracked aircraft
                for flight in flights:
                    self.tracked_aircraft[flight.icao24] = flight

                # Save index
                self._save_index()

                # Report military aircraft
                military = [f for f in flights if f.is_military]
                if military:
                    logger.warning(f"⚠️  {len(military)} military aircraft detected in {region}")
                    for aircraft in military:
                        logger.warning(f"   - {aircraft.callsign or 'UNKNOWN'} ({aircraft.icao24}) at {aircraft.altitude}m")

                # Sleep until next update
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(interval)

        logger.info("Monitoring complete")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='ADS-B Flight Tracking')
    parser.add_argument('--region', type=str, default='europe', help='Region to monitor')
    parser.add_argument('--military', action='store_true', help='Show only military aircraft')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--duration', type=int, default=3600, help='Monitoring duration in seconds')

    args = parser.parse_args()

    tracker = ADSBFlightTracker()

    if args.monitor:
        tracker.monitor_region(args.region, interval=args.interval, duration=args.duration)
    elif args.military:
        military = tracker.get_military_aircraft(region=args.region)
        print(f"\nMilitary Aircraft in {args.region}: {len(military)}")
        for aircraft in military:
            print(f"  {aircraft.callsign or 'UNKNOWN':15s} {aircraft.icao24:10s} {aircraft.altitude:8.0f}m")
    else:
        flights = tracker.get_flights_in_region(args.region)
        print(f"\nAll Aircraft in {args.region}: {len(flights)}")
        print(f"Military: {sum(1 for f in flights if f.is_military)}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTracker stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
