#!/usr/bin/env python3
"""
Military & Geopolitical OSINT Collector

Specialized OSINT collection for military, defense, and geopolitical intelligence.
Focus on SIGINT/COMINT-adjacent open-source data:

Data Sources:
1. Military News & Press Releases
   - DoD Press Releases, Pentagon
   - NATO, UN Security Council
   - Defense News, Jane's, Military.com
   - Stars and Stripes, Task & Purpose

2. Aircraft Tracking (ADS-B)
   - Military aircraft movements
   - Tanker/refueling operations (SIGINT indicator)
   - Transport aircraft (troop movements)
   - Surveillance aircraft (SIGINT/ELINT platforms)
   - Notable call signs (BLOCKED, military codes)

3. Naval Vessel Tracking (AIS)
   - Warship movements
   - Aircraft carrier positions
   - Submarine tender activity
   - Naval exercises
   - Port visits

4. Conflict & Crisis Monitoring
   - ACLED (Armed Conflict Location & Event Data)
   - GDELT (Global Database of Events, Language, and Tone)
   - Sanctions databases (OFAC, EU, UN)
   - Arms trade monitoring

5. Military Infrastructure
   - Satellite imagery analysis (commercial)
   - Military base activity
   - Nuclear facilities (IAEA reports)
   - Missile test monitoring

6. Diplomatic & Intelligence
   - Embassy closures/openings
   - Diplomatic cables (WikiLeaks, etc.)
   - Think tank analysis (CSIS, RUSI, IISS)
   - Intelligence community press releases

Usage:
    # Collect military news
    python3 military_osint_collector.py --sources military_news

    # Track military aircraft
    python3 military_osint_collector.py --sources aircraft --regions all

    # Monitor naval vessels
    python3 military_osint_collector.py --sources ships --regions persian_gulf

    # Comprehensive collection
    python3 military_osint_collector.py --sources all --monitor --interval 3600

    # Export intelligence report
    python3 military_osint_collector.py --export markdown --output military_intel.md
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import re

# Setup logging
LOG_DIR = Path("rag_system")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "military_osint.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MILITARY_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/military')
AIRCRAFT_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/aircraft')
NAVAL_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/naval')
CONFLICT_DATA_DIR = Path('00-documentation/Security_Feed/OSINT/conflicts')

MILITARY_INDEX_FILE = Path('rag_system/military_osint_index.json')
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_AIRCRAFT_FILE = SAMPLE_DATA_DIR / "aircraft_tracks.json"
SAMPLE_NAVAL_FILE = SAMPLE_DATA_DIR / "naval_tracks.json"

# Create directories
for dir in [MILITARY_DATA_DIR, AIRCRAFT_DATA_DIR, NAVAL_DATA_DIR, CONFLICT_DATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

try:
    from adsb_flight_tracker import ADSBFlightTracker, AircraftPosition
except ImportError:
    ADSBFlightTracker = None
    AircraftPosition = None


# Military News RSS Feeds
MILITARY_NEWS_FEEDS = {
    "DoD_News": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?max=10&ContentType=1&Site=945",
    "Pentagon_Releases": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?max=10&ContentType=9&Site=945",
    "US_Navy_News": "https://www.navy.mil/RSS/rss_news.xml",
    "US_Air_Force": "https://www.af.mil/DesktopModules/ArticleCS/RSS.ashx",
    "US_Army_News": "https://www.army.mil/rss/home.xml",
    "Defense_News": "https://www.defensenews.com/arc/outboundfeeds/rss/",
    "Stars_and_Stripes": "https://www.stripes.com/rss/news",
    "Military_Times": "https://www.militarytimes.com/arc/outboundfeeds/rss/",
    "Jane's_360": "https://www.janes.com/feeds/news",
    "CSIS_Analysis": "https://www.csis.org/rss/analysis-feed",
    "ISW_Updates": "https://www.understandingwar.org/rss.xml",  # Institute for Study of War
    "War_on_the_Rocks": "https://warontherocks.com/feed/",
    "Bellingcat": "https://www.bellingcat.com/feed/",
}

# Conflict & Crisis Monitoring
CONFLICT_FEEDS = {
    "ACLED_API": "https://api.acleddata.com/acled/read",  # Requires API key
    "GDELT_Events": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
    "Global_Terrorism_DB": "https://www.start.umd.edu/gtd/",
}

# Diplomatic & Intelligence
INTEL_FEEDS = {
    "CIA_News": "https://www.cia.gov/rss/press-releases-statements.rss",
    "FBI_Press": "https://www.fbi.gov/feeds/fbi-news/all",
    "NSA_News": "https://www.nsa.gov/portals/75/documents/news-features/rss-feeds/news-release.xml",
    "ODNI_Releases": "https://www.dni.gov/index.php/newsroom/rss-feeds",
}

# Arms & Sanctions
SANCTIONS_SOURCES = {
    "OFAC_Sanctions": "https://sanctionssearch.ofac.treas.gov/",  # Web scraping
    "EU_Sanctions": "https://www.sanctionsmap.eu/",
    "UN_Security_Council": "https://www.un.org/securitycouncil/content/un-sc-consolidated-list",
}


@dataclass
class MilitaryIntelItem:
    """Military/geopolitical intelligence item"""
    id: str                          # Unique identifier
    title: str                       # Title/summary
    description: str                 # Full description
    source: str                      # Source name
    intel_type: str                  # Type: news, aircraft, ship, conflict, sanctions
    url: Optional[str] = None        # Source URL
    published: Optional[str] = None  # Publication time
    collected: str = None            # Collection time
    classification: str = "UNCLASS"  # Classification (all are unclassified OSINT)
    tags: List[str] = None           # Tags/categories
    geolocation: Dict = None         # Lat/lon if applicable
    entities: Dict = None            # Named entities (countries, units, weapons)
    metadata: Dict = None            # Additional metadata

    def __post_init__(self):
        if self.collected is None:
            self.collected = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.geolocation is None:
            self.geolocation = {}
        if self.entities is None:
            self.entities = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MilitaryNewsCollector:
    """Collect military news from RSS feeds"""

    def __init__(self):
        self.items: List[MilitaryIntelItem] = []

    def fetch_feed(self, name: str, url: str) -> List[MilitaryIntelItem]:
        """Fetch and parse military news RSS feed"""
        logger.info(f"Fetching military news: {name}")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Detect RSS vs Atom
            if root.tag == 'rss':
                return self._parse_rss(name, root)
            elif root.tag.endswith('feed'):  # Atom
                return self._parse_atom(name, root)
            else:
                logger.warning(f"Unknown feed format for {name}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch feed {name}: {e}")
            return []

    def _parse_rss(self, source: str, root: ET.Element) -> List[MilitaryIntelItem]:
        """Parse RSS 2.0 feed"""
        items = []

        for item in root.findall('.//item'):
            try:
                title = item.findtext('title', '').strip()
                description = item.findtext('description', '').strip()
                link = item.findtext('link', '').strip()
                pub_date = item.findtext('pubDate', '').strip()

                # Create unique ID
                item_id = hashlib.sha256(f"{source}:{link}:{title}".encode()).hexdigest()[:16]

                # Extract entities (countries, military units, weapons)
                entities = self._extract_entities(title + ' ' + description)

                # Extract tags
                tags = self._extract_military_tags(title, description)

                # Determine if this is high-priority intelligence
                priority = self._assess_priority(title, description)

                intel_item = MilitaryIntelItem(
                    id=item_id,
                    title=title,
                    description=description,
                    source=source,
                    intel_type='military_news',
                    url=link,
                    published=pub_date,
                    tags=tags,
                    entities=entities,
                    metadata={'priority': priority}
                )

                items.append(intel_item)

            except Exception as e:
                logger.error(f"Error parsing RSS item: {e}")

        return items

    def _parse_atom(self, source: str, root: ET.Element) -> List[MilitaryIntelItem]:
        """Parse Atom feed"""
        items = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            try:
                title = entry.findtext('atom:title', '', ns).strip()
                content = entry.findtext('atom:content', '', ns).strip()
                if not content:
                    content = entry.findtext('atom:summary', '', ns).strip()

                link_elem = entry.find('atom:link', ns)
                link = link_elem.get('href', '') if link_elem is not None else ''

                published = entry.findtext('atom:published', '', ns).strip()

                item_id = hashlib.sha256(f"{source}:{link}:{title}".encode()).hexdigest()[:16]

                entities = self._extract_entities(title + ' ' + content)
                tags = self._extract_military_tags(title, content)
                priority = self._assess_priority(title, content)

                intel_item = MilitaryIntelItem(
                    id=item_id,
                    title=title,
                    description=content,
                    source=source,
                    intel_type='military_news',
                    url=link,
                    published=published,
                    tags=tags,
                    entities=entities,
                    metadata={'priority': priority}
                )

                items.append(intel_item)

            except Exception as e:
                logger.error(f"Error parsing Atom entry: {e}")

        return items

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (countries, units, weapons, etc.)"""
        entities = {
            'countries': [],
            'military_units': [],
            'weapons': [],
            'locations': [],
            'people': []
        }

        # Countries (simple keyword matching - in production use NER)
        country_patterns = {
            'russia': 'Russia',
            'ukraine': 'Ukraine',
            'china': 'China',
            'iran': 'Iran',
            'north korea': 'North Korea',
            'syria': 'Syria',
            'afghanistan': 'Afghanistan',
            'iraq': 'Iraq',
            'israel': 'Israel',
            'taiwan': 'Taiwan',
            'saudi arabia': 'Saudi Arabia',
        }

        text_lower = text.lower()
        for pattern, country in country_patterns.items():
            if pattern in text_lower:
                entities['countries'].append(country)

        # Military units
        unit_patterns = [
            r'(\d+(?:st|nd|rd|th)\s+(?:Infantry|Cavalry|Armored|Marine|Airborne)\s+Division)',
            r'(USS\s+\w+)',
            r'(HMS\s+\w+)',
            r'(\w+\s+Carrier\s+Strike\s+Group)',
        ]

        for pattern in unit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['military_units'].extend(matches)

        # Weapon systems
        weapon_keywords = [
            'F-35', 'F-22', 'B-52', 'B-21', 'F-16', 'F-18',
            'Patriot', 'THAAD', 'Aegis',
            'Javelin', 'HIMARS', 'M1 Abrams',
            'S-400', 'S-300', 'Iskander',
            'Kinzhal', 'Kalibr', 'Kh-47',
            'J-20', 'J-16', 'DF-21', 'DF-26'
        ]

        for weapon in weapon_keywords:
            if weapon.lower() in text_lower:
                entities['weapons'].append(weapon)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        # Remove empty lists
        return {k: v for k, v in entities.items() if v}

    def _extract_military_tags(self, title: str, description: str) -> List[str]:
        """Extract military-specific tags"""
        tags = []
        text = (title + ' ' + description).lower()

        # Military operation tags
        operation_keywords = {
            'strike': 'air_strike',
            'airstrike': 'air_strike',
            'bombing': 'air_strike',
            'exercise': 'military_exercise',
            'drill': 'military_exercise',
            'deployment': 'deployment',
            'withdraw': 'withdrawal',
            'troop movement': 'troop_movement',
            'carrier': 'naval_deployment',
            'warship': 'naval_activity',
            'submarine': 'submarine_activity',
            'missile test': 'missile_test',
            'nuclear': 'nuclear',
            'sanctions': 'sanctions',
            'arms deal': 'arms_trade',
            'weapons sale': 'arms_trade',
            'cyberattack': 'cyber',
            'espionage': 'espionage',
            'intelligence': 'intelligence',
        }

        for keyword, tag in operation_keywords.items():
            if keyword in text:
                tags.append(tag)

        # Theater tags
        theaters = {
            'pacific': 'indo_pacific',
            'south china sea': 'indo_pacific',
            'taiwan': 'taiwan_strait',
            'ukraine': 'ukraine',
            'middle east': 'middle_east',
            'persian gulf': 'persian_gulf',
            'strait of hormuz': 'strait_of_hormuz',
            'red sea': 'red_sea',
            'baltic': 'baltic',
            'arctic': 'arctic',
        }

        for keyword, tag in theaters.items():
            if keyword in text:
                tags.append(tag)

        return list(set(tags))

    def _assess_priority(self, title: str, description: str) -> str:
        """Assess intelligence priority (HIGH/MEDIUM/LOW)"""
        text = (title + ' ' + description).lower()

        # High priority indicators
        high_priority_keywords = [
            'nuclear', 'missile test', 'strike', 'airstrike',
            'carrier deployment', 'invasion', 'conflict',
            'war', 'attack', 'incident'
        ]

        # Medium priority indicators
        medium_priority_keywords = [
            'exercise', 'drill', 'deployment', 'military',
            'sanctions', 'arms deal', 'treaty'
        ]

        if any(kw in text for kw in high_priority_keywords):
            return 'HIGH'
        elif any(kw in text for kw in medium_priority_keywords):
            return 'MEDIUM'
        else:
            return 'LOW'

    def collect_all(self) -> List[MilitaryIntelItem]:
        """Collect from all military news feeds"""
        all_items = []

        for name, url in MILITARY_NEWS_FEEDS.items():
            items = self.fetch_feed(name, url)
            all_items.extend(items)
            logger.info(f"Collected {len(items)} items from {name}")
            time.sleep(1)  # Rate limiting

        return all_items


class AircraftTrackingCollector:
    """
    Collect military aircraft tracking data

    Uses public ADS-B data to track military aircraft movements.
    Key indicators:
    - Tanker/refueling aircraft (KC-135, KC-46, KC-10) = SIGINT mission support
    - AWACS (E-3, E-7) = surveillance operations
    - SIGINT aircraft (RC-135, E-8 JSTARS, P-8 Poseidon)
    - Strategic bombers (B-52, B-1, B-2) = readiness/deterrence
    - Transport aircraft (C-17, C-130) = troop/equipment movements
    """

    PLATFORM_PRIORITY = {
        "b52": "HIGH",
        "b1": "HIGH",
        "b2": "HIGH",
        "kc135": "HIGH",
        "kc46": "HIGH",
        "kc10": "HIGH",
        "e3": "HIGH",
        "e7": "HIGH",
        "rc135": "HIGH",
        "p8": "MEDIUM",
        "c17": "MEDIUM",
        "c130": "MEDIUM",
        "rq4": "MEDIUM",
    }

    def __init__(self):
        self.items: List[MilitaryIntelItem] = []

    def fetch_military_aircraft(self, region: str = 'all') -> List[MilitaryIntelItem]:
        """Collect military aircraft data from live sources or curated samples"""
        collected: List[MilitaryIntelItem] = []

        # Attempt live ADS-B collection first
        if ADSBFlightTracker:
            try:
                tracker = ADSBFlightTracker()
                flights = tracker.get_military_aircraft(None if region == 'all' else region)

                for flight in flights:
                    item = self._flight_to_intel_item(flight, region)
                    if item:
                        collected.append(item)

                if collected:
                    logger.info(f"Collected {len(collected)} military flights via ADS-B tracker")
                    self.items = collected
                    return collected
            except Exception as exc:
                logger.warning(f"Live ADS-B collection failed ({exc}); falling back to curated data")

        # Fallback to curated sample data
        fallback_items = self._load_sample_aircraft(region)
        self.items = fallback_items
        logger.info(f"Loaded {len(fallback_items)} flights from curated dataset")
        return fallback_items

    def _flight_to_intel_item(self, flight: AircraftPosition, region: str) -> Optional[MilitaryIntelItem]:
        """Convert AircraftPosition to MilitaryIntelItem"""
        if not flight.latitude or not flight.longitude:
            return None

        flight_region = region if region != 'all' else 'global'
        platform = (flight.aircraft_type or "military_aircraft").upper()
        priority = self._assess_priority(platform)

        tags = ['aircraft', platform.lower()]
        if flight_region:
            tags.append(flight_region)
        if flight.is_military:
            tags.append('military')

        entities = {
            'aircraft': [platform],
            'operators': [flight.origin_country or "Unknown"],
        }
        if flight.callsign:
            entities.setdefault('call_signs', []).append(flight.callsign.strip())

        description = (
            f"{flight.callsign or 'Unidentified aircraft'} ({platform}) "
            f"tracked at {flight.latitude:.2f}, {flight.longitude:.2f} "
            f"alt {flight.altitude or 0:.0f}m speed {flight.velocity or 0:.0f} m/s."
        )

        metadata = {
            'priority': priority,
            'region': flight_region,
            'altitude_m': flight.altitude,
            'velocity_m_s': flight.velocity,
            'origin_country': flight.origin_country,
            'icao24': flight.icao24,
            'callsign': flight.callsign,
        }

        return MilitaryIntelItem(
            id=f"air_{flight.icao24}_{flight.last_contact}",
            title=f"Military Flight {flight.callsign or flight.icao24}",
            description=description,
            source="ADS-B Tracking",
            intel_type="air_activity",
            tags=tags,
            geolocation={'latitude': flight.latitude, 'longitude': flight.longitude},
            entities=entities,
            metadata=metadata
        )

    def _load_sample_aircraft(self, region: str) -> List[MilitaryIntelItem]:
        """Load curated aircraft tracks from repository"""
        if not SAMPLE_AIRCRAFT_FILE.exists():
            return []

        with open(SAMPLE_AIRCRAFT_FILE) as f:
            data = json.load(f)

        items: List[MilitaryIntelItem] = []
        for record in data:
            if region != 'all' and record.get('region') != region:
                continue
            lat = record.get('latitude')
            lon = record.get('longitude')
            if lat is None or lon is None:
                continue

            platform = record.get('type', 'Military Aircraft')
            priority = self._assess_priority(platform)
            tags = ['aircraft', record.get('region', 'global')]
            if record.get('mission'):
                tags.append(record['mission'].replace(' ', '_').lower())

            entities = {
                'aircraft': [platform],
                'operators': [record.get('operator', 'Unknown')],
            }
            if record.get('callsign'):
                entities['call_signs'] = [record['callsign']]
            if record.get('origin_country'):
                entities['countries'] = [record['origin_country']]

            metadata = {
                'priority': priority,
                'region': record.get('region'),
                'mission': record.get('mission'),
                'speed_kts': record.get('speed_kts'),
                'altitude_ft': record.get('altitude_ft'),
                'departure': record.get('departure'),
                'destination': record.get('destination'),
                'timestamp': record.get('timestamp'),
                'source': record.get('source', 'Curated ADS-B Snapshot')
            }

            items.append(MilitaryIntelItem(
                id=record.get('id', f"air_sample_{record.get('icao24', record.get('callsign', len(items)))}"),
                title=record.get('title') or f"{record.get('callsign', platform)} near {record.get('region', 'global').replace('_', ' ').title()}",
                description=record.get('summary') or record.get('description') or
                            f"{platform} operated by {record.get('operator', 'unknown operator')} "
                            f"observed at {lat:.2f}, {lon:.2f}",
                source=record.get('source', 'Curated ADS-B Snapshot'),
                intel_type="air_activity",
                url=record.get('source_url'),
                published=record.get('timestamp'),
                tags=tags,
                geolocation={'latitude': lat, 'longitude': lon},
                entities=entities,
                metadata=metadata
            ))

        return items

    def _assess_priority(self, platform: str) -> str:
        """Determine priority score based on aircraft type"""
        platform_key = platform.lower().replace('-', '')
        for key, priority in self.PLATFORM_PRIORITY.items():
            if key in platform_key:
                return priority
        return 'LOW'


class NavalTrackingCollector:
    """
    Collect naval vessel tracking data

    Uses AIS (Automatic Identification System) data for surface ships.
    Key indicators:
    - Aircraft carriers
    - Amphibious assault ships
    - Destroyers/cruisers
    - Submarine tenders (submarine activity nearby)
    - Naval exercises (multiple vessels converging)
    """

    PRIORITY_TYPES = {
        'aircraft carrier': 'HIGH',
        'amphibious assault': 'HIGH',
        'destroyer': 'MEDIUM',
        'frigate': 'MEDIUM',
        'submarine': 'HIGH',
        'tanker': 'LOW'
    }

    def __init__(self):
        self.items: List[MilitaryIntelItem] = []

    def fetch_naval_vessels(self, region: str = 'all') -> List[MilitaryIntelItem]:
        """Collect naval vessel data from curated AIS snapshots"""
        if not SAMPLE_NAVAL_FILE.exists():
            logger.warning("No curated naval tracking data available")
            return []

        with open(SAMPLE_NAVAL_FILE) as f:
            data = json.load(f)

        items: List[MilitaryIntelItem] = []
        for record in data:
            if region != 'all' and record.get('region') != region:
                continue

            lat = record.get('latitude')
            lon = record.get('longitude')
            if lat is None or lon is None:
                continue

            vessel_type = record.get('type', 'Naval Vessel')
            priority = self._assess_priority(vessel_type)
            tags = ['naval', record.get('region', 'global')]
            if record.get('task_group'):
                tags.append(record['task_group'].replace(' ', '_').lower())

            entities = {
                'vessels': [record.get('name', vessel_type)],
                'countries': [record.get('flag', 'Unknown')],
            }
            if record.get('task_group'):
                entities['task_groups'] = [record['task_group']]

            metadata = {
                'priority': priority,
                'region': record.get('region'),
                'speed_kts': record.get('speed_kts'),
                'heading': record.get('heading'),
                'status': record.get('status'),
                'task_group': record.get('task_group'),
                'timestamp': record.get('timestamp'),
                'notes': record.get('notes'),
                'source': record.get('source', 'Curated AIS Snapshot'),
            }

            description = record.get('summary') or (
                f"{record.get('name', vessel_type)} ({vessel_type}) "
                f"flagged to {record.get('flag', 'unknown nation')} "
                f"reported at {lat:.2f}, {lon:.2f} heading {record.get('heading', 0)}°."
            )

            items.append(MilitaryIntelItem(
                id=record.get('id', f"ship_{record.get('mmsi', len(items))}"),
                title=record.get('title') or f"Naval Activity: {record.get('name', vessel_type)}",
                description=description,
                source=record.get('source', 'Curated AIS Snapshot'),
                intel_type="naval_activity",
                url=record.get('source_url'),
                tags=tags,
                geolocation={'latitude': lat, 'longitude': lon},
                entities=entities,
                metadata=metadata
            ))

        self.items = items
        logger.info(f"Loaded {len(items)} naval tracks from curated dataset")
        return items

    def _assess_priority(self, vessel_type: str) -> str:
        vessel_key = vessel_type.lower()
        for key, priority in self.PRIORITY_TYPES.items():
            if key in vessel_key:
                return priority
        return 'LOW'


class MilitaryOSINTAggregator:
    """Main military/geopolitical OSINT aggregator"""

    def __init__(self):
        self.military_news_collector = MilitaryNewsCollector()
        self.aircraft_collector = AircraftTrackingCollector()
        self.naval_collector = NavalTrackingCollector()

        self.all_items: Dict[str, MilitaryIntelItem] = {}
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load existing military OSINT index"""
        if MILITARY_INDEX_FILE.exists():
            with open(MILITARY_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {'items': {}, 'last_update': None, 'stats': {}}

    def _save_index(self):
        """Save military OSINT index"""
        self.index['last_update'] = datetime.now().isoformat()
        self.index['items'] = {id: item.to_dict() for id, item in self.all_items.items()}

        # Calculate stats
        stats = {
            'total_items': len(self.all_items),
            'by_intel_type': {},
            'by_priority': {},
            'by_country': {},
            'by_tag': {}
        }

        for item in self.all_items.values():
            # By intel type
            intel_type = item.intel_type
            stats['by_intel_type'][intel_type] = stats['by_intel_type'].get(intel_type, 0) + 1

            # By priority
            priority = item.metadata.get('priority', 'UNKNOWN')
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1

            # By country
            for country in item.entities.get('countries', []):
                stats['by_country'][country] = stats['by_country'].get(country, 0) + 1

            # By tag
            for tag in item.tags:
                stats['by_tag'][tag] = stats['by_tag'].get(tag, 0) + 1

        self.index['stats'] = stats

        with open(MILITARY_INDEX_FILE, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.info(f"Military OSINT index saved: {len(self.all_items)} items")

    def _save_item_to_file(self, item: MilitaryIntelItem):
        """Save military intel item as markdown file"""
        # Organize by type
        type_dir = MILITARY_DATA_DIR / item.intel_type
        type_dir.mkdir(parents=True, exist_ok=True)

        filename = type_dir / f"{item.id}.md"

        content = f"""# {item.title}

**Source:** {item.source}
**Type:** {item.intel_type}
**Priority:** {item.metadata.get('priority', 'UNKNOWN')}
**Classification:** {item.classification}
**Published:** {item.published or 'Unknown'}
**Collected:** {item.collected}

## Description

{item.description}

"""

        if item.url:
            content += f"## Source URL\n\n{item.url}\n\n"

        if item.entities:
            content += "## Named Entities\n\n"
            for entity_type, entity_list in item.entities.items():
                content += f"### {entity_type.replace('_', ' ').title()}\n\n"
                for entity in entity_list:
                    content += f"- {entity}\n"
                content += "\n"

        if item.tags:
            content += f"## Tags\n\n{', '.join(item.tags)}\n\n"

        if item.geolocation:
            content += "## Geolocation\n\n```json\n"
            content += json.dumps(item.geolocation, indent=2)
            content += "\n```\n\n"

        if item.metadata:
            content += "## Metadata\n\n```json\n"
            content += json.dumps(item.metadata, indent=2)
            content += "\n```\n\n"

        content += f"""
---
*Collected by Military OSINT Aggregator*
*Category:* Military Intelligence / {item.intel_type.replace('_', ' ').title()}
*Classification:* {item.classification}
"""

        with open(filename, 'w') as f:
            f.write(content)

    def collect(self, sources: List[str] = None):
        """
        Collect military OSINT from specified sources

        Args:
            sources: List of source types ['military_news', 'aircraft', 'ships', 'all']
        """
        if sources is None or 'all' in sources:
            sources = ['military_news', 'aircraft', 'ships']

        new_items = []

        # Military News
        if 'military_news' in sources:
            logger.info("=" * 80)
            logger.info("Collecting Military News")
            logger.info("=" * 80)
            news_items = self.military_news_collector.collect_all()
            new_items.extend(news_items)

        # Aircraft Tracking
        if 'aircraft' in sources:
            logger.info("=" * 80)
            logger.info("Collecting Aircraft Tracking Data")
            logger.info("=" * 80)
            aircraft_items = self.aircraft_collector.fetch_military_aircraft()
            new_items.extend(aircraft_items)

        # Naval Tracking
        if 'ships' in sources:
            logger.info("=" * 80)
            logger.info("Collecting Naval Vessel Tracking Data")
            logger.info("=" * 80)
            ship_items = self.naval_collector.fetch_naval_vessels()
            new_items.extend(ship_items)

        # Process new items
        added_count = 0
        for item in new_items:
            if item.id not in self.all_items:
                self.all_items[item.id] = item
                self._save_item_to_file(item)
                added_count += 1

                # Log high-priority items
                if item.metadata.get('priority') == 'HIGH':
                    logger.warning(f"⚠️  HIGH PRIORITY: {item.title}")

        # Save index
        self._save_index()

        logger.info("=" * 80)
        logger.info(f"Collection Complete: {added_count} new items ({len(self.all_items)} total)")
        logger.info("=" * 80)

    def get_statistics(self) -> Dict:
        """Get statistics"""
        return self.index.get('stats', {})


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Military & Geopolitical OSINT Collector')
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['military_news', 'aircraft', 'ships', 'all'],
        default=['all'],
        help='Sources to collect from'
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

    args = parser.parse_args()

    aggregator = MilitaryOSINTAggregator()

    if args.stats:
        # Show statistics
        stats = aggregator.get_statistics()
        print("\n" + "=" * 80)
        print("Military OSINT Statistics")
        print("=" * 80)
        print(f"\nTotal Items: {stats.get('total_items', 0)}")
        print(f"Last Update: {aggregator.index.get('last_update', 'Never')}")

        print("\nBy Priority:")
        for priority, count in sorted(stats.get('by_priority', {}).items()):
            print(f"  {priority:10s}: {count:4d}")

        print("\nBy Country:")
        for country, count in sorted(stats.get('by_country', {}).items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {country:20s}: {count:4d}")

        print("\nTop Tags:")
        for tag, count in sorted(stats.get('by_tag', {}).items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {tag:20s}: {count:4d}")

        print()
        return

    # Collection mode
    if args.monitor:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            aggregator.collect(sources=args.sources)
            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        # One-shot collection
        aggregator.collect(sources=args.sources)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nAggregator stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
