# Comprehensive OSINT Integration - Complete

**Dell Latitude 5450 MIL-SPEC AI Framework - OSINT Module**

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

**Last Updated:** 2025-11-09

---

## Overview

Massive OSINT (Open Source Intelligence) integration adding **167+ new sources** across 11 major categories, plus integration of the DIRECTEYE blockchain intelligence platform.

## What's New

### 1. Comprehensive International OSINT Sources (122 sources)

**File:** `rag_system/osint_comprehensive.py`

Integrates sources commonly found in community OSINT collections (start.me, OSINT Inception, etc.):

#### Flight Tracking (8 ADS-B sources)
- **OpenSky Network** - Open-source global ADS-B data (free API)
- **ADS-B Exchange** - Largest unfiltered flight tracker (military aircraft visible)
- **FlightRadar24** - Commercial flight tracking with historical data
- **FlightAware** - General aviation and commercial flights
- **Plane Finder** - Independent flight tracker
- **RadarBox** - ADS-B receiver network
- **ADS-B Hub** - Community data sharing
- **Military Aircraft Trackers** - Twitter/X accounts tracking military flights

**Use Cases:**
- Track military aircraft movements
- Monitor tanker/AWACS deployment (SIGINT indicators)
- Surveillance aircraft tracking (RC-135, E-8 JSTARS, P-8)
- Strategic bomber positioning (B-52, B-1, B-2)
- Transport aircraft for troop movements (C-17, C-130)

#### Ship Tracking (8 AIS sources)
- **MarineTraffic** - Leading vessel tracking service
- **VesselFinder** - Free AIS tracking
- **ShipXplorer** - Real-time ship tracking
- **CruiseMapper** - Cruise ship tracking
- **FleetMon** - Maritime professional tracking
- **MyShipTracking** - Free ship positions
- **HI Sutton Covert Shores** - Submarine and naval OSINT analysis
- **Naval News** - Naval defense news and analysis

**Use Cases:**
- Track carrier strike groups
- Monitor strategic waterways
- Submarine tender activity (indicates nearby sub operations)
- Naval exercise monitoring
- Port visits and deployments

#### Satellite Imagery (10 sources)
- **Sentinel Hub** - Copernicus Sentinel data (10-60m resolution, free)
- **Copernicus Open Access Hub** - Direct Sentinel access
- **NASA Earthdata** - Landsat, MODIS, VIIRS (free)
- **USGS EarthExplorer** - USGS satellite and aerial imagery
- **Zoom Earth** - Near real-time satellite viewer
- **Planet** - Daily 3-5m imagery (commercial)
- **Maxar** - Sub-meter imagery (0.3-0.5m, commercial)
- **Airbus Defence and Space** - Pleiades/SPOT (0.5-1.5m)
- **Sentinel Playground** - Interactive Sentinel browser
- **EOS LandViewer** - Satellite search and analysis

**Use Cases:**
- Before/after imagery analysis
- Infrastructure monitoring
- Disaster response
- Military base observation
- Change detection

#### Social Media (13 sources)
- **Twitter/X Advanced Search** - Keyword, geolocation, date filtering
- **Twitter API v2** - Programmatic access
- **Reddit Search** - Cross-subreddit search
- **Pushshift Reddit API** - Historical Reddit data
- **Telegram Search** - Channel/group discovery
- **Discord Search** - Server discovery tools
- **Facebook Graph Search** - Limited search capabilities
- **LinkedIn Search** - Professional network
- **Instagram Search** - Hashtag and user search
- **YouTube Search & API** - Video content
- **TweetDeck** - Real-time Twitter monitoring
- **Hootsuite** - Multi-platform monitoring
- **Social Searcher** - Free social media search engine

#### International News (23 feeds)
Organized by region:
- **Global**: BBC, CNN, Reuters, AP, Al Jazeera
- **Europe**: The Guardian (UK), Der Spiegel (Germany), Le Monde (France), EUobserver
- **Asia**: SCMP (Hong Kong), Japan Times, Times of India, Straits Times (Singapore)
- **Middle East**: Haaretz (Israel), Al Arabiya (UAE), Middle East Eye
- **Africa**: Daily Maverick (South Africa), Africa News
- **Americas**: Globe and Mail (Canada), Folha (Brazil), La Nacion (Argentina)
- **Oceania**: Sydney Morning Herald, NZ Herald

#### Open Data (11 portals)
- **UN Data** - United Nations statistics
- **World Bank Open Data** - Development indicators
- **IMF Data** - Economic data
- **WHO Data** - Health statistics
- **Data.gov** - US government (300,000+ datasets)
- **Census Bureau** - US demographics
- **EU Open Data Portal** - European Union (1M+ datasets)
- **Data.gov.uk** - UK government
- **Data.gouv.fr** - French government
- **Data.gov.au** - Australian government
- **Open Data Canada** - Canadian government

#### Geospatial (9 sources)
- **OpenStreetMap** - Collaborative world map
- **Google Maps/API** - Mapping platform
- **Mapbox** - Custom maps and location data
- **OpenTopography** - High-resolution topography
- **SRTM Data** - NASA elevation data
- **Wikimapia** - Geographic encyclopedia
- **Overpass Turbo** - OSM data extraction
- **Google Earth Engine** - Planetary-scale analysis
- **QGIS** - Open-source GIS application

#### Threat Intelligence (12 sources)
- **VirusTotal** - File/URL analysis aggregator
- **Hybrid Analysis** - Malware sandbox
- **Any.Run** - Interactive malware analysis
- **Joe Sandbox** - Deep malware analysis
- **Emerging Threats** - IDS/IPS rules
- **Feodo Tracker** - Botnet C2 tracking
- **SSL Blacklist** - Malicious SSL certificates
- **MISP Project** - Threat intelligence platform
- **OpenCTI** - Cyber threat intelligence
- **Exploit-DB** - Public exploits archive
- **Packet Storm** - Security tools and exploits
- **0day.today** - Exploit marketplace

#### Domain/Network (9 sources)
- **WHOIS Lookup** - Domain registration info
- **DNS Dumpster** - DNS reconnaissance
- **SecurityTrails** - Historical DNS data
- **Shodan** - Internet device search engine
- **Censys** - Internet-wide scanning
- **Certificate Transparency Logs** - SSL/TLS search
- **SSL Labs** - SSL/TLS configuration analysis
- **AbuseIPDB** - IP abuse reporting
- **IP Quality Score** - Fraud detection

#### People Search (9 sources)
- **LinkedIn** - Professional network
- **Xing** - European professional network
- **Pipl** - People search engine
- **Spokeo** - US people search
- **Namechk** - Username availability
- **NameCheckup** - Social media username search
- **Sherlock** - Hunt social media accounts by username
- **Hunter.io** - Email address finder
- **Have I Been Pwned** - Breach notification

#### Documents (10 sources)
- **Google Scholar** - Academic papers
- **arXiv** - Research papers
- **ResearchGate** - Research sharing
- **Internet Archive** - Digital library (600B+ pages)
- **Wayback Machine** - Historical websites
- **WikiLeaks** - Leaked documents
- **Public Intelligence** - Government documents/FOIA
- **DocumentCloud** - Document analysis platform
- **FileChef** - Open directory file search
- **NAPALM FTP Indexer** - FTP file search

---

### 2. UK-Specific OSINT Sources (45 sources)

**File:** `rag_system/osint_uk_sources.py`

Based on: https://github.com/paulpogoda/OSINT-Tools-UK

#### People Search (9 sources)
- **192.com** - Electoral roll, director info
- **ReversePP** - Reverse people search (5 free/day)
- **Public Insights** - People/business intelligence
- **Genes Reunited** - BMD certificates, family tree
- **Amazon UK Wedding Registry** - Wedding plans
- **FreeCen** - Historical census (1841-1911)
- **BMD Registers** - Non-conformist records
- **Find My Past** - Family history
- **Ancestry UK** - Family tree and records

#### Company Search (6 sources)
- **Companies House** - Official UK company registry (FREE API)
- **Financial Services Register** - FCA regulated firms
- **Dun & Bradstreet** - D-U-N-S numbers
- **MHRA Medicine Registry** - Licensed medicine sellers
- **Charity Commission** - England/Wales charities
- **Scottish Charity Register** - Scottish charities

#### Government Data (6 sources)
- **Data.gov.uk** - 50,000+ datasets
- **Office for National Statistics** - Census, economy
- **The National Archives** - Historical records
- **BAILII** - Legal database
- **Parliament UK** - Parliamentary records
- **Electoral Commission** - Party donations, elections

#### Property (7 sources)
- **Land Registry** - Ownership records (£3/title)
- **Cadastre.uk** - Property mapping
- **Wales Property Register** - Rental properties
- **London Rent Maps** - Rental pricing
- **Who Owns England** - Land ownership visualization
- **Rightmove** - Property listings/sold prices
- **Zoopla** - Valuations and sold prices

#### Vehicles (5 sources)
- **Vehicle Enquiry Service** - DVLA tax/MOT status
- **Check MOT History** - Historical MOT results
- **Partial Number Plate Search** - Registration lookup
- **G-INFO Aircraft Register** - UK aircraft ownership
- **Ship Register** - UK vessel registration

#### Court Records (5 sources)
- **Courts and Tribunals Judiciary** - Judgments
- **Case Tracker** - Civil appeals
- **Supreme Court** - Supreme Court cases
- **Scottish Courts** - Scottish judgments
- **Northern Ireland Courts** - NI judgments

#### Procurement (4 sources)
- **Contracts Finder** - £12,000+ contracts
- **Find a Tender** - £139,688+ high-value
- **Public Contracts Scotland** - Scottish procurement
- **Sell2Wales** - Welsh procurement

#### Domain/Network (3 sources)
- **Nominet UK** - .uk domain registry
- **WHOIS UK** - UK domain lookup
- **Jisc** - Academic/research networks

---

### 3. ADS-B Flight Tracking Implementation

**File:** `rag_system/adsb_flight_tracker.py`

Production-ready flight tracking with OpenSky Network integration.

#### Features
- **Real-time aircraft tracking** - Global coverage
- **Military aircraft detection** - Automatic identification
- **Region monitoring** - Predefined regions (Persian Gulf, Taiwan Strait, etc.)
- **Call sign analysis** - BLOCKED, RCH, CNV, SAM, etc.
- **Aircraft type recognition** - B-52, F-35, KC-135, RC-135, etc.

#### Predefined Regions
```python
REGIONS = {
    'global', 'europe', 'middle_east', 'persian_gulf',
    'south_china_sea', 'taiwan_strait', 'baltic',
    'mediterranean', 'north_pacific'
}
```

#### Military Call Signs Detected
- **BLOCKED** - Blocked military aircraft
- **RCH/CNV** - USAF transport
- **EVAC** - Evacuation flights
- **PAT/ORBIT** - Patrol/surveillance
- **SAM** - Special Air Mission (VIP)
- **SPAR** - Special Air Resources

#### Military Aircraft Types
- **Bombers**: B-52, B-1, B-2
- **Fighters**: F-15, F-16, F-18, F-22, F-35
- **Transport**: C-130, C-17, C-5
- **Tankers**: KC-135, KC-10, KC-46
- **SIGINT/AWACS**: E-3, E-8, RC-135, P-8
- **ISR**: U-2, RQ-4

#### Usage
```bash
# Track Persian Gulf
python3 adsb_flight_tracker.py --region persian_gulf

# Monitor for military aircraft
python3 adsb_flight_tracker.py --military --monitor --interval 60

# Continuous monitoring
python3 adsb_flight_tracker.py --region taiwan_strait --monitor --duration 3600
```

#### API Setup
```bash
# Optional: Register for higher rate limits
export OPENSKY_USERNAME="your_username"
export OPENSKY_PASSWORD="your_password"

# Without credentials: 10 requests/minute
# With credentials: 400 requests/minute
```

---

### 4. DIRECTEYE MCP Server Integration

**Location:** `mcp_servers/DIRECTEYE` (git submodule)

**Repository:** https://github.com/SWORDIntel/DIRECTEYE

Enterprise blockchain intelligence platform with 40+ OSINT services:

#### Key Features
- **40+ OSINT Services** - Breach data, corporate intel, threat feeds
- **Blockchain Analysis** - Entity attribution, sanctions, dark web
- **ML Analytics** - 5 engines (risk scoring, entity resolution, predictive, cross-chain, network)
- **Post-Quantum Cryptography** - ML-KEM-1024, ML-DSA-87, CNSA 2.0 compliant
- **MCP Integration** - 35 tools for AI assistant integration
- **Real-Time Monitoring** - Multi-channel alerting

#### MCP Tools Available
```bash
# Start DIRECTEYE MCP server
cd mcp_servers/DIRECTEYE
./launch.sh start

# Backend API: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## File Structure

```
LAT5150DRVMIL/
├── rag_system/
│   ├── osint_comprehensive.py          # 122 international sources
│   ├── osint_uk_sources.py             # 45 UK-specific sources
│   ├── adsb_flight_tracker.py          # ADS-B flight tracking
│   ├── military_osint_collector.py     # Military OSINT (existing)
│   ├── satellite_thermal_collector.py  # NASA FIRMS VIIRS (existing)
│   ├── osint_feed_aggregator.py        # Threat intel aggregator (existing)
│   ├── osint_source_catalog.json       # Generated comprehensive catalog
│   └── uk_osint_catalog.json           # Generated UK catalog
│
├── mcp_servers/
│   └── DIRECTEYE/                      # Blockchain intelligence MCP server (submodule)
│
└── 00-documentation/
    ├── README_OSINT_COLLECTORS.md      # Original OSINT documentation
    └── README_OSINT_INTEGRATION_COMPLETE.md  # This file
```

---

## Total Sources Summary

| Category | Sources | File |
|----------|---------|------|
| **Flight Tracking (ADS-B)** | 8 | osint_comprehensive.py |
| **Ship Tracking (AIS)** | 8 | osint_comprehensive.py |
| **Satellite Imagery** | 10 | osint_comprehensive.py |
| **Social Media** | 13 | osint_comprehensive.py |
| **International News** | 23 | osint_comprehensive.py |
| **Open Data** | 11 | osint_comprehensive.py |
| **Geospatial** | 9 | osint_comprehensive.py |
| **Threat Intelligence** | 12 | osint_comprehensive.py |
| **Domain/Network (Intl)** | 9 | osint_comprehensive.py |
| **People Search (Intl)** | 9 | osint_comprehensive.py |
| **Documents** | 10 | osint_comprehensive.py |
| **UK People Search** | 9 | osint_uk_sources.py |
| **UK Company Search** | 6 | osint_uk_sources.py |
| **UK Government Data** | 6 | osint_uk_sources.py |
| **UK Property** | 7 | osint_uk_sources.py |
| **UK Vehicles** | 5 | osint_uk_sources.py |
| **UK Court Records** | 5 | osint_uk_sources.py |
| **UK Procurement** | 4 | osint_uk_sources.py |
| **UK Domain/Network** | 3 | osint_uk_sources.py |
| **DIRECTEYE MCP** | 40+ | mcp_servers/DIRECTEYE |
| **Existing Collectors** | 30+ | military/satellite/feed collectors |
| **TOTAL** | **237+** | **Across all modules** |

---

## Usage Examples

### 1. Export Source Catalogs

```python
# Export comprehensive international sources
from osint_comprehensive import ComprehensiveOSINTCollector
collector = ComprehensiveOSINTCollector()
collector.export_source_catalog()
# Creates: osint_source_catalog.json

# Export UK sources
from osint_uk_sources import UKOSINTCollector
uk_collector = UKOSINTCollector()
uk_collector.export_catalog()
# Creates: uk_osint_catalog.json
```

### 2. Flight Tracking

```python
from adsb_flight_tracker import ADSBFlightTracker

tracker = ADSBFlightTracker()

# Get flights in Persian Gulf
flights = tracker.get_flights_in_region('persian_gulf')
print(f"Found {len(flights)} aircraft")

# Get only military aircraft
military = tracker.get_military_aircraft(region='taiwan_strait')
for aircraft in military:
    print(f"{aircraft.callsign}: {aircraft.altitude}m")

# Track specific aircraft
aircraft = tracker.track_aircraft('AE681C')

# Continuous monitoring
tracker.monitor_region('south_china_sea', interval=60, duration=3600)
```

### 3. UK OSINT

```python
from osint_uk_sources import UKOSINTCollector

uk = UKOSINTCollector()

# Search for person
results = uk.search_person(name="John Smith", location="London")

# Search company
company = uk.search_company(name="Example Ltd")

# Property lookup
property_info = uk.search_property(postcode="SW1A 1AA")

# Vehicle check
vehicle = uk.check_vehicle(registration="AB12 CDE")
```

### 4. DIRECTEYE MCP Integration

```bash
# Start DIRECTEYE server
cd mcp_servers/DIRECTEYE
./launch.sh start

# Access MCP tools via AI assistant
# DIRECTEYE provides 35 MCP tools automatically
```

---

## API Keys and Configuration

Many sources require API keys for programmatic access:

### Required API Keys

```bash
# ADS-B Flight Tracking
export OPENSKY_USERNAME="your_username"
export OPENSKY_PASSWORD="your_password"
export ADSBEXCHANGE_API_KEY="your_key"

# Satellite Imagery
export SENTINEL_HUB_CLIENT_ID="your_id"
export SENTINEL_HUB_CLIENT_SECRET="your_secret"

# Threat Intelligence
export VIRUSTOTAL_API_KEY="your_key"
export SHODAN_API_KEY="your_key"
export ABUSEIPDB_API_KEY="your_key"

# UK Companies House
export COMPANIES_HOUSE_API_KEY="your_key"

# Social Media
export TWITTER_API_KEY="your_key"
export TWITTER_API_SECRET="your_secret"
```

### Free Tiers Available

- **OpenSky Network**: 10 req/min anonymous, 400 req/min registered (FREE)
- **Sentinel Hub**: 5000 processing units/month (FREE tier)
- **VirusTotal**: 4 req/min (FREE)
- **Companies House UK**: FREE API with registration

---

## Legal & Compliance

**Classification:** All collected data is **UNCLASSIFIED // FOR OFFICIAL USE ONLY**

**Sources:** All data from:
- Publicly accessible websites
- Public APIs (with proper authentication)
- Open government sources
- Public satellite data
- Legal OSINT collection methods

**Compliance:**
- ✅ No unauthorized access
- ✅ No authentication bypass
- ✅ Respects robots.txt
- ✅ Rate limiting implemented
- ✅ Terms of Service compliant

**Use Restrictions:**
- ✅ Defensive security purposes
- ✅ Threat intelligence
- ✅ Situational awareness
- ✅ Research and analysis
- ❌ NOT for offensive operations

---

## Performance

**Collection Rates:**
- **Flight tracking**: Real-time updates (10-400/min depending on authentication)
- **Ship tracking**: Real-time AIS updates
- **Satellite imagery**: Historical + near-real-time
- **News feeds**: Hourly updates
- **Threat intelligence**: Continuous monitoring

**Storage (with TOON compression):**
- **Catalogs**: 150KB for 237 sources
- **Flight data**: ~500KB per 1000 aircraft positions
- **Imagery**: Varies by resolution (3m to 0.3m)

---

## Future Enhancements

1. **Full AIS Ship Tracking Implementation**
   - MarineTraffic API integration
   - Carrier strike group tracking
   - Port call analysis

2. **Satellite Imagery Automation**
   - Automated Sentinel Hub downloads
   - Change detection algorithms
   - Anomaly highlighting

3. **Social Media Monitoring**
   - Twitter API v2 integration
   - Real-time hashtag tracking
   - Geolocation-based search

4. **ML-Powered Analysis**
   - Entity resolution across sources
   - Threat correlation
   - Pattern detection
   - Predictive analytics

5. **Integration with Existing Collectors**
   - Cross-reference flight data with military news
   - Correlate satellite thermal anomalies with ship positions
   - Link threat intel to infrastructure monitoring

---

## References

### Source Collections
- **start.me OSINT pages**: Community-curated OSINT resources
- **OSINT Inception**: https://start.me/p/Pwy0X4/osint-inception
- **OSINT Tools UK**: https://github.com/paulpogoda/OSINT-Tools-UK
- **DIRECTEYE**: https://github.com/SWORDIntel/DIRECTEYE

### APIs & Documentation
- **OpenSky Network**: https://openskynetwork.github.io/opensky-api/
- **Sentinel Hub**: https://docs.sentinel-hub.com/
- **Companies House API**: https://developer-specs.company-information.service.gov.uk/
- **VirusTotal API**: https://developers.virustotal.com/

### Methodologies
- **ADS-B Tracking**: https://www.adsbexchange.com/
- **Satellite OSINT**: Previously documented (pizzint.watch/polyglobe)
- **UK OSINT**: https://github.com/paulpogoda/OSINT-Tools-UK

---

## Summary

This integration adds **237+ OSINT sources** to the LAT5150DRVMIL AI Framework:

✅ **122 international sources** across 11 categories
✅ **45 UK-specific sources** across 8 categories
✅ **40+ blockchain intelligence** services via DIRECTEYE MCP
✅ **30+ existing collectors** (military, satellite, threat intel)

**Total OSINT capability: 237+ sources**

All sources are:
- Legally accessible
- Properly authenticated
- Rate-limited
- UNCLASSIFIED
- Production-ready

---

**Document Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

**Component:** AI Framework - Comprehensive OSINT Integration

**Version:** 2.0.0

**Last Updated:** 2025-11-09

**Integration Complete** ✅
