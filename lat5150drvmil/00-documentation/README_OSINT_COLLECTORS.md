# OSINT Collectors - Comprehensive Documentation

**Dell Latitude 5450 MIL-SPEC AI Framework**

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

**Last Updated:** 2025-11-09

---

## Overview

The OSINT (Open Source Intelligence) Collectors provide comprehensive intelligence gathering capabilities from publicly available sources. All data collected is **UNCLASSIFIED** and sourced from open, legal, publicly accessible sources.

## Key Components

### 1. Military OSINT Collector (`military_osint_collector.py`)

Specialized collection for military, defense, and geopolitical intelligence.

**Data Sources:**
- Military news (DoD, Pentagon, NATO, UN)
- Defense press (Defense News, Jane's, Stars and Stripes)
- Intelligence community releases (CIA, FBI, NSA, ODNI)
- Think tank analysis (CSIS, ISW, RUSI)
- Framework for ADS-B aircraft tracking
- Framework for AIS naval vessel tracking

**Features:**
- Named entity extraction (countries, units, weapons)
- Priority assessment (HIGH/MEDIUM/LOW)
- Theater tagging (Indo-Pacific, Middle East, Ukraine, etc.)
- Automatic deduplication
- Markdown export for RAG integration

**Usage:**
```bash
# Collect military news
python3 military_osint_collector.py --sources military_news

# Continuous monitoring
python3 military_osint_collector.py --sources all --monitor --interval 3600

# Show statistics
python3 military_osint_collector.py --stats
```

**Key Capabilities:**

*Named Entity Recognition:*
- Countries: Russia, China, Iran, North Korea, etc.
- Military units: USS Ronald Reagan, 101st Airborne Division
- Weapon systems: F-35, HIMARS, Patriot, S-400, DF-21

*Theater Classification:*
- Indo-Pacific (South China Sea, Taiwan Strait)
- Middle East (Persian Gulf, Strait of Hormuz, Red Sea)
- Europe (Ukraine, Baltic, Arctic)

*Priority Assessment:*
- **HIGH**: Nuclear events, missile tests, strikes, conflicts
- **MEDIUM**: Exercises, deployments, sanctions, arms deals
- **LOW**: General news, routine operations

### 2. Satellite Thermal Collector (`satellite_thermal_collector.py`)

NASA FIRMS VIIRS satellite thermal anomaly detection for geospatial intelligence.

**Data Source:**
- NASA FIRMS (Fire Information for Resource Management System)
- VIIRS 375m resolution thermal anomalies
- Near-real-time (3-5 hour latency)
- Global coverage

**Features:**
- Automatic anomaly detection
- Ocean vs land classification
- Distance calculations (Haversine formula)
- Confidence scoring based on Fire Radiative Power (FRP)
- Regional monitoring (Eastern Pacific, Persian Gulf, South China Sea)
- Temporal analysis (one-off vs recurring)

**Usage:**
```bash
# Monitor Eastern Pacific ocean anomalies
python3 satellite_thermal_collector.py --region eastern_pacific --days 7

# Monitor specific coordinates
python3 satellite_thermal_collector.py --lat 14.0 --lon -106.5 --radius 100

# Continuous monitoring
python3 satellite_thermal_collector.py --monitor --interval 3600
```

**Methodology:**

Based on [pizzint.watch/polyglobe](https://pizzint.watch/polyglobe) approach:

1. **Thermal Detection**: VIIRS detects thermal anomalies from fires, explosions, industrial activity
2. **Geospatial Filtering**: Identify ocean vs land locations
3. **Anomaly Scoring**:
   - Ocean location (+40 points)
   - High FRP (>10 MW) (+30 points)
   - High confidence (+15 points)
   - Daytime detection (+10 points)
   - Distance from infrastructure (+5 points)
4. **Validation**: Cross-reference with reported events

**Use Cases:**
- Maritime interdiction verification
- "International waters" incident validation
- Industrial accident detection
- Conflict monitoring
- Search and rescue support

**Predefined Regions:**
```python
- eastern_pacific: Mexico coast (drug interdiction)
- caribbean: Caribbean trafficking routes
- persian_gulf: Strategic waterway
- south_china_sea: Disputed territories
- strait_of_hormuz: Critical chokepoint
- red_sea: Shipping lanes
- mediterranean: Migration routes
- black_sea: Ukraine conflict
```

### 3. OSINT Feed Aggregator (`osint_feed_aggregator.py`)

Multi-source threat intelligence and security feed aggregation.

**Data Sources:**
- **RSS Feeds** (12+): Krebs, Schneier, Threatpost, BleepingComputer
- **Threat Intel APIs**: AlienVault OTX, abuse.ch, VirusTotal
- **GitHub Security**: CVEs, advisories, security patches
- **IOC Feeds**: Malware hashes, IPs, domains, URLs
- **Certificate Transparency**: SSL/TLS certificates

**Features:**
- IOC extraction (IPs, domains, hashes, URLs, emails)
- CVE tracking and correlation
- TOON compression for efficient storage
- Automatic RAG integration
- Configurable update intervals

**Usage:**
```bash
# Collect from all sources
python3 osint_feed_aggregator.py --all

# Specific sources
python3 osint_feed_aggregator.py --rss --threat-intel --github

# Continuous monitoring
python3 osint_feed_aggregator.py --monitor --interval 3600

# Export
python3 osint_feed_aggregator.py --export markdown --output osint_report.md
```

**IOC Extraction:**
- **IPv4**: `192.168.1.1`, `10.0.0.0/8`
- **Domains**: `malware.example.com`
- **URLs**: `http://malicious.site/payload`
- **Hashes**: MD5, SHA1, SHA256
- **Emails**: `attacker@malicious.com`
- **CVEs**: `CVE-2024-1234`

### 4. Telegram Scrapers

#### Enhanced Security Scraper (`telegram_document_scraper.py`)

Multi-channel Telegram scraper with document download capabilities.

**Channels:**
- **cveNotify**: CVE announcements
- **Pwn3rzs**: Security research and exploits
- Custom channels (configurable)

**Features:**
- File downloads (.md, .pdf, .txt, .doc, .docx)
- SHA256 deduplication
- Automatic RAG updates
- CVE indexing
- Security document categorization

**Configuration:**
```bash
# Copy template
cp .env.telegram.template .env.telegram

# Edit with your API credentials
# Get API key from: https://my.telegram.org/apps
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
SECURITY_CHANNELS=cveNotify,Pwn3rzs
```

**Usage:**
```bash
# Scrape all configured channels
python3 telegram_document_scraper.py

# Specific channel
python3 telegram_document_scraper.py --channel Pwn3rzs

# Monitor mode
python3 telegram_document_scraper.py --monitor --interval 300
```

#### VX Underground Archive Downloader (`vxunderground_archive_downloader.py`)

Download VX Underground research paper archive.

**Collections:**
- APT reports (APT28, APT29, Lazarus, etc.)
- Malware analysis (ransomware, trojans, rootkits)
- Technique papers (persistence, evasion, C2)
- Security research

**Usage:**
```bash
# Download full archive
python3 vxunderground_archive_downloader.py --full

# Specific categories
python3 vxunderground_archive_downloader.py --categories apt,malware

# Resume partial downloads
python3 vxunderground_archive_downloader.py --resume
```

---

## Integration with RAG System

All OSINT collectors integrate with the RAG (Retrieval-Augmented Generation) system:

### 1. Document Storage

```
00-documentation/Security_Feed/OSINT/
├── military/
│   ├── military_news/
│   │   ├── {id}.md
│   ├── aircraft/
│   └── naval/
├── thermal_anomalies/
│   ├── {date}/
│   │   ├── {anomaly_id}.md
├── threat_intel/
│   ├── cves/
│   ├── malware/
│   └── iocs/
└── telegram/
    ├── documents/
    └── vxunderground/
```

### 2. Index Files

All collectors maintain JSON/TOON index files:

```
rag_system/
├── military_osint_index.json
├── thermal_anomaly_index.json
├── osint_index.json
└── telegram_security_index.json
```

### 3. TOON Compression

OSINT data uses TOON (Token-Oriented Object Notation) for:
- 40-60% storage savings
- 30-60% token savings for LLM queries
- Faster serialization

**Example:**
```python
from utils.toon_integration import save_toon_json, load_toon_json

# Save with compression
stats = save_toon_json("osint_data.toon", collected_items)
print(f"Saved {stats['savings_percent']:.1f}% disk space")

# Load
data = load_toon_json("osint_data.toon")
```

### 4. Automatic RAG Updates

When `AUTO_UPDATE_EMBEDDINGS=true` in configuration:
- New documents automatically embedded
- Vector index updated
- Available for semantic search

---

## Data Classification & Legal

**Classification:** All collected data is **UNCLASSIFIED // FOR OFFICIAL USE ONLY**

**Sources:** All data comes from:
- Publicly accessible websites
- Public RSS feeds
- Public APIs (with proper authentication)
- Open government sources
- Publicly available satellite data (NASA FIRMS)

**Legal Compliance:**
- No unauthorized access
- No authentication bypass
- Respects robots.txt
- Rate limiting implemented
- Terms of Service compliant

**Use Restrictions:**
- For defensive security purposes
- For threat intelligence
- For situational awareness
- For research and analysis
- NOT for offensive operations

---

## Testing

Comprehensive test suite available:

```bash
# Test OSINT collectors
cd rag_system
pytest test_osint_collectors.py -v

# Test TOON integration
cd 02-ai-engine
pytest test_toon_encoder.py -v
```

**Test Coverage:**
- Named entity extraction
- Geospatial calculations (Haversine distance)
- Priority assessment
- Anomaly scoring
- Round-trip encoding/decoding
- File operations
- Integration workflows

---

## Architecture

### Data Flow

```
[OSINT Sources] → [Collectors] → [Index Files] → [RAG System] → [AI Engine]
                       ↓
                  [TOON Storage]
                       ↓
                [Markdown Docs]
```

### Named Entity Extraction Pipeline

```
Raw Text → Regex Patterns → Entity Lists → Deduplication → Storage
              ↓
         (Countries, Units, Weapons, Locations)
```

### Thermal Anomaly Pipeline

```
NASA FIRMS API → Raw Detections → Geospatial Analysis → Scoring → High-Value Anomalies
                                          ↓
                                   (Ocean/Land, Distance, FRP)
```

---

## Performance

**Collection Rates:**
- Military news: ~100-200 items/day
- Thermal anomalies: ~500-2000/day (global), ~10-50/day (filtered)
- Threat intel: ~50-100 items/day
- Telegram: Real-time as posted

**Storage:**
- JSON: ~500KB per 100 items
- TOON: ~200KB per 100 items (60% savings)
- Markdown: ~10KB per document

**Resource Usage:**
- CPU: <5% during collection
- Memory: ~100-200MB
- Network: ~10-50MB/hour (depending on sources)

---

## Configuration

### Environment Variables

```bash
# Telegram API
TELEGRAM_API_ID=your_id
TELEGRAM_API_HASH=your_hash
SECURITY_CHANNELS=cveNotify,Pwn3rzs

# NASA FIRMS
FIRMS_API_KEY=your_key  # Get from https://firms.modaps.eosdis.nasa.gov/

# Auto-update
AUTO_UPDATE_EMBEDDINGS=true
UPDATE_BATCH_SIZE=10
UPDATE_INTERVAL_SECONDS=300
```

### Custom Regions (Satellite)

```python
# Add to satellite_thermal_collector.py
REGIONS = {
    'custom_region': {
        'name': 'Custom Region Name',
        'bounds': {
            'min_lat': 10.0, 'max_lat': 20.0,
            'min_lon': -110.0, 'max_lon': -100.0
        },
        'description': 'Description of region'
    }
}
```

### Custom Channels (Telegram)

Edit `.env.telegram`:
```bash
SECURITY_CHANNELS=cveNotify,Pwn3rzs,your_channel1,your_channel2
```

---

## Future Enhancements

### Planned Features

1. **ADS-B Aircraft Tracking**
   - Real-time military aircraft monitoring
   - Tanker/AWACS/SIGINT platform detection
   - Call sign analysis (BLOCKED, military codes)
   - Integration with ADS-B Exchange API

2. **AIS Naval Tracking**
   - Warship movement monitoring
   - Carrier strike group tracking
   - Port visit analysis
   - Integration with MarineTraffic API

3. **Enhanced Entity Recognition**
   - Deep learning NER models
   - Context-aware extraction
   - Relationship mapping
   - Timeline construction

4. **Correlation Engine**
   - Cross-source event correlation
   - Temporal pattern detection
   - Geospatial clustering
   - Automated reporting

5. **Machine Learning Enhancements**
   - Anomaly priority prediction
   - Event classification
   - Threat scoring
   - Trend analysis

---

## Troubleshooting

### Common Issues

**Issue:** `FIRMS_API_KEY not found`
- **Solution:** Set environment variable or use 'DEMO' key (limited to 10 requests/hour)

**Issue:** `Telegram authentication failed`
- **Solution:** Verify API_ID and API_HASH from https://my.telegram.org/apps

**Issue:** `Rate limit exceeded`
- **Solution:** Increase `--interval` parameter, default rate limits are conservative

**Issue:** `No thermal anomalies found`
- **Solution:** Verify region bounds, try larger area or longer time window

---

## References

### Data Sources

- **NASA FIRMS**: https://firms.modaps.eosdis.nasa.gov/
- **Department of Defense**: https://www.defense.gov/
- **GitHub Security**: https://github.com/advisories
- **AlienVault OTX**: https://otx.alienvault.com/
- **abuse.ch**: https://abuse.ch/
- **VX Underground**: https://github.com/vxunderground/VXUG-Papers

### Methodologies

- **Satellite OSINT**: https://pizzint.watch/polyglobe
- **Haversine Formula**: https://en.wikipedia.org/wiki/Haversine_formula
- **VIIRS Specifications**: https://www.earthdata.nasa.gov/sensors/viirs

### Specifications

- **TOON Format**: https://github.com/toon-format/spec
- **CVE Naming**: https://cve.mitre.org/
- **IOC Formats**: STIX, OpenIOC, MISP

---

## Contact & Support

For issues or enhancements:
- Create issue in project repository
- Tag as `osint`, `intelligence`, or `data-collection`
- Include collector name and error logs

---

**Document Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

**Component:** AI Framework - OSINT Collection System

**Version:** 1.0.0

**Last Updated:** 2025-11-09
