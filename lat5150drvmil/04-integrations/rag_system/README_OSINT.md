# OSINT Feed Aggregator & Satellite Thermal Intelligence

Comprehensive Open-Source Intelligence (OSINT) collection and analysis suite for security and geospatial intelligence.

## Overview

This system collects, analyzes, and indexes intelligence from multiple public sources:

### 1. RSS/Atom Feeds (Security Blogs & News)
- KrebsOnSecurity, Schneier, Threatpost, The Hacker News
- Bleeping Computer, Dark Reading, Security Week
- SANS ISC, US-CERT, Cisco Talos
- **12+ premium security sources**

### 2. Threat Intelligence APIs
- abuse.ch (URLhaus, ThreatFox, Malware Bazaar)
- AlienVault OTX
- Blocklist.de
- **Real-time malware & IOC feeds**

### 3. GitHub Security Advisories
- Automated CVE tracking
- Dependency vulnerabilities
- Security patches

### 4. **Satellite Thermal Anomaly Detection** 🛰️
- NASA FIRMS VIIRS thermal data
- Near-real-time (3-5 hour latency)
- **Detect military strikes, fires, explosions from space**
- Geospatial correlation with reported events
- Based on methodology from pizzint.watch/polyglobe

### 5. IOCs (Indicators of Compromise)
- Malware hashes (MD5/SHA1/SHA256)
- Malicious IPs and domains
- Phishing URLs
- C2 infrastructure

## Quick Start

### Install Dependencies

All collectors use Python stdlib only - no external dependencies required!

### Set up API Keys (Optional but Recommended)

```bash
# NASA FIRMS (for satellite thermal data)
# Get free key: https://firms.modaps.eosdis.nasa.gov/api/area/
export FIRMS_API_KEY="your_key_here"

# Add to .env file
echo "FIRMS_API_KEY=your_key_here" >> .env
```

### Basic Usage

```bash
cd /home/user/LAT5150DRVMIL/rag_system

# Collect from all sources
python3 osint_feed_aggregator.py --sources all

# Specific sources
python3 osint_feed_aggregator.py --sources rss threat_intel github

# View statistics
python3 osint_feed_aggregator.py --stats

# Export report
python3 osint_feed_aggregator.py --export markdown --output osint_report.md

# Continuous monitoring (updates every hour)
python3 osint_feed_aggregator.py --monitor --interval 3600

# Update RAG embeddings after collection
python3 osint_feed_aggregator.py --sources all --update-rag
```

### Military OSINT Collector (ADS-B + AIS)

The dedicated military collector (`military_osint_collector.py`) fuses RSS intelligence with aircraft and naval telemetry.

1. **Configure flight data sources**
   ```bash
   export OPENSKY_USERNAME="your_callsign"
   export OPENSKY_PASSWORD="super_secret"
   export ADSBEXCHANGE_API_KEY="optional-paid-key"
   ```
   Anonymous OpenSky access works but is rate-limited; credentials unlock 40x higher throughput.

2. **Run collection**
   ```bash
   cd /home/user/LAT5150DRVMIL/04-integrations/rag_system
   python3 military_osint_collector.py --sources aircraft ships
   python3 military_osint_collector.py --sources all --monitor --interval 900
   ```

3. **Offline / air-gapped mode**
   - Curated ADS-B and AIS snapshots live under `04-integrations/rag_system/sample_data/`.
   - When live endpoints fail (firewalled lab, no API key), the collector automatically falls back to those curated datasets so downstream analysis never receives empty feeds.

4. **Artifacts & logging**
   - Logs: `rag_system/military_osint.log`
   - Structured index: `rag_system/military_osint_index.json` (used by RAG pipelines)
   - Raw feeds: `00-documentation/Security_Feed/OSINT/{military,aircraft,naval,conflicts}/…`

5. **Troubleshooting**
   - Ensure the `rag_system/` directory exists (the collector now creates it automatically before writing logs and indexes).
   - Review `rag_system/adsb_flight_tracker.log` for rate-limit or authentication errors.

## Satellite Thermal Anomaly Detection

**Advanced geospatial OSINT using NASA satellite data**

### Use Cases

- Detect naval strikes/interdiction operations before official announcements
- Monitor conflicts and military activity
- Track industrial incidents and disasters
- Verify reported events with satellite data
- Geolocate "international waters" incidents to precise coordinates

### Example: October 27, 2025 U.S. Interdiction Strike

From pizzint.watch methodology:

**Reported:** U.S. interdiction strike "~400nm SW of Acapulco" in international waters

**Satellite Detection:**
```bash
python3 satellite_thermal_collector.py --lat 14.0387 --lon -106.4606 --radius 50 --days 1
```

**Result:**
- Single daytime VIIRS hotspot at 14.0387° N, 106.4606° W
- FRP (Fire Radiative Power): ~500 MW (large explosion with sustained flame)
- Time: October 27, daytime pass
- Distance from Acapulco: ~415 nautical miles (matches report)
- **No recurring detections** (rules out industrial source)

### Regional Monitoring

```bash
# Monitor Eastern Pacific (Mexico coast - interdiction operations)
python3 satellite_thermal_collector.py --regions eastern_pacific --days 7

# Caribbean drug trafficking routes
python3 satellite_thermal_collector.py --regions caribbean --days 3

# Persian Gulf & Red Sea (naval operations)
python3 satellite_thermal_collector.py --regions persian_gulf red_sea --days 7

# Ukraine conflict - Black Sea
python3 satellite_thermal_collector.py --regions black_sea --days 7

# Global ocean monitoring
python3 satellite_thermal_collector.py --regions global_ocean --days 1
```

### Continuous Monitoring

```bash
# Monitor all strategic waterways, check every hour
python3 satellite_thermal_collector.py \
  --regions eastern_pacific caribbean persian_gulf red_sea black_sea \
  --monitor --interval 3600
```

### Understanding Thermal Anomaly Scores

The system automatically scores each detection (0.0 - 1.0):

| Score | Meaning | Typical Scenario |
|-------|---------|-----------------|
| **0.8-1.0** | ⚠️ **CRITICAL** | Large ocean explosion, likely naval strike |
| **0.6-0.8** | High | Significant fire/explosion, investigate |
| **0.4-0.6** | Moderate | Worth reviewing, may be industrial |
| **0.2-0.4** | Low | Likely routine/industrial |
| **0.0-0.2** | Very Low | Background/noise |

**Scoring Factors:**
- Ocean location (+0.3)
- High FRP (+0.3 if >100 MW)
- Satellite confidence (+0.2 if high)
- Daytime detection (+0.1 - easier to correlate with visual reports)
- Non-recurring (+implicit - one-off events more interesting)

### VIIRS Satellite Coverage

- **Satellites:** Suomi NPP (N) and NOAA-20 (J)
- **Resolution:** 375m thermal anomaly detection
- **Revisit:** 2-4 passes per day per satellite
- **Latency:** 3-5 hours from satellite pass to data availability
- **Coverage:** Global

## Data Collection Statistics

### RSS Feeds

**Sources:** 12 premium security blogs/news sites
**Update Frequency:** Every 1-6 hours (varies by source)
**Data Types:** CVEs, vulnerabilities, breaches, malware analysis, threat intelligence

### Threat Intelligence

**Sources:** abuse.ch (3 feeds), AlienVault OTX, Blocklist.de
**Update Frequency:** Real-time to hourly
**Data Types:**
- Malicious URLs (URLhaus)
- IOCs: IPs, domains, hashes (ThreatFox)
- Malware samples (Malware Bazaar)

### Satellite Thermal Anomalies

**Source:** NASA FIRMS VIIRS
**Update Frequency:** 3-5 hour latency, check every 1-6 hours
**Data Types:** Thermal anomalies with:
- Coordinates (lat/lon, 375m accuracy)
- Fire Radiative Power (MW)
- Brightness temperature (Kelvin)
- Confidence level
- Day/night flag
- Satellite ID

### GitHub Security Advisories

**Source:** GitHub Advisory Database
**Update Frequency:** Real-time (RSS feed)
**Data Types:** CVEs, dependency vulnerabilities, security patches

## Directory Structure

```
00-documentation/Security_Feed/
├── OSINT/
│   ├── rss/
│   │   ├── krebs_001.md
│   │   ├── schneier_002.md
│   │   └── ...
│   ├── threat_intel/
│   │   ├── urlhaus_001.md
│   │   ├── threatfox_002.md
│   │   └── ...
│   ├── github/
│   │   ├── ghsa_001.md
│   │   └── ...
│   ├── thermal_anomalies/
│   │   ├── a3f2b1c4.md  # Thermal anomaly detections
│   │   ├── b7e9a2d1.md
│   │   └── ...
│   └── ioc/
│       └── ...
│
rag_system/
├── osint_index.json              # Main OSINT index
├── thermal_anomaly_index.json    # Satellite thermal index
├── osint_aggregator.log
└── satellite_thermal.log
```

## Integration with RAG System

All OSINT data is automatically indexed for RAG queries:

```python
# Query example
result = rag_query("What thermal anomalies were detected in the Eastern Pacific last week?")

# The RAG system will search across:
# - Satellite thermal detection files
# - RSS security news mentioning the region
# - IOCs associated with maritime activity
# - GitHub advisories for relevant CVEs
```

### Auto-Update RAG

```bash
# Collect and update RAG in one command
python3 osint_feed_aggregator.py --sources all --update-rag
```

## Advanced Workflows

### 1. Verify Reported Event with Satellite Data

When you hear about an "incident in international waters":

```bash
# Example: "Strike reported ~400nm SW of Acapulco on Oct 27"

# Step 1: Calculate approximate coordinates
# 400 nm SW of Acapulco ≈ 14.0° N, 106.5° W

# Step 2: Search satellite data
python3 satellite_thermal_collector.py --lat 14.0 --lon -106.5 --radius 100 --days 3

# Step 3: Review high-score detections
# Look for:
# - One-off detection (not recurring)
# - Daytime (if visual confirmation exists)
# - High FRP (large explosion)
# - Matches distance from reported location
```

### 2. Monitor Conflict Zones

```bash
# Ukraine/Russia - Black Sea
python3 satellite_thermal_collector.py --regions black_sea --days 7 --monitor --interval 7200

# Red Sea - Houthi attacks
python3 satellite_thermal_collector.py --regions red_sea --days 3 --monitor --interval 3600

# South China Sea - Naval tensions
python3 satellite_thermal_collector.py --regions south_china_sea --days 7 --monitor --interval 7200
```

### 3. Correlate Multiple Sources

```bash
# Collect all OSINT
python3 osint_feed_aggregator.py --sources all

# Collect satellite thermal
python3 satellite_thermal_collector.py --regions eastern_pacific caribbean --days 7

# Query RAG for correlations
# "thermal anomalies mentioned in security news"
# "IOCs associated with Caribbean incidents"
```

### 4. Export Intelligence Report

```bash
# Collect data
python3 osint_feed_aggregator.py --sources all

# Export comprehensive report
python3 osint_feed_aggregator.py --export markdown --output reports/weekly_osint_$(date +%Y%m%d).md

# Include thermal anomalies
python3 satellite_thermal_collector.py --regions all --days 7
cat 00-documentation/Security_Feed/OSINT/thermal_anomalies/*.md > reports/thermal_$(date +%Y%m%d).md
```

## API Reference

### OSINTAggregator

```python
from rag_system.osint_feed_aggregator import OSINTAggregator

aggregator = OSINTAggregator(api_keys={'alienvault': 'key'})

# Collect from sources
aggregator.collect(sources=['rss', 'threat_intel', 'github'])

# Get statistics
stats = aggregator.get_statistics()

# Export report
aggregator.export_markdown_report(Path('report.md'))

# Update RAG
aggregator.update_rag_embeddings()
```

### SatelliteThermalCollector

```python
from rag_system.satellite_thermal_collector import SatelliteThermalCollector

collector = SatelliteThermalCollector(api_key='your_firms_api_key')

# Collect region
anomalies = collector.fetch_region('eastern_pacific', days=7)

# Search near point
anomalies = collector.fetch_point(lat=14.0, lon=-106.5, radius_km=50, days=3)

# Analyze
for anom in anomalies:
    if anom.anomaly_score > 0.7:
        print(f"⚠️  {anom.latitude}, {anom.longitude} - FRP: {anom.frp} MW")

# Save and index
collector.collect(regions=['eastern_pacific', 'caribbean'], days=7)
```

## Limitations & Considerations

### Satellite Thermal Detection

**Strengths:**
- Near-real-time (3-5 hour latency)
- Global coverage
- Independent verification of reported events
- Free/public data

**Limitations:**
- 375m resolution (can't identify specific vessels)
- Requires large thermal signature (explosions, fires)
- Weather can obscure detections (clouds)
- 2-4 passes per day (not continuous coverage)
- False positives from industrial sources
- Daytime detections more reliable (easier visual correlation)

**Best Practices:**
- Cross-reference with reported events
- Check for recurring detections (rules out industrial)
- Higher FRP = higher confidence (>100 MW for explosions)
- Daytime detections easier to verify
- Use distance calculations to verify reported locations

### RSS/Threat Intel

**Rate Limiting:**
- Respect source rate limits (1-2 second delays between requests)
- Some sources may block if too aggressive
- Use caching to avoid redundant requests

**API Keys:**
- FIRMS API key recommended (free, unlimited with registration)
- AlienVault OTX requires API key for full access
- Other sources work without keys but may be rate-limited

## Systemd Service (Continuous Monitoring)

### OSINT Aggregator Service

```ini
# /etc/systemd/system/osint-aggregator.service
[Unit]
Description=OSINT Feed Aggregator
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/user/LAT5150DRVMIL/rag_system
Environment="FIRMS_API_KEY=your_key"
ExecStart=/usr/bin/python3 osint_feed_aggregator.py --sources all --monitor --interval 3600 --update-rag
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
```

### Satellite Thermal Monitor Service

```ini
# /etc/systemd/system/satellite-thermal.service
[Unit]
Description=Satellite Thermal Anomaly Monitor
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/user/LAT5150DRVMIL/rag_system
Environment="FIRMS_API_KEY=your_key"
ExecStart=/usr/bin/python3 satellite_thermal_collector.py --regions eastern_pacific caribbean persian_gulf red_sea black_sea --monitor --interval 7200
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable osint-aggregator satellite-thermal
sudo systemctl start osint-aggregator satellite-thermal
sudo systemctl status osint-aggregator satellite-thermal
```

## Troubleshooting

### "FIRMS API key required"

Get a free API key:
1. Visit https://firms.modaps.eosdis.nasa.gov/api/area/
2. Register with email
3. Copy API key
4. Set environment variable: `export FIRMS_API_KEY="your_key"`

### "No thermal anomalies found"

- Check date range (use --days 7 for wider search)
- Verify region boundaries (use --stats to see what's collected)
- Some regions may have no thermal events
- VIIRS only detects large fires/explosions (FRP > ~5 MW)

### "Rate limited by source"

- Add delays between requests (code already includes 1-2s delays)
- Use caching (check if data already collected)
- Some sources may require API keys for higher rate limits

## Credits & References

- **NASA FIRMS:** https://firms.modaps.eosdis.nasa.gov/
- **Methodology:** pizzint.watch/polyglobe (Pentagon pizza deliveries & thermal anomalies)
- **abuse.ch:** https://abuse.ch/ (URLhaus, ThreatFox, Malware Bazaar)
- **GitHub Advisories:** https://github.com/advisories
- **VIIRS Instrument:** https://www.earthdata.nasa.gov/learn/find-data/near-real-time/viirs

## Security & Ethics

- **Public Data Only:** All sources are publicly available
- **Responsible Use:** For defensive security, research, journalism
- **No Classified Data:** VIIRS is unclassified satellite data
- **Attribution:** Properly cite sources in any publications
- **Rate Limiting:** Respect API limits and terms of service

---

**Document Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-11-09
**Component:** OSINT Intelligence Collection System
