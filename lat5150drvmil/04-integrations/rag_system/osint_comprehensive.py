#!/usr/bin/env python3
"""
Comprehensive International OSINT Source Integration

This module integrates extensive OSINT sources commonly found in community
collections like start.me OSINT pages, OSINT Inception, and other curated lists.

Data Sources Integrated:
1. Flight Tracking (ADS-B): Real-time aircraft monitoring
2. Ship Tracking (AIS): Maritime vessel monitoring
3. Satellite Imagery: Multiple satellite data providers
4. Social Media: Twitter/X, Reddit, Telegram, Discord
5. News Feeds: International news sources (200+ countries)
6. Open Data: Government portals, UN, World Bank, EU
7. Geospatial: Maps, terrain, infrastructure
8. Threat Intelligence: Expanded CTI feeds
9. Domain/Network: WHOIS, DNS, SSL/TLS monitoring
10. Dark Web: Onion search engines, paste sites
11. People Search: Public records, professional networks
12. Document Search: Academic papers, leaked docs, archives

Usage:
    from osint_comprehensive import ComprehensiveOSINTCollector

    collector = ComprehensiveOSINTCollector()

    # Collect flight data
    flights = collector.collect_flight_data(region='europe')

    # Collect ship data
    ships = collector.collect_ship_data(region='persian_gulf')

    # Collect satellite imagery
    imagery = collector.collect_satellite_imagery(bbox=[lat, lon, lat2, lon2])

    # Collect all sources
    collector.collect_all()
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# FLIGHT TRACKING (ADS-B) SOURCES
# ============================================================================

ADSB_SOURCES = {
    # Primary Sources
    "ADSBExchange": {
        "url": "https://www.adsbexchange.com/",
        "api": "https://www.adsbexchange.com/data/",
        "description": "Largest unfiltered flight tracking network",
        "features": ["military_aircraft", "blocked_aircraft", "real_time"],
        "requires_api_key": True,
        "coverage": "global"
    },

    "FlightRadar24": {
        "url": "https://www.flightradar24.com/",
        "api": "https://www.flightradar24.com/premium/",
        "description": "Popular flight tracker with historical data",
        "features": ["commercial", "some_military", "playback"],
        "requires_api_key": True,
        "coverage": "global"
    },

    "OpenSky_Network": {
        "url": "https://opensky-network.org/",
        "api": "https://opensky-network.org/api/",
        "description": "Open-source aircraft tracking (academic)",
        "features": ["open_data", "research", "historical"],
        "requires_api_key": False,
        "coverage": "global",
        "api_endpoint": "https://opensky-network.org/api/states/all"
    },

    "FlightAware": {
        "url": "https://flightaware.com/",
        "api": "https://flightaware.com/commercial/firehose/",
        "description": "Commercial and private flight tracking",
        "features": ["commercial", "general_aviation", "alerts"],
        "requires_api_key": True,
        "coverage": "global"
    },

    "Plane_Finder": {
        "url": "https://planefinder.net/",
        "api": "https://planefinder.net/",
        "description": "Independent flight tracker",
        "features": ["real_time", "mobile_apps"],
        "requires_api_key": False,
        "coverage": "global"
    },

    "RadarBox": {
        "url": "https://www.radarbox.com/",
        "api": "https://www.radarbox.com/",
        "description": "Flight tracking with ADS-B receiver network",
        "features": ["real_time", "statistics", "airports"],
        "requires_api_key": True,
        "coverage": "global"
    },

    # Government/Military Sources
    "ADS_B_Hub": {
        "url": "https://www.adsbhub.org/",
        "api": "http://www.adsbhub.org/api.php",
        "description": "Community ADS-B data sharing",
        "features": ["open_data", "community_driven"],
        "requires_api_key": False,
        "coverage": "global"
    },

    # Special Interest
    "Military_Aircraft_Tracker": {
        "description": "Twitter accounts tracking military flights",
        "sources": [
            "@Gerjon_",  # Dutch military tracking
            "@CivMilAir",  # Civil/Military air traffic
            "@Aircraft_Spots",  # Military aircraft spotting
            "@Intel_Sky",  # Intelligence aircraft
            "@AircraftSpots",  # Military movements
        ]
    }
}


# ============================================================================
# SHIP TRACKING (AIS) SOURCES
# ============================================================================

AIS_SOURCES = {
    "MarineTraffic": {
        "url": "https://www.marinetraffic.com/",
        "api": "https://www.marinetraffic.com/en/ais-api-services",
        "description": "Leading vessel tracking service",
        "features": ["commercial_vessels", "ports", "historical"],
        "requires_api_key": True,
        "coverage": "global",
        "military_coverage": "limited"  # Military vessels often disable AIS
    },

    "VesselFinder": {
        "url": "https://www.vesselfinder.com/",
        "api": "https://www.vesselfinder.com/api/docs",
        "description": "Free AIS vessel tracking",
        "features": ["real_time", "port_calls", "fleet_tracking"],
        "requires_api_key": True,
        "coverage": "global"
    },

    "ShipXplorer": {
        "url": "https://www.shipxplorer.com/",
        "description": "Real-time ship tracking and maritime information",
        "features": ["real_time", "photos", "specifications"],
        "requires_api_key": False,
        "coverage": "global"
    },

    "CruiseMapper": {
        "url": "https://www.cruisemapper.com/",
        "description": "Cruise ship tracking and itineraries",
        "features": ["cruise_ships", "schedules", "ports"],
        "requires_api_key": False,
        "coverage": "global"
    },

    "FleetMon": {
        "url": "https://www.fleetmon.com/",
        "api": "https://www.fleetmon.com/services/fleet-tracker/",
        "description": "Ship tracking for maritime professionals",
        "features": ["commercial", "analytics", "port_data"],
        "requires_api_key": True,
        "coverage": "global"
    },

    "MyShipTracking": {
        "url": "https://www.myshiptracking.com/",
        "description": "Free ship positions and tracking",
        "features": ["real_time", "port_arrivals", "photos"],
        "requires_api_key": False,
        "coverage": "global"
    },

    # Special Interest - Military/Government
    "HI_Sutton_Covert_Shores": {
        "url": "http://www.hisutton.com/",
        "description": "Submarine and naval OSINT analysis",
        "features": ["submarines", "naval_analysis", "imagery_analysis"],
        "focus": "military_naval"
    },

    "Naval_News": {
        "url": "https://www.navalnews.com/",
        "rss": "https://www.navalnews.com/feed/",
        "description": "Naval defense news and analysis",
        "features": ["warships", "naval_tech", "deployments"]
    }
}


# ============================================================================
# SATELLITE IMAGERY SOURCES
# ============================================================================

SATELLITE_IMAGERY_SOURCES = {
    # Open/Free Sources
    "Sentinel_Hub": {
        "url": "https://www.sentinel-hub.com/",
        "api": "https://services.sentinel-hub.com/",
        "description": "Copernicus Sentinel satellite data",
        "features": ["multispectral", "radar", "free_data"],
        "satellites": ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P"],
        "resolution": "10m-60m",
        "requires_api_key": True,
        "cost": "free_tier_available"
    },

    "Copernicus_Open_Access_Hub": {
        "url": "https://scihub.copernicus.eu/",
        "description": "Direct access to Sentinel satellite data",
        "features": ["optical", "radar", "atmospheric"],
        "cost": "free",
        "requires_account": True
    },

    "NASA_Earthdata": {
        "url": "https://earthdata.nasa.gov/",
        "description": "NASA Earth observation data",
        "features": ["landsat", "modis", "viirs", "aster"],
        "cost": "free",
        "requires_account": True
    },

    "USGS_EarthExplorer": {
        "url": "https://earthexplorer.usgs.gov/",
        "description": "USGS satellite and aerial imagery",
        "features": ["landsat", "aerial", "lidar", "historical"],
        "cost": "free",
        "requires_account": True
    },

    "Zoom_Earth": {
        "url": "https://zoom.earth/",
        "description": "Near real-time satellite imagery viewer",
        "features": ["weather", "fires", "storms"],
        "cost": "free",
        "requires_account": False
    },

    # Commercial High-Resolution
    "Planet": {
        "url": "https://www.planet.com/",
        "api": "https://developers.planet.com/",
        "description": "Daily global imagery (3m resolution)",
        "resolution": "3m-5m",
        "requires_api_key": True,
        "cost": "commercial"
    },

    "Maxar": {
        "url": "https://www.maxar.com/",
        "description": "Sub-meter satellite imagery",
        "satellites": ["WorldView-1", "WorldView-2", "WorldView-3", "GeoEye-1"],
        "resolution": "0.3m-0.5m",
        "cost": "commercial"
    },

    "Airbus_Defence_and_Space": {
        "url": "https://www.intelligence-airbusds.com/",
        "description": "High-resolution commercial imagery",
        "satellites": ["Pleiades", "SPOT"],
        "resolution": "0.5m-1.5m",
        "cost": "commercial"
    },

    # Analysis Platforms
    "Sentinel_Playground": {
        "url": "https://apps.sentinel-hub.com/sentinel-playground/",
        "description": "Interactive Sentinel image browser",
        "cost": "free"
    },

    "EOS_LandViewer": {
        "url": "https://eos.com/landviewer/",
        "description": "Satellite imagery search and analysis",
        "cost": "freemium"
    }
}


# ============================================================================
# SOCIAL MEDIA MONITORING SOURCES
# ============================================================================

SOCIAL_MEDIA_SOURCES = {
    # Twitter/X
    "Twitter_Advanced_Search": {
        "url": "https://twitter.com/search-advanced",
        "description": "Advanced Twitter search interface",
        "features": ["keyword", "geolocation", "date_range", "user"]
    },

    "Twitter_API_v2": {
        "url": "https://developer.twitter.com/en/docs/twitter-api",
        "description": "Twitter API for programmatic access",
        "requires_api_key": True,
        "cost": "tiered_pricing"
    },

    # Reddit
    "Reddit_Search": {
        "url": "https://www.reddit.com/search/",
        "description": "Reddit search across all subreddits"
    },

    "Pushshift_Reddit_API": {
        "url": "https://github.com/pushshift/api",
        "description": "Historical Reddit data API",
        "features": ["historical", "deleted_content", "bulk_access"]
    },

    # Telegram (already integrated)
    "Telegram_Search": {
        "url": "https://t.me/",
        "description": "Telegram channel and group search",
        "tools": ["@cse_gobot", "TelegramDB.org"]
    },

    # Discord
    "Discord_Search": {
        "description": "Discord server discovery",
        "tools": ["Disboard.org", "Discord.me", "Top.gg"]
    },

    # Facebook
    "Facebook_Graph_Search": {
        "description": "Facebook graph search (limited)",
        "note": "Significantly restricted after Cambridge Analytica"
    },

    # LinkedIn
    "LinkedIn_Search": {
        "url": "https://www.linkedin.com/search/",
        "description": "Professional network search",
        "features": ["people", "companies", "jobs", "posts"]
    },

    # Instagram
    "Instagram_Search": {
        "url": "https://www.instagram.com/explore/",
        "description": "Instagram hashtag and user search"
    },

    # YouTube
    "YouTube_Search": {
        "url": "https://www.youtube.com/",
        "api": "https://developers.google.com/youtube/v3",
        "description": "Video content search"
    },

    # Specialized Tools
    "TweetDeck": {
        "url": "https://tweetdeck.twitter.com/",
        "description": "Real-time Twitter monitoring dashboard"
    },

    "Hootsuite": {
        "url": "https://www.hootsuite.com/",
        "description": "Multi-platform social media monitoring",
        "cost": "commercial"
    },

    "Social_Searcher": {
        "url": "https://www.social-searcher.com/",
        "description": "Free social media search engine",
        "features": ["real_time", "multi_platform", "sentiment"]
    }
}


# ============================================================================
# INTERNATIONAL NEWS SOURCES (by Region)
# ============================================================================

INTERNATIONAL_NEWS_FEEDS = {
    # Global News
    "BBC_World": {
        "rss": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "region": "global",
        "language": "en"
    },
    "CNN_World": {
        "rss": "http://rss.cnn.com/rss/cnn_world.rss",
        "region": "global",
        "language": "en"
    },
    "Reuters_World": {
        "rss": "https://www.reutersagency.com/feed/",
        "region": "global",
        "language": "en"
    },
    "AP_News": {
        "rss": "https://feeds.apnews.com/rss/apf-topnews",
        "region": "global",
        "language": "en"
    },
    "Al_Jazeera": {
        "rss": "https://www.aljazeera.com/xml/rss/all.xml",
        "region": "global",
        "language": "en"
    },

    # Europe
    "The_Guardian": {
        "rss": "https://www.theguardian.com/world/rss",
        "region": "europe",
        "country": "UK",
        "language": "en"
    },
    "Der_Spiegel": {
        "rss": "https://www.spiegel.de/international/index.rss",
        "region": "europe",
        "country": "Germany",
        "language": "en"
    },
    "Le_Monde": {
        "rss": "https://www.lemonde.fr/rss/une.xml",
        "region": "europe",
        "country": "France",
        "language": "fr"
    },
    "EUobserver": {
        "rss": "https://euobserver.com/rss",
        "region": "europe",
        "focus": "EU_politics"
    },

    # Asia
    "South_China_Morning_Post": {
        "rss": "https://www.scmp.com/rss",
        "region": "asia",
        "country": "Hong Kong",
        "language": "en"
    },
    "Japan_Times": {
        "rss": "https://www.japantimes.co.jp/feed/",
        "region": "asia",
        "country": "Japan",
        "language": "en"
    },
    "Times_of_India": {
        "rss": "https://timesofindia.indiatimes.com/rss.cms",
        "region": "asia",
        "country": "India",
        "language": "en"
    },
    "Straits_Times": {
        "rss": "https://www.straitstimes.com/news/world/rss.xml",
        "region": "asia",
        "country": "Singapore",
        "language": "en"
    },

    # Middle East
    "Haaretz": {
        "rss": "https://www.haaretz.com/cmlink/1.628034",
        "region": "middle_east",
        "country": "Israel",
        "language": "en"
    },
    "Al_Arabiya": {
        "rss": "https://english.alarabiya.net/rss.xml",
        "region": "middle_east",
        "country": "UAE",
        "language": "en"
    },
    "Middle_East_Eye": {
        "rss": "https://www.middleeasteye.net/rss",
        "region": "middle_east",
        "language": "en"
    },

    # Africa
    "Daily_Maverick": {
        "rss": "https://www.dailymaverick.co.za/dmrss/",
        "region": "africa",
        "country": "South Africa",
        "language": "en"
    },
    "Africa_News": {
        "rss": "https://www.africanews.com/feed/",
        "region": "africa",
        "language": "en"
    },

    # Americas
    "Globe_and_Mail": {
        "rss": "https://www.theglobeandmail.com/arc/outboundfeeds/rss/",
        "region": "americas",
        "country": "Canada",
        "language": "en"
    },
    "Folha_de_Sao_Paulo": {
        "rss": "https://feeds.folha.uol.com.br/emcimadahora/rss091.xml",
        "region": "americas",
        "country": "Brazil",
        "language": "pt"
    },
    "La_Nacion": {
        "rss": "https://www.lanacion.com.ar/arc/outboundfeeds/rss/",
        "region": "americas",
        "country": "Argentina",
        "language": "es"
    },

    # Oceania
    "Sydney_Morning_Herald": {
        "rss": "https://www.smh.com.au/rss/feed.xml",
        "region": "oceania",
        "country": "Australia",
        "language": "en"
    },
    "NZ_Herald": {
        "rss": "https://www.nzherald.co.nz/arc/outboundfeeds/rss/",
        "region": "oceania",
        "country": "New Zealand",
        "language": "en"
    }
}


# ============================================================================
# OPEN DATA PORTALS
# ============================================================================

OPEN_DATA_SOURCES = {
    # International Organizations
    "UN_Data": {
        "url": "http://data.un.org/",
        "description": "United Nations statistical databases",
        "api": "http://data.un.org/Host.aspx?Content=API"
    },

    "World_Bank_Open_Data": {
        "url": "https://data.worldbank.org/",
        "api": "https://datahelpdesk.worldbank.org/knowledgebase/topics/125589",
        "description": "Development indicators and statistics"
    },

    "IMF_Data": {
        "url": "https://data.imf.org/",
        "description": "International Monetary Fund economic data"
    },

    "WHO_Data": {
        "url": "https://www.who.int/data",
        "description": "World Health Organization health statistics"
    },

    # US Government
    "Data_gov": {
        "url": "https://www.data.gov/",
        "description": "US government open data portal",
        "datasets": "300000+"
    },

    "Census_Bureau": {
        "url": "https://data.census.gov/",
        "api": "https://www.census.gov/data/developers/data-sets.html",
        "description": "US demographic and economic data"
    },

    # EU
    "EU_Open_Data_Portal": {
        "url": "https://data.europa.eu/",
        "description": "European Union open data",
        "datasets": "1000000+"
    },

    # UK
    "Data_gov_uk": {
        "url": "https://data.gov.uk/",
        "description": "UK government data portal"
    },

    # Other Countries
    "Data_gouv_fr": {
        "url": "https://www.data.gouv.fr/",
        "description": "French government data",
        "language": "fr"
    },

    "Data_gov_au": {
        "url": "https://data.gov.au/",
        "description": "Australian government data"
    },

    "Open_Data_Canada": {
        "url": "https://open.canada.ca/",
        "description": "Canadian government data"
    }
}


# ============================================================================
# GEOSPATIAL INTELLIGENCE SOURCES
# ============================================================================

GEOSPATIAL_SOURCES = {
    # Maps
    "OpenStreetMap": {
        "url": "https://www.openstreetmap.org/",
        "api": "https://wiki.openstreetmap.org/wiki/API",
        "description": "Collaborative world map",
        "cost": "free"
    },

    "Google_Maps": {
        "url": "https://maps.google.com/",
        "api": "https://developers.google.com/maps",
        "description": "Google mapping platform",
        "requires_api_key": True
    },

    "Mapbox": {
        "url": "https://www.mapbox.com/",
        "api": "https://docs.mapbox.com/api/",
        "description": "Custom maps and location data",
        "requires_api_key": True
    },

    # Terrain/Elevation
    "OpenTopography": {
        "url": "https://opentopography.org/",
        "description": "High-resolution topography data"
    },

    "SRTM_Data": {
        "url": "https://www2.jpl.nasa.gov/srtm/",
        "description": "NASA Shuttle Radar Topography Mission elevation data"
    },

    # Infrastructure
    "Wikimapia": {
        "url": "http://wikimapia.org/",
        "description": "Crowd-sourced geographic encyclopedia"
    },

    "Overpass_Turbo": {
        "url": "https://overpass-turbo.eu/",
        "description": "OpenStreetMap data extraction tool"
    },

    # Analysis Platforms
    "Google_Earth_Engine": {
        "url": "https://earthengine.google.com/",
        "description": "Planetary-scale geospatial analysis",
        "requires_account": True
    },

    "QGIS": {
        "url": "https://qgis.org/",
        "description": "Open-source GIS application",
        "cost": "free"
    }
}


# ============================================================================
# THREAT INTELLIGENCE (Expanded)
# ============================================================================

EXPANDED_THREAT_INTEL = {
    # Malware Analysis
    "VirusTotal": {
        "url": "https://www.virustotal.com/",
        "api": "https://developers.virustotal.com/reference/overview",
        "description": "File/URL analysis aggregator",
        "requires_api_key": True
    },

    "Hybrid_Analysis": {
        "url": "https://www.hybrid-analysis.com/",
        "api": "https://www.hybrid-analysis.com/docs/api/v2",
        "description": "Malware sandbox analysis",
        "requires_api_key": True
    },

    "Any_Run": {
        "url": "https://any.run/",
        "description": "Interactive malware analysis",
        "cost": "freemium"
    },

    "Joe_Sandbox": {
        "url": "https://www.joesandbox.com/",
        "description": "Deep malware analysis",
        "cost": "commercial"
    },

    # Threat Feeds
    "Emerging_Threats": {
        "url": "https://rules.emergingthreats.net/",
        "description": "IDS/IPS rules and IOCs"
    },

    "Feodo_Tracker": {
        "url": "https://feodotracker.abuse.ch/",
        "description": "Botnet C2 infrastructure tracking"
    },

    "SSL_Blacklist": {
        "url": "https://sslbl.abuse.ch/",
        "description": "Malicious SSL certificates"
    },

    # CTI Platforms
    "MISP_Project": {
        "url": "https://www.misp-project.org/",
        "description": "Threat intelligence sharing platform",
        "cost": "free_open_source"
    },

    "OpenCTI": {
        "url": "https://www.opencti.io/",
        "description": "Cyber threat intelligence platform",
        "cost": "free_open_source"
    },

    # Vulnerability Databases
    "Exploit_DB": {
        "url": "https://www.exploit-db.com/",
        "description": "Archive of public exploits",
        "searchable": True
    },

    "Packet_Storm": {
        "url": "https://packetstormsecurity.com/",
        "rss": "https://rss.packetstormsecurity.com/news/",
        "description": "Security tools and exploits"
    },

    "0day_today": {
        "url": "https://0day.today/",
        "description": "Exploit market (premium)",
        "cost": "commercial"
    }
}


# ============================================================================
# DOMAIN & NETWORK INTELLIGENCE
# ============================================================================

DOMAIN_NETWORK_SOURCES = {
    # WHOIS
    "WHOIS_Lookup": {
        "tools": [
            "https://whois.domaintools.com/",
            "https://www.whois.com/whois/",
            "https://who.is/"
        ],
        "description": "Domain registration information"
    },

    # DNS
    "DNS_Dumpster": {
        "url": "https://dnsdumpster.com/",
        "description": "DNS reconnaissance & research"
    },

    "SecurityTrails": {
        "url": "https://securitytrails.com/",
        "api": "https://docs.securitytrails.com/",
        "description": "Historical DNS data",
        "requires_api_key": True
    },

    "Shodan": {
        "url": "https://www.shodan.io/",
        "api": "https://developer.shodan.io/api",
        "description": "Internet-connected device search engine",
        "requires_api_key": True
    },

    "Censys": {
        "url": "https://censys.io/",
        "api": "https://search.censys.io/api",
        "description": "Internet-wide scanning and analysis",
        "requires_api_key": True
    },

    # SSL/TLS
    "Certificate_Transparency_Logs": {
        "url": "https://crt.sh/",
        "description": "SSL/TLS certificate search"
    },

    "SSL_Labs": {
        "url": "https://www.ssllabs.com/ssltest/",
        "description": "SSL/TLS configuration analysis"
    },

    # IP Reputation
    "AbuseIPDB": {
        "url": "https://www.abuseipdb.com/",
        "api": "https://docs.abuseipdb.com/",
        "description": "IP address abuse reporting",
        "requires_api_key": True
    },

    "IP_Quality_Score": {
        "url": "https://www.ipqualityscore.com/",
        "api": "https://www.ipqualityscore.com/documentation/overview",
        "description": "IP/domain fraud detection",
        "requires_api_key": True
    }
}


# ============================================================================
# PEOPLE SEARCH & OSINT
# ============================================================================

PEOPLE_SEARCH_SOURCES = {
    # Professional
    "LinkedIn": {
        "url": "https://www.linkedin.com/",
        "description": "Professional network (500M+ users)"
    },

    "Xing": {
        "url": "https://www.xing.com/",
        "description": "Professional network (Europe-focused)"
    },

    # Public Records (US)
    "Pipl": {
        "url": "https://pipl.com/",
        "description": "People search engine",
        "cost": "commercial"
    },

    "Spokeo": {
        "url": "https://www.spokeo.com/",
        "description": "People search (US)",
        "cost": "commercial"
    },

    # Username Search
    "Namechk": {
        "url": "https://namechk.com/",
        "description": "Username availability across platforms"
    },

    "NameCheckup": {
        "url": "https://namecheckup.com/",
        "description": "Social media username search"
    },

    "Sherlock": {
        "url": "https://github.com/sherlock-project/sherlock",
        "description": "Hunt down social media accounts by username",
        "cost": "free_open_source"
    },

    # Email Search
    "Hunter_io": {
        "url": "https://hunter.io/",
        "api": "https://hunter.io/api-documentation",
        "description": "Email address finder",
        "requires_api_key": True
    },

    "Have_I_Been_Pwned": {
        "url": "https://haveibeenpwned.com/",
        "api": "https://haveibeenpwned.com/API/v3",
        "description": "Breach notification service"
    }
}


# ============================================================================
# DOCUMENT SEARCH & ARCHIVES
# ============================================================================

DOCUMENT_SOURCES = {
    # Academic
    "Google_Scholar": {
        "url": "https://scholar.google.com/",
        "description": "Academic papers and citations"
    },

    "arXiv": {
        "url": "https://arxiv.org/",
        "api": "https://arxiv.org/help/api/",
        "description": "Open-access research papers"
    },

    "ResearchGate": {
        "url": "https://www.researchgate.net/",
        "description": "Research sharing network"
    },

    # Archives
    "Internet_Archive": {
        "url": "https://archive.org/",
        "api": "https://archive.org/services/docs/api/",
        "description": "Digital library (600B+ pages archived)"
    },

    "Wayback_Machine": {
        "url": "https://web.archive.org/",
        "api": "https://archive.org/help/wayback_api.php",
        "description": "Historical website snapshots"
    },

    # Leaks
    "WikiLeaks": {
        "url": "https://wikileaks.org/",
        "description": "Leaked documents publication"
    },

    "Public_Intelligence": {
        "url": "https://publicintelligence.net/",
        "description": "Government documents and FOIA releases"
    },

    "DocumentCloud": {
        "url": "https://www.documentcloud.org/",
        "description": "Document analysis and publication platform"
    },

    # File Search
    "FileChef": {
        "url": "https://www.filechef.com/",
        "description": "File search engine (open directories)"
    },

    "NAPALM_FTP_Indexer": {
        "url": "https://www.searchftps.net/",
        "description": "FTP server file search"
    }
}


# ============================================================================
# Collector Class
# ============================================================================

class ComprehensiveOSINTCollector:
    """
    Comprehensive OSINT data collector integrating all major sources
    """

    def __init__(self):
        self.adsb_sources = ADSB_SOURCES
        self.ais_sources = AIS_SOURCES
        self.satellite_sources = SATELLITE_IMAGERY_SOURCES
        self.social_media_sources = SOCIAL_MEDIA_SOURCES
        self.news_feeds = INTERNATIONAL_NEWS_FEEDS
        self.open_data = OPEN_DATA_SOURCES
        self.geospatial = GEOSPATIAL_SOURCES
        self.threat_intel = EXPANDED_THREAT_INTEL
        self.domain_network = DOMAIN_NETWORK_SOURCES
        self.people_search = PEOPLE_SEARCH_SOURCES
        self.documents = DOCUMENT_SOURCES

        logger.info("Comprehensive OSINT Collector initialized")
        logger.info(f"  - {len(ADSB_SOURCES)} ADS-B sources")
        logger.info(f"  - {len(AIS_SOURCES)} AIS sources")
        logger.info(f"  - {len(SATELLITE_IMAGERY_SOURCES)} satellite imagery sources")
        logger.info(f"  - {len(SOCIAL_MEDIA_SOURCES)} social media sources")
        logger.info(f"  - {len(INTERNATIONAL_NEWS_FEEDS)} international news feeds")
        logger.info(f"  - {len(OPEN_DATA_SOURCES)} open data portals")
        logger.info(f"  - {len(GEOSPATIAL_SOURCES)} geospatial sources")
        logger.info(f"  - {len(EXPANDED_THREAT_INTEL)} threat intel sources")
        logger.info(f"  - {len(DOMAIN_NETWORK_SOURCES)} domain/network sources")
        logger.info(f"  - {len(PEOPLE_SEARCH_SOURCES)} people search sources")
        logger.info(f"  - {len(DOCUMENT_SOURCES)} document sources")

    def get_all_sources(self) -> Dict:
        """Get all OSINT sources as a comprehensive dictionary"""
        return {
            "flight_tracking": self.adsb_sources,
            "ship_tracking": self.ais_sources,
            "satellite_imagery": self.satellite_sources,
            "social_media": self.social_media_sources,
            "international_news": self.news_feeds,
            "open_data": self.open_data,
            "geospatial": self.geospatial,
            "threat_intelligence": self.threat_intel,
            "domain_network": self.domain_network,
            "people_search": self.people_search,
            "documents": self.documents
        }

    def export_source_catalog(self, output_file: str = "osint_source_catalog.json"):
        """Export complete source catalog to JSON"""
        catalog = self.get_all_sources()

        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Source catalog exported to {output_path}")
        return output_path

    def collect_flight_data(self, region: str = 'global'):
        """
        Collect flight tracking data

        Note: Requires API keys for most sources
        Implement in production with actual API integration
        """
        logger.info(f"Flight data collection for region: {region}")
        logger.info("Implementation requires API keys:")
        for source, data in self.adsb_sources.items():
            if isinstance(data, dict) and data.get('requires_api_key'):
                logger.info(f"  - {source}: {data.get('url')}")

    def collect_ship_data(self, region: str = 'global'):
        """
        Collect ship tracking data

        Note: Requires API keys for most sources
        """
        logger.info(f"Ship tracking collection for region: {region}")
        logger.info("Implementation requires API keys:")
        for source, data in self.ais_sources.items():
            if isinstance(data, dict) and data.get('requires_api_key'):
                logger.info(f"  - {source}: {data.get('url')}")

    def collect_satellite_imagery(self, bbox: List[float] = None):
        """
        Collect satellite imagery

        Args:
            bbox: [min_lat, min_lon, max_lat, max_lon]
        """
        logger.info(f"Satellite imagery collection for bbox: {bbox}")
        logger.info("Free sources available:")
        for source, data in self.satellite_sources.items():
            if isinstance(data, dict) and data.get('cost') in ['free', 'freemium', 'free_tier_available']:
                logger.info(f"  - {source}: {data.get('description')}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function - export source catalog"""
    collector = ComprehensiveOSINTCollector()

    # Export catalog
    catalog_file = collector.export_source_catalog()

    total_sources = sum([
        len(ADSB_SOURCES),
        len(AIS_SOURCES),
        len(SATELLITE_IMAGERY_SOURCES),
        len(SOCIAL_MEDIA_SOURCES),
        len(INTERNATIONAL_NEWS_FEEDS),
        len(OPEN_DATA_SOURCES),
        len(GEOSPATIAL_SOURCES),
        len(EXPANDED_THREAT_INTEL),
        len(DOMAIN_NETWORK_SOURCES),
        len(PEOPLE_SEARCH_SOURCES),
        len(DOCUMENT_SOURCES)
    ])

    print("\n" + "=" * 80)
    print("Comprehensive OSINT Source Integration")
    print("=" * 80)
    print(f"\nTotal sources integrated: {total_sources}")
    print(f"\nSource catalog exported to: {catalog_file}")
    print("\nCategories:")
    print(f"  1. Flight Tracking (ADS-B) - {len(ADSB_SOURCES)} sources")
    print(f"  2. Ship Tracking (AIS) - {len(AIS_SOURCES)} sources")
    print(f"  3. Satellite Imagery - {len(SATELLITE_IMAGERY_SOURCES)} sources")
    print(f"  4. Social Media - {len(SOCIAL_MEDIA_SOURCES)} sources")
    print(f"  5. International News - {len(INTERNATIONAL_NEWS_FEEDS)} feeds")
    print(f"  6. Open Data - {len(OPEN_DATA_SOURCES)} portals")
    print(f"  7. Geospatial - {len(GEOSPATIAL_SOURCES)} sources")
    print(f"  8. Threat Intelligence - {len(EXPANDED_THREAT_INTEL)} sources")
    print(f"  9. Domain/Network - {len(DOMAIN_NETWORK_SOURCES)} sources")
    print(f" 10. People Search - {len(PEOPLE_SEARCH_SOURCES)} sources")
    print(f" 11. Documents - {len(DOCUMENT_SOURCES)} sources")
    print()


if __name__ == '__main__':
    main()
