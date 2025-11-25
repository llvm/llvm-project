#!/usr/bin/env python3
"""
RAG System - Dell Latitude 5450 MIL-SPEC AI Framework

This package contains the Retrieval-Augmented Generation (RAG) system components
including OSINT collectors, document processors, and security intelligence feeds.

Key Components:
- OSINT Collection: Military, geopolitical, security, and threat intelligence
- Telegram Scrapers: Security channels, vulnerability feeds, research papers
- Satellite Monitoring: NASA FIRMS VIIRS thermal anomaly detection
- Document Processing: Multi-format ingestion for RAG knowledge base
- Vector Search: Embedding generation and similarity search
"""

# OSINT Collectors
from .osint_feed_aggregator import (
    OSINTFeedAggregator,
    ThreatIntelItem,
    IOCExtractor
)

from .military_osint_collector import (
    MilitaryOSINTAggregator,
    MilitaryIntelItem,
    MilitaryNewsCollector,
    AircraftTrackingCollector,
    NavalTrackingCollector
)

from .satellite_thermal_collector import (
    SatelliteThermalCollector,
    ThermalAnomaly,
    AnomalyAnalyzer
)

# Telegram Scrapers
from .telegram_document_scraper import (
    EnhancedSecurityScraper,
    SecurityDocument
)

from .vxunderground_archive_downloader import (
    VXUndergroundDownloader,
    ResearchPaper
)

__all__ = [
    # OSINT Collectors
    'OSINTFeedAggregator',
    'ThreatIntelItem',
    'IOCExtractor',
    'MilitaryOSINTAggregator',
    'MilitaryIntelItem',
    'MilitaryNewsCollector',
    'AircraftTrackingCollector',
    'NavalTrackingCollector',
    'SatelliteThermalCollector',
    'ThermalAnomaly',
    'AnomalyAnalyzer',
    # Telegram Scrapers
    'EnhancedSecurityScraper',
    'SecurityDocument',
    'VXUndergroundDownloader',
    'ResearchPaper',
]

__version__ = '1.0.0'
__author__ = 'LAT5150DRVMIL AI Framework'
__classification__ = 'UNCLASSIFIED // FOR OFFICIAL USE ONLY'
