"""
LAT5150DRVMIL Screenshot Intelligence Module

AI-driven screenshot analysis with OCR and vector search.

Components:
- VectorRAGSystem: Qdrant vector database with BAAI embeddings
- ScreenshotIntelligence: Core screenshot analysis
- AIAnalysisLayer: AI-powered event correlation
- SystemHealthMonitor: Monitoring and maintenance
- ResilienceUtils: Error handling patterns

Usage:
    from LAT5150DRVMIL.screenshot_intel import ScreenshotIntelligence

    intel = ScreenshotIntelligence()
    intel.ingest_screenshot("/path/to/screenshot.png")
    results = intel.rag.search("error message")
"""

import sys
from pathlib import Path

# Add RAG system directory to path
RAG_SYSTEM_DIR = Path(__file__).parent.parent / "04-integrations" / "rag_system"
sys.path.insert(0, str(RAG_SYSTEM_DIR))

# Conditional imports
try:
    from vector_rag_system import VectorRAGSystem, Document
except ImportError:
    VectorRAGSystem = None
    Document = None

try:
    from screenshot_intelligence import ScreenshotIntelligence, Event, Incident
except ImportError:
    ScreenshotIntelligence = None
    Event = None
    Incident = None

try:
    from ai_analysis_layer import AIAnalysisLayer
except ImportError:
    AIAnalysisLayer = None

try:
    from telegram_integration import TelegramIntegration
except ImportError:
    TelegramIntegration = None

try:
    from signal_integration import SignalIntegration
except ImportError:
    SignalIntegration = None

try:
    from system_health_monitor import SystemHealthMonitor
except ImportError:
    SystemHealthMonitor = None

try:
    from resilience_utils import (
        with_retry,
        CircuitBreaker,
        FallbackHandler,
        RateLimiter
    )
except ImportError:
    with_retry = None
    CircuitBreaker = None
    FallbackHandler = None
    RateLimiter = None

__all__ = [
    'VectorRAGSystem',
    'Document',
    'ScreenshotIntelligence',
    'Event',
    'Incident',
    'AIAnalysisLayer',
    'TelegramIntegration',
    'SignalIntegration',
    'SystemHealthMonitor',
    'with_retry',
    'CircuitBreaker',
    'FallbackHandler',
    'RateLimiter',
]
