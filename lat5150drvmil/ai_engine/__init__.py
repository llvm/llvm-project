"""
LAT5150DRVMIL AI Engine Module

Core AI components for local-first LLM inference with DIRECTEYE Intelligence integration.

Components:
- DSMILAIEngine: Main AI engine with 5 models
- UnifiedAIOrchestrator: Multi-backend routing
- MCP Servers: Model Context Protocol servers
- Hardware Integration: NPU, TPM support
- DirectEyeIntelligence: OSINT, blockchain, and threat intelligence (40+ services)

Usage:
    from LAT5150DRVMIL.ai_engine import DSMILAIEngine, DirectEyeIntelligence

    # AI Engine
    engine = DSMILAIEngine()
    response = engine.generate("Your prompt")

    # Intelligence
    intel = DirectEyeIntelligence()
    osint_results = await intel.osint_query("target@example.com")
"""

import sys
from pathlib import Path

# Add ai-engine directory to path
AI_ENGINE_DIR = Path(__file__).parent.parent / "02-ai-engine"
sys.path.insert(0, str(AI_ENGINE_DIR))

# Conditional imports with graceful fallback
try:
    from dsmil_ai_engine import DSMILAIEngine
except ImportError:
    DSMILAIEngine = None

try:
    from unified_orchestrator import UnifiedAIOrchestrator
except ImportError:
    UnifiedAIOrchestrator = None

try:
    from system_validator import SystemValidator
except ImportError:
    SystemValidator = None

# DSMIL Hardware Activation Components
try:
    from dsmil_integrated_activation import DSMILIntegratedActivation
except ImportError:
    DSMILIntegratedActivation = None

try:
    from dsmil_device_activation import DSMILDeviceActivator
except ImportError:
    DSMILDeviceActivator = None

try:
    from dsmil_ml_discovery import DSMILMLDiscovery
except ImportError:
    DSMILMLDiscovery = None

try:
    from dsmil_hardware_analyzer import DSMILHardwareAnalyzer
except ImportError:
    DSMILHardwareAnalyzer = None

try:
    from dsmil_subsystem_controller import DSMILSubsystemController
except ImportError:
    DSMILSubsystemController = None

# Import DIRECTEYE Intelligence
try:
    from .directeye_intelligence import DirectEyeIntelligence
    DIRECTEYE_AVAILABLE = True
except ImportError:
    DirectEyeIntelligence = None
    DIRECTEYE_AVAILABLE = False

__all__ = [
    'DSMILAIEngine',
    'UnifiedAIOrchestrator',
    'SystemValidator',
    'DSMILIntegratedActivation',
    'DSMILDeviceActivator',
    'DSMILMLDiscovery',
    'DSMILHardwareAnalyzer',
    'DSMILSubsystemController',
    'DirectEyeIntelligence',
    'DIRECTEYE_AVAILABLE',
]
