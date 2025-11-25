"""
LAT5150DRVMIL - Dell Latitude 5450 Covert AI Platform
Military-Grade AI System with Screenshot Intelligence

Main entry point for LAT5150DRVMIL as a Python package/submodule.

Components:
- DSMIL AI Engine: Local-first LLM system
- Unified Orchestrator: Multi-backend AI routing
- MCP Servers: 12 specialized AI servers
- Screenshot Intelligence: AI-driven screenshot analysis
- Hardware Integration: NPU, TPM 2.0, PQC support
- RAG System: Vector database with Qdrant

Usage as Submodule:
    from LAT5150DRVMIL import DSMILSystem

    system = DSMILSystem()
    response = system.generate("Your prompt here")

Usage for Integration:
    from LAT5150DRVMIL.ai_engine import DSMILAIEngine
    from LAT5150DRVMIL.orchestrator import UnifiedAIOrchestrator
    from LAT5150DRVMIL.screenshot_intel import ScreenshotIntelligence
"""

__version__ = "1.0.0"
__author__ = "LAT5150DRVMIL Project"
__license__ = "Proprietary"

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02-ai-engine"))
sys.path.insert(0, str(PROJECT_ROOT / "04-integrations" / "rag_system"))

# Core imports with graceful fallback
try:
    from LAT5150DRVMIL.ai_engine.dsmil_ai_engine import DSMILAIEngine
    DSMIL_AVAILABLE = True
except ImportError:
    DSMIL_AVAILABLE = False
    DSMILAIEngine = None

try:
    from LAT5150DRVMIL.ai_engine.unified_orchestrator import UnifiedAIOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    UnifiedAIOrchestrator = None

try:
    from LAT5150DRVMIL.screenshot_intel.screenshot_intelligence import ScreenshotIntelligence
    from LAT5150DRVMIL.screenshot_intel.vector_rag_system import VectorRAGSystem
    SCREENSHOT_INTEL_AVAILABLE = True
except ImportError:
    SCREENSHOT_INTEL_AVAILABLE = False
    ScreenshotIntelligence = None
    VectorRAGSystem = None


class DSMILSystem:
    """
    Main system interface for LAT5150DRVMIL platform

    Provides unified access to all subsystems:
    - AI Engine (DSMIL)
    - Unified Orchestrator
    - Screenshot Intelligence
    - Hardware Integration

    Example:
        >>> system = DSMILSystem()
        >>> response = system.generate("Explain quantum computing")
        >>> print(response)
    """

    def __init__(
        self,
        enable_orchestrator: bool = False,
        enable_screenshot_intel: bool = False
    ):
        """
        Initialize DSMIL system

        Args:
            enable_orchestrator: Use unified orchestrator (Gemini/OpenAI fallback)
            enable_screenshot_intel: Enable screenshot intelligence
        """
        self.components = {}

        # Core AI Engine (always available)
        if DSMIL_AVAILABLE:
            self.ai_engine = DSMILAIEngine()
            self.components['ai_engine'] = self.ai_engine
        else:
            self.ai_engine = None

        # Unified Orchestrator (optional)
        if enable_orchestrator and ORCHESTRATOR_AVAILABLE:
            self.orchestrator = UnifiedAIOrchestrator()
            self.components['orchestrator'] = self.orchestrator
        else:
            self.orchestrator = None

        # Screenshot Intelligence (optional)
        if enable_screenshot_intel and SCREENSHOT_INTEL_AVAILABLE:
            self.screenshot_intel = ScreenshotIntelligence()
            self.components['screenshot_intel'] = self.screenshot_intel
        else:
            self.screenshot_intel = None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate AI response

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments (model, stream, etc.)

        Returns:
            Generated response
        """
        if self.orchestrator:
            return self.orchestrator.generate(prompt, **kwargs)
        elif self.ai_engine:
            return self.ai_engine.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No AI engine available")

    def get_status(self) -> dict:
        """Get system status"""
        return {
            'version': __version__,
            'components': list(self.components.keys()),
            'dsmil_available': DSMIL_AVAILABLE,
            'orchestrator_available': ORCHESTRATOR_AVAILABLE,
            'screenshot_intel_available': SCREENSHOT_INTEL_AVAILABLE,
        }


# Convenience exports
__all__ = [
    'DSMILSystem',
    'DSMILAIEngine',
    'UnifiedAIOrchestrator',
    'ScreenshotIntelligence',
    'VectorRAGSystem',
    '__version__',
]
