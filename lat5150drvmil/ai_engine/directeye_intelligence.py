#!/usr/bin/env python3
"""
DIRECTEYE Intelligence Integration for LAT5150DRVMIL AI Engine

Provides AI engine with comprehensive OSINT, blockchain, and threat intelligence
capabilities via the DIRECTEYE platform.

This module integrates DIRECTEYE's 40+ intelligence services into the AI engine,
enabling:
- OSINT queries (people search, breach data, corporate intel)
- Blockchain intelligence (12+ chains)
- Threat intelligence (AlienVault OTX, etc.)
- MCP Server integration (35+ AI tools)
- Native binary protocol with AVX2/AVX512 optimization
- ML-powered analytics (5 engines)

This is a convenience wrapper that delegates to the main integration at:
    06-intel-systems/directeye_intelligence_integration.py

Usage:
    from LAT5150DRVMIL.ai_engine import DirectEyeIntelligence

    intel = DirectEyeIntelligence()
    results = await intel.osint_query("target@example.com")
    blockchain_info = await intel.blockchain_analyze("0xAddress...")
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add both DIRECTEYE and intel-systems to Python path
DIRECTEYE_PATH = Path(__file__).parent.parent / "rag_system" / "mcp_servers" / "DIRECTEYE"
INTEL_SYSTEMS_PATH = Path(__file__).parent.parent / "06-intel-systems"

if DIRECTEYE_PATH.exists():
    sys.path.insert(0, str(DIRECTEYE_PATH))

if INTEL_SYSTEMS_PATH.exists():
    sys.path.insert(0, str(INTEL_SYSTEMS_PATH))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectEyeIntelligence:
    """
    DIRECTEYE Intelligence wrapper for AI Engine integration.

    Provides unified access to all DIRECTEYE capabilities including
    OSINT, blockchain analysis, threat intelligence, and MCP tools.

    This wrapper delegates to the main intelligence integration at:
        06-intel-systems/directeye_intelligence_integration.py

    It provides both the full-featured integration and convenience methods
    for quick access from the AI engine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DIRECTEYE Intelligence integration.

        Args:
            config: Optional configuration dict. Defaults will be loaded from
                   DIRECTEYE's config/directeye_config.yaml
        """
        self.config = config or {}
        self.directeye_path = DIRECTEYE_PATH
        self._orchestrator = None
        self._osint_service = None
        self._mcp_tools = None
        self._cpu_capabilities = None
        self._full_integration = None

        # Check if DIRECTEYE is available
        if not self.directeye_path.exists():
            logger.error(f"DIRECTEYE not found at {self.directeye_path}")
            raise ImportError("DIRECTEYE submodule not initialized")

        # Try to load the full integration from intel-systems
        try:
            from directeye_intelligence_integration import DirectEyeIntelligence as FullIntegration
            self._full_integration = FullIntegration(config_path=None)
            logger.info("Loaded full DIRECTEYE integration from 06-intel-systems")
        except ImportError:
            logger.warning("Full integration not available, using direct access")

        logger.info(f"DIRECTEYE Intelligence initialized from {self.directeye_path}")

    @property
    def orchestrator(self):
        """Lazy-load DIRECTEYE orchestrator."""
        if self._orchestrator is None:
            try:
                from directeye_main import DirectEyeOrchestrator
                self._orchestrator = DirectEyeOrchestrator()
                logger.info("DIRECTEYE Orchestrator loaded")
            except ImportError as e:
                logger.error(f"Failed to load DIRECTEYE Orchestrator: {e}")
                raise
        return self._orchestrator

    @property
    def cpu_capabilities(self):
        """Get system CPU capabilities (AVX512/AVX2 detection)."""
        if self._cpu_capabilities is None:
            try:
                from directeye_main import CPUDetector
                detector = CPUDetector()
                self._cpu_capabilities = {
                    'arch': detector.capabilities.cpu_arch.value,
                    'cores': detector.capabilities.total_cores,
                    'p_cores': detector.capabilities.p_cores,
                    'e_cores': detector.capabilities.e_cores,
                    'avx512': detector.capabilities.avx512_available,
                    'avx2': detector.capabilities.avx2_available,
                    'simd_enabled': detector.capabilities.simd_enabled,
                }
                logger.info(f"CPU Capabilities: {self._cpu_capabilities['arch']}")
            except ImportError as e:
                logger.warning(f"Could not detect CPU capabilities: {e}")
                self._cpu_capabilities = {}
        return self._cpu_capabilities

    async def start_services(self,
                           backend: bool = True,
                           mcp_server: bool = True,
                           enable_simd: bool = True):
        """
        Start DIRECTEYE services.

        Args:
            backend: Start backend API (port 8000)
            mcp_server: Start MCP server (port 8001)
            enable_simd: Enable SIMD optimizations (AVX2/AVX512)
        """
        orch = self.orchestrator

        if backend:
            orch.services['backend_api'].auto_start = True
            logger.info("Backend API enabled on port 8000")

        if mcp_server:
            orch.services['mcp_server'].auto_start = True
            logger.info("MCP Server enabled on port 8001")

        # Start services
        await orch.start_all(enable_simd=enable_simd)
        logger.info("DIRECTEYE services started successfully")

    async def stop_services(self):
        """Stop all DIRECTEYE services gracefully."""
        if self._orchestrator:
            await self._orchestrator.stop_all()
            logger.info("DIRECTEYE services stopped")

    async def osint_query(self,
                         query: str,
                         services: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform OSINT query across multiple intelligence services.

        Args:
            query: Search query (email, name, phone, domain, etc.)
            services: Optional list of specific services to query.
                     If None, queries all available services.

        Returns:
            Dict containing results from each service

        Available services:
            - 'truepeoplesearch': People search
            - 'hunter': Email finding
            - 'emailrep': Email reputation
            - 'hibp': Have I Been Pwned
            - 'snusbase': Breach data
            - 'leakosint': Leak data
            - 'sec_edgar': SEC filings
            - 'companies_house': UK companies
            - 'icij': Offshore leaks
            - 'alienvault': Threat intelligence
            - And 30+ more...
        """
        # Use full integration if available
        if self._full_integration:
            return await self._full_integration.osint_query(query, services)

        # Fallback to direct access
        try:
            sys.path.insert(0, str(self.directeye_path / "us_lookup" / "mcp_server"))
            from services.osint_service import OSINTService

            if self._osint_service is None:
                self._osint_service = OSINTService()

            results = await self._osint_service.universal_query(query, services=services)
            logger.info(f"OSINT query completed for: {query}")
            return results

        except Exception as e:
            logger.error(f"OSINT query failed: {e}")
            return {"error": str(e)}

    async def blockchain_analyze(self,
                                address: str,
                                chain: str = "ethereum") -> Dict[str, Any]:
        """
        Analyze blockchain address across supported chains.

        Args:
            address: Blockchain address to analyze
            chain: Blockchain network (ethereum, bitcoin, polygon, etc.)
                  Supports 12+ chains

        Returns:
            Dict containing blockchain intelligence data
        """
        try:
            sys.path.insert(0, str(self.directeye_path / "us_lookup" / "mcp_server"))
            from services.blockchain_service import BlockchainService

            blockchain_svc = BlockchainService()
            results = await blockchain_svc.analyze_address(address, chain)
            logger.info(f"Blockchain analysis completed for {chain}:{address}")
            return results

        except Exception as e:
            logger.error(f"Blockchain analysis failed: {e}")
            return {"error": str(e)}

    async def threat_intelligence(self,
                                 indicator: str,
                                 indicator_type: str = "auto") -> Dict[str, Any]:
        """
        Query threat intelligence services for IoCs.

        Args:
            indicator: IP, domain, hash, URL, etc.
            indicator_type: Type of indicator (auto-detected if not specified)
                           Options: ip, domain, hash, url, email

        Returns:
            Dict containing threat intelligence data
        """
        try:
            sys.path.insert(0, str(self.directeye_path / "us_lookup" / "mcp_server"))
            from services.threat_intel_service import ThreatIntelService

            threat_svc = ThreatIntelService()
            results = await threat_svc.query_indicator(indicator, indicator_type)
            logger.info(f"Threat intel query completed for: {indicator}")
            return results

        except Exception as e:
            logger.error(f"Threat intelligence query failed: {e}")
            return {"error": str(e)}

    def get_mcp_tools(self) -> List[str]:
        """
        Get list of available MCP tools (35+ tools).

        Returns:
            List of MCP tool names
        """
        try:
            sys.path.insert(0, str(self.directeye_path / "core"))
            from mcp_server.tools import get_available_tools

            tools = get_available_tools()
            logger.info(f"Found {len(tools)} MCP tools")
            return tools

        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return []

    async def mcp_tool_execute(self,
                              tool_name: str,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool.

        Args:
            tool_name: Name of the MCP tool to execute
            params: Parameters for the tool

        Returns:
            Tool execution results
        """
        try:
            sys.path.insert(0, str(self.directeye_path / "core"))
            from mcp_server.executor import execute_tool

            result = await execute_tool(tool_name, params)
            logger.info(f"MCP tool '{tool_name}' executed successfully")
            return result

        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return {"error": str(e)}

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all DIRECTEYE services.

        Returns:
            Dict with service status information
        """
        if self._orchestrator is None:
            return {"status": "not_initialized"}

        return {
            "backend_api": {
                "enabled": self.orchestrator.services['backend_api'].enabled,
                "port": self.orchestrator.services['backend_api'].port,
                "status": "running" if self.orchestrator.services['backend_api'].auto_start else "stopped"
            },
            "mcp_server": {
                "enabled": self.orchestrator.services['mcp_server'].enabled,
                "port": self.orchestrator.services['mcp_server'].port,
                "status": "running" if self.orchestrator.services['mcp_server'].auto_start else "stopped"
            },
            "cpu_capabilities": self.cpu_capabilities,
        }

    def get_available_services(self) -> Dict[str, List[str]]:
        """
        Get list of all available intelligence services.

        Returns:
            Dict categorizing all 40+ available services
        """
        return {
            "osint": {
                "people_search": [
                    "truepeoplesearch",
                    "hunter.io",
                    "emailrep"
                ],
                "breach_data": [
                    "hibp",
                    "spycloud",
                    "snusbase",
                    "leakosint",
                    "hackcheck"
                ],
                "corporate": [
                    "sec_edgar",
                    "companies_house",
                    "icij_offshore"
                ],
                "government_data": [
                    "data.gov",
                    "socrata",
                    "ckan",
                    "dkan"
                ],
                "threat_intel": [
                    "alienvault_otx",
                    "censys",
                    "fofa",
                    "ipgeolocation"
                ]
            },
            "blockchain": [
                "ethereum",
                "bitcoin",
                "polygon",
                "avalanche",
                "arbitrum",
                "optimism",
                "base",
                "binance_smart_chain",
                "fantom",
                "cronos",
                "gnosis",
                "moonbeam"
            ],
            "mcp_tools": [
                "35+ AI-powered tools",
                "Entity resolution",
                "Risk scoring",
                "Pattern detection",
                "Natural language queries"
            ],
            "analytics": [
                "ML risk scoring",
                "Entity correlation",
                "Pattern detection",
                "Anomaly detection",
                "Confidence scoring"
            ]
        }

    def __repr__(self) -> str:
        return (
            f"DirectEyeIntelligence(\n"
            f"  path='{self.directeye_path}',\n"
            f"  cpu='{self.cpu_capabilities.get('arch', 'unknown')}',\n"
            f"  services=40+,\n"
            f"  mcp_tools=35+\n"
            f")"
        )


# Convenience functions for quick access
async def quick_osint(query: str) -> Dict[str, Any]:
    """Quick OSINT query without instantiating class."""
    intel = DirectEyeIntelligence()
    return await intel.osint_query(query)


async def quick_blockchain(address: str, chain: str = "ethereum") -> Dict[str, Any]:
    """Quick blockchain analysis without instantiating class."""
    intel = DirectEyeIntelligence()
    return await intel.blockchain_analyze(address, chain)


async def quick_threat_intel(indicator: str) -> Dict[str, Any]:
    """Quick threat intelligence query without instantiating class."""
    intel = DirectEyeIntelligence()
    return await intel.threat_intelligence(indicator)


# Module-level exports
__all__ = [
    'DirectEyeIntelligence',
    'quick_osint',
    'quick_blockchain',
    'quick_threat_intel',
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("=== DIRECTEYE Intelligence Integration Demo ===\n")

        intel = DirectEyeIntelligence()
        print(intel)
        print("\n=== Available Services ===")
        services = intel.get_available_services()
        for category, items in services.items():
            print(f"\n{category.upper()}:")
            if isinstance(items, dict):
                for subcat, services_list in items.items():
                    print(f"  {subcat}: {', '.join(services_list)}")
            else:
                print(f"  {', '.join(items)}")

        print("\n=== CPU Capabilities ===")
        caps = intel.cpu_capabilities
        for key, value in caps.items():
            print(f"  {key}: {value}")

        print("\n=== Service Status ===")
        status = intel.get_service_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

    asyncio.run(demo())
