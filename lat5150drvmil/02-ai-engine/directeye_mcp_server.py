#!/usr/bin/env python3
"""
DIRECTEYE MCP Server
Exposes DIRECTEYE Intelligence Platform via Model Context Protocol (MCP)

Features:
- 40+ OSINT services (people, breaches, corporate intel)
- 12+ blockchain networks (crypto analysis)
- Threat intelligence (IOC lookups)
- 35+ MCP AI tools (entity resolution, risk scoring)
- ML analytics (5 engines)
- CPU optimization (AVX512/AVX2)

MCP Protocol: Standard tool calling interface for AI systems
Source: ai_engine/directeye_intelligence.py
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add DIRECTEYE to path
AI_ENGINE_PATH = Path(__file__).parent.parent / "ai_engine"
sys.path.insert(0, str(AI_ENGINE_PATH))

try:
    from directeye_intelligence import DirectEyeIntelligence
    DIRECTEYE_AVAILABLE = True
except ImportError:
    DIRECTEYE_AVAILABLE = False


class DirectEyeMCPServer:
    """
    MCP Server wrapper for DIRECTEYE Intelligence Platform

    Provides standardized tool interface for:
    - OSINT queries
    - Blockchain analysis
    - Threat intelligence
    - MCP tool execution
    - Service management
    """

    def __init__(self):
        """Initialize DIRECTEYE MCP server"""
        if not DIRECTEYE_AVAILABLE:
            raise ImportError(
                "DIRECTEYE Intelligence not available. "
                "Check ai_engine/directeye_intelligence.py"
            )

        self.intel = DirectEyeIntelligence()
        self.server_name = "directeye"
        self.version = "1.0.0"

    async def handle_tool_call(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """
        Handle MCP tool call

        Args:
            tool_name: Tool to invoke
            params: Tool parameters

        Returns:
            Tool execution result
        """
        try:
            if tool_name == "osint_query":
                return await self._osint_query(params)

            elif tool_name == "blockchain_analyze":
                return await self._blockchain_analyze(params)

            elif tool_name == "threat_intelligence":
                return await self._threat_intelligence(params)

            elif tool_name == "get_mcp_tools":
                return await self._get_mcp_tools(params)

            elif tool_name == "mcp_tool_execute":
                return await self._mcp_tool_execute(params)

            elif tool_name == "get_available_services":
                return self._get_available_services()

            elif tool_name == "get_service_status":
                return self._get_service_status()

            elif tool_name == "get_cpu_capabilities":
                return self._get_cpu_capabilities()

            else:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [t["name"] for t in self.get_tool_definitions()]
                }

        except Exception as e:
            return {
                "error": str(e),
                "tool": tool_name,
                "params": params
            }

    async def _osint_query(self, params: Dict) -> Dict:
        """
        OSINT query across 40+ services

        Params:
            query: Search query (email, name, phone, domain, etc.)
            services: Optional list of specific services

        Services:
            - truepeoplesearch, hunter, emailrep (people search)
            - hibp, snusbase, leakosint (breach data)
            - sec_edgar, companies_house, icij (corporate)
            - alienvault, censys, fofa (threat intel)
        """
        query = params.get("query")
        services = params.get("services")

        if not query:
            return {"error": "Missing required parameter: query"}

        result = await self.intel.osint_query(query, services)

        return {
            "tool": "osint_query",
            "query": query,
            "services_queried": services or "all",
            "result": result
        }

    async def _blockchain_analyze(self, params: Dict) -> Dict:
        """
        Blockchain address analysis across 12+ chains

        Params:
            address: Blockchain address
            chain: Network (ethereum, bitcoin, polygon, etc.)

        Chains:
            ethereum, bitcoin, polygon, avalanche, arbitrum, optimism,
            base, binance_smart_chain, fantom, cronos, gnosis, moonbeam
        """
        address = params.get("address")
        chain = params.get("chain", "ethereum")

        if not address:
            return {"error": "Missing required parameter: address"}

        result = await self.intel.blockchain_analyze(address, chain)

        return {
            "tool": "blockchain_analyze",
            "address": address,
            "chain": chain,
            "result": result
        }

    async def _threat_intelligence(self, params: Dict) -> Dict:
        """
        Threat intelligence lookup for IOCs

        Params:
            indicator: IP, domain, hash, URL, email
            indicator_type: Type (auto-detected if not specified)
        """
        indicator = params.get("indicator")
        indicator_type = params.get("indicator_type", "auto")

        if not indicator:
            return {"error": "Missing required parameter: indicator"}

        result = await self.intel.threat_intelligence(indicator, indicator_type)

        return {
            "tool": "threat_intelligence",
            "indicator": indicator,
            "indicator_type": indicator_type,
            "result": result
        }

    async def _get_mcp_tools(self, params: Dict) -> Dict:
        """Get list of available MCP tools (35+)"""
        tools = self.intel.get_mcp_tools()

        return {
            "tool": "get_mcp_tools",
            "tools": tools,
            "count": len(tools)
        }

    async def _mcp_tool_execute(self, params: Dict) -> Dict:
        """
        Execute an MCP tool

        Params:
            tool_name: Tool to execute
            tool_params: Parameters for the tool
        """
        tool_name = params.get("tool_name")
        tool_params = params.get("tool_params", {})

        if not tool_name:
            return {"error": "Missing required parameter: tool_name"}

        result = await self.intel.mcp_tool_execute(tool_name, tool_params)

        return {
            "tool": "mcp_tool_execute",
            "tool_name": tool_name,
            "result": result
        }

    def _get_available_services(self) -> Dict:
        """Get all available services by category"""
        services = self.intel.get_available_services()

        return {
            "tool": "get_available_services",
            "services": services
        }

    def _get_service_status(self) -> Dict:
        """Get service status"""
        status = self.intel.get_service_status()

        return {
            "tool": "get_service_status",
            "status": status
        }

    def _get_cpu_capabilities(self) -> Dict:
        """Get CPU capabilities (AVX512/AVX2)"""
        caps = self.intel.cpu_capabilities

        return {
            "tool": "get_cpu_capabilities",
            "capabilities": caps
        }

    def get_tool_definitions(self) -> List[Dict]:
        """
        Get MCP tool definitions

        Returns tool metadata for AI systems to understand available capabilities.
        """
        return [
            {
                "name": "osint_query",
                "description": "Query 40+ OSINT services (people search, breaches, corporate intel, govt data)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (email, name, phone, domain, IP, etc.)"
                        },
                        "services": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of specific services to query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "blockchain_analyze",
                "description": "Analyze blockchain address across 12+ networks (Ethereum, Bitcoin, Polygon, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "Blockchain address to analyze"
                        },
                        "chain": {
                            "type": "string",
                            "description": "Blockchain network (default: ethereum)",
                            "enum": [
                                "ethereum", "bitcoin", "polygon", "avalanche",
                                "arbitrum", "optimism", "base", "binance_smart_chain",
                                "fantom", "cronos", "gnosis", "moonbeam"
                            ]
                        }
                    },
                    "required": ["address"]
                }
            },
            {
                "name": "threat_intelligence",
                "description": "Lookup threat intelligence for indicators of compromise (IOCs)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "indicator": {
                            "type": "string",
                            "description": "IOC to lookup (IP, domain, hash, URL, email)"
                        },
                        "indicator_type": {
                            "type": "string",
                            "description": "Type of indicator (auto-detected if not specified)",
                            "enum": ["auto", "ip", "domain", "hash", "url", "email"]
                        }
                    },
                    "required": ["indicator"]
                }
            },
            {
                "name": "get_mcp_tools",
                "description": "Get list of 35+ available MCP AI tools (entity resolution, risk scoring, pattern detection)",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "mcp_tool_execute",
                "description": "Execute a specific MCP AI tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of MCP tool to execute"
                        },
                        "tool_params": {
                            "type": "object",
                            "description": "Parameters for the tool"
                        }
                    },
                    "required": ["tool_name"]
                }
            },
            {
                "name": "get_available_services",
                "description": "Get all available services organized by category (OSINT, blockchain, threat intel, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_service_status",
                "description": "Get DIRECTEYE service status (backend API, MCP server, CPU capabilities)",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_cpu_capabilities",
                "description": "Get CPU capabilities (architecture, cores, AVX512/AVX2 support, SIMD status)",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def get_server_info(self) -> Dict:
        """Get server information"""
        return {
            "name": self.server_name,
            "version": self.version,
            "description": "DIRECTEYE Intelligence Platform MCP Server",
            "capabilities": {
                "osint_services": "40+",
                "blockchain_networks": "12+",
                "mcp_tools": "35+",
                "analytics_engines": "5",
                "cpu_optimization": "AVX512/AVX2"
            },
            "tools": [t["name"] for t in self.get_tool_definitions()]
        }


async def serve_stdio():
    """
    Serve MCP server via STDIO

    This is the standard MCP protocol for tool execution.
    AI systems communicate via stdin/stdout with JSON messages.
    """
    server = DirectEyeMCPServer()

    print(json.dumps({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": server.get_server_info()
    }), flush=True)

    # Main loop: read requests from stdin, write responses to stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method == "tools/list":
                # List available tools
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": server.get_tool_definitions()
                    }
                }

            elif method == "tools/call":
                # Call a tool
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                result = await server.handle_tool_call(tool_name, tool_params)

                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }

            elif method == "ping":
                # Health check
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"status": "ok"}
                }

            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}"
                }
            }
            print(json.dumps(error_response), flush=True)

        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {e}"
                }
            }
            print(json.dumps(error_response), flush=True)


async def demo():
    """Demo usage"""
    print("=== DIRECTEYE MCP Server Demo ===\n")

    if not DIRECTEYE_AVAILABLE:
        print("❌ DIRECTEYE Intelligence not available")
        print("   Check: ai_engine/directeye_intelligence.py")
        return

    server = DirectEyeMCPServer()

    # Server info
    print("1. Server Info:")
    info = server.get_server_info()
    print(json.dumps(info, indent=2))

    # Available tools
    print("\n2. Available Tools:")
    tools = server.get_tool_definitions()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")

    # Test tool calls
    print("\n3. Testing tool calls...")

    # Get available services
    print("\n   a) Get available services:")
    result = server._get_available_services()
    services = result['services']
    print(f"      OSINT services: {len(services.get('osint', {}))} categories")
    print(f"      Blockchain networks: {len(services.get('blockchain', []))}")

    # Get service status
    print("\n   b) Get service status:")
    result = server._get_service_status()
    print(json.dumps(result, indent=2))

    # CPU capabilities
    print("\n   c) CPU capabilities:")
    result = server._get_cpu_capabilities()
    caps = result['capabilities']
    print(f"      Architecture: {caps.get('arch', 'unknown')}")
    print(f"      Cores: {caps.get('cores', 'unknown')}")
    print(f"      AVX512: {caps.get('avx512', False)}")
    print(f"      AVX2: {caps.get('avx2', False)}")

    print("\n✅ Demo complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo
        asyncio.run(demo())
    else:
        # Run MCP server (stdio mode)
        asyncio.run(serve_stdio())
