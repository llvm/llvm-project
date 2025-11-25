#!/usr/bin/env python3
"""
Screenshot Intelligence MCP Server
MCP (Model Context Protocol) server for screenshot and intelligence analysis

Provides tools for:
- Screenshot ingestion and OCR
- Chat log correlation
- Timeline queries
- Event linking
- Incident management

Integration:
- Compatible with Claude Code and other MCP clients
- Uses Vector RAG System
- Integrates with DSMIL AI Engine
- SWORD Intelligence compatible
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations" / "rag_system"))

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP SDK not available")

# Screenshot Intelligence
try:
    from vector_rag_system import VectorRAGSystem
    from screenshot_intelligence import ScreenshotIntelligence
except ImportError as e:
    print(f"⚠️  Screenshot Intelligence not available: {e}")
    ScreenshotIntelligence = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenshotIntelMCPServer:
    """MCP Server for Screenshot Intelligence"""

    def __init__(self):
        """Initialize Screenshot Intelligence MCP Server"""
        self.server = Server("screenshot-intelligence")

        # Initialize Screenshot Intelligence
        try:
            self.intel = ScreenshotIntelligence()
            logger.info("✓ Screenshot Intelligence initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Screenshot Intelligence: {e}")
            self.intel = None

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools"""

        # Tool: Ingest Screenshot
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="ingest_screenshot",
                    description="Ingest a screenshot with OCR and metadata extraction",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "screenshot_path": {
                                "type": "string",
                                "description": "Path to screenshot file"
                            },
                            "device_id": {
                                "type": "string",
                                "description": "Device identifier (optional)"
                            }
                        },
                        "required": ["screenshot_path"]
                    }
                ),
                Tool(
                    name="scan_device",
                    description="Scan and ingest all screenshots from a registered device",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "device_id": {
                                "type": "string",
                                "description": "Device identifier"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "File pattern (default: *.png)"
                            }
                        },
                        "required": ["device_id"]
                    }
                ),
                Tool(
                    name="search_intel",
                    description="Search screenshot and chat intelligence with semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum results (default: 10)"
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "Filter by type: image, chat_message, pdf, text"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="timeline_query",
                    description="Query events by timeline (date range)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD HH:MM:SS)"
                            },
                            "doc_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by document types"
                            }
                        },
                        "required": ["start_date", "end_date"]
                    }
                ),
                Tool(
                    name="generate_timeline_report",
                    description="Generate a timeline report for a date range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD HH:MM:SS)"
                            },
                            "format": {
                                "type": "string",
                                "description": "Output format: markdown or json",
                                "default": "markdown"
                            }
                        },
                        "required": ["start_date", "end_date"]
                    }
                ),
                Tool(
                    name="register_device",
                    description="Register a new device for screenshot scanning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "device_id": {
                                "type": "string",
                                "description": "Unique device identifier"
                            },
                            "device_name": {
                                "type": "string",
                                "description": "Human-readable device name"
                            },
                            "device_type": {
                                "type": "string",
                                "description": "Device type: grapheneos, laptop, pc"
                            },
                            "screenshot_path": {
                                "type": "string",
                                "description": "Path to screenshot directory"
                            }
                        },
                        "required": ["device_id", "device_name", "device_type", "screenshot_path"]
                    }
                ),
                Tool(
                    name="get_stats",
                    description="Get Screenshot Intelligence statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""

            if not self.intel:
                return [TextContent(
                    type="text",
                    text="❌ Screenshot Intelligence not initialized"
                )]

            try:
                if name == "ingest_screenshot":
                    result = self.intel.ingest_screenshot(
                        Path(arguments["screenshot_path"]),
                        device_id=arguments.get("device_id")
                    )
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]

                elif name == "scan_device":
                    result = self.intel.scan_device_screenshots(
                        arguments["device_id"],
                        pattern=arguments.get("pattern", "*.png")
                    )
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]

                elif name == "search_intel":
                    filters = {}
                    if "doc_type" in arguments:
                        filters["type"] = arguments["doc_type"]

                    results = self.intel.rag.search(
                        arguments["query"],
                        limit=arguments.get("limit", 10),
                        filters=filters if filters else None
                    )

                    output = {
                        "query": arguments["query"],
                        "total_results": len(results),
                        "results": []
                    }

                    for result in results:
                        output["results"].append({
                            "score": result.score,
                            "type": result.document.doc_type,
                            "filename": result.document.filename,
                            "timestamp": result.document.timestamp.isoformat(),
                            "text_preview": result.document.text[:200],
                            "metadata": result.document.metadata
                        })

                    return [TextContent(
                        type="text",
                        text=json.dumps(output, indent=2)
                    )]

                elif name == "timeline_query":
                    start_time = datetime.fromisoformat(arguments["start_date"])
                    end_time = datetime.fromisoformat(arguments["end_date"])

                    events = self.intel.rag.timeline_query(
                        start_time,
                        end_time,
                        doc_types=arguments.get("doc_types")
                    )

                    output = {
                        "start_date": arguments["start_date"],
                        "end_date": arguments["end_date"],
                        "total_events": len(events),
                        "events": []
                    }

                    for event in events:
                        output["events"].append({
                            "timestamp": event.timestamp.isoformat(),
                            "type": event.doc_type,
                            "filename": event.filename,
                            "text_preview": event.text[:150],
                            "metadata": event.metadata
                        })

                    return [TextContent(
                        type="text",
                        text=json.dumps(output, indent=2)
                    )]

                elif name == "generate_timeline_report":
                    start_time = datetime.fromisoformat(arguments["start_date"])
                    end_time = datetime.fromisoformat(arguments["end_date"])
                    format_type = arguments.get("format", "markdown")

                    report = self.intel.generate_timeline_report(
                        start_time,
                        end_time,
                        output_format=format_type
                    )

                    return [TextContent(
                        type="text",
                        text=report
                    )]

                elif name == "register_device":
                    self.intel.register_device(
                        device_id=arguments["device_id"],
                        device_name=arguments["device_name"],
                        device_type=arguments["device_type"],
                        screenshot_path=Path(arguments["screenshot_path"])
                    )

                    return [TextContent(
                        type="text",
                        text=f"✓ Device registered: {arguments['device_name']} ({arguments['device_id']})"
                    )]

                elif name == "get_stats":
                    stats = self.intel.rag.get_stats()
                    stats["devices_registered"] = len(self.intel.devices)
                    stats["incidents"] = len(self.intel.incidents)

                    return [TextContent(
                        type="text",
                        text=json.dumps(stats, indent=2)
                    )]

                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Error: {str(e)}"
                )]

    async def run(self):
        """Run the MCP server"""
        logger.info("🚀 Starting Screenshot Intelligence MCP Server")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP SDK not installed. Install: pip install mcp")
        return 1

    server = ScreenshotIntelMCPServer()
    await server.run()
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
