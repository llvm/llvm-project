#!/usr/bin/env python3
"""
NotebookLM MCP Server
---------------------
MCP server providing Google NotebookLM-style functionality:
- Document source management
- Multi-source Q&A with grounding
- Summary, FAQ, and study guide generation
- Source synthesis and analysis

Powered by Google Gemini API with 2M token context.

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sub_agents.notebooklm_wrapper import NotebookLMAgent

# MCP Protocol
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class NotebookLMMCPServer:
    """MCP Server for NotebookLM functionality"""

    def __init__(self):
        self.agent = NotebookLMAgent()
        self.server = Server("notebooklm")

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools"""

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available NotebookLM tools"""
            return [
                types.Tool(
                    name="notebooklm_add_source",
                    description="Add a document source to NotebookLM workspace (text content or file path)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Text content to add as source"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to file (PDF, markdown, text)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Source title"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_query",
                    description="Ask questions about your sources with natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Your question or request"
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific source IDs to query (optional, defaults to all)"
                            },
                            "notebook_id": {
                                "type": "string",
                                "description": "Notebook ID to query within (optional)"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["qa", "summarize", "faq", "study_guide", "synthesis", "briefing"],
                                "description": "Query mode (default: qa)"
                            }
                        },
                        "required": ["prompt"]
                    }
                ),
                types.Tool(
                    name="notebooklm_summarize",
                    description="Generate a comprehensive summary of sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to summarize (optional)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_create_faq",
                    description="Create FAQ document from sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to create FAQ from (optional)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_create_study_guide",
                    description="Create study guide from sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to create study guide from (optional)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_synthesize",
                    description="Synthesize insights across multiple sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to synthesize (optional)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_create_briefing",
                    description="Create executive briefing from sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to create briefing from (optional)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="notebooklm_create_notebook",
                    description="Create a new notebook with selected sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Notebook title"
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source IDs to include in notebook"
                            }
                        },
                        "required": ["title"]
                    }
                ),
                types.Tool(
                    name="notebooklm_list_sources",
                    description="List all sources in the workspace",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="notebooklm_list_notebooks",
                    description="List all notebooks",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="notebooklm_remove_source",
                    description="Remove a source from the workspace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "Source ID to remove"
                            }
                        },
                        "required": ["source_id"]
                    }
                ),
                types.Tool(
                    name="notebooklm_clear_sources",
                    description="Clear all sources from the workspace",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="notebooklm_status",
                    description="Get NotebookLM agent status and statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution"""

            if not arguments:
                arguments = {}

            try:
                # Route to appropriate handler
                if name == "notebooklm_add_source":
                    result = self.agent.add_source(
                        content=arguments.get("content"),
                        file_path=arguments.get("file_path"),
                        title=arguments.get("title"),
                        metadata=arguments.get("metadata")
                    )

                elif name == "notebooklm_query":
                    result = self.agent.query(
                        prompt=arguments["prompt"],
                        source_ids=arguments.get("source_ids"),
                        notebook_id=arguments.get("notebook_id"),
                        mode=arguments.get("mode", "qa")
                    )

                elif name == "notebooklm_summarize":
                    result = self.agent.summarize_sources(
                        source_ids=arguments.get("source_ids")
                    )

                elif name == "notebooklm_create_faq":
                    result = self.agent.create_faq(
                        source_ids=arguments.get("source_ids")
                    )

                elif name == "notebooklm_create_study_guide":
                    result = self.agent.create_study_guide(
                        source_ids=arguments.get("source_ids")
                    )

                elif name == "notebooklm_synthesize":
                    result = self.agent.synthesize(
                        source_ids=arguments.get("source_ids")
                    )

                elif name == "notebooklm_create_briefing":
                    result = self.agent.create_briefing(
                        source_ids=arguments.get("source_ids")
                    )

                elif name == "notebooklm_create_notebook":
                    result = self.agent.create_notebook(
                        title=arguments["title"],
                        source_ids=arguments.get("source_ids", [])
                    )

                elif name == "notebooklm_list_sources":
                    result = self.agent.list_sources()

                elif name == "notebooklm_list_notebooks":
                    result = self.agent.list_notebooks()

                elif name == "notebooklm_remove_source":
                    result = self.agent.remove_source(
                        source_id=arguments["source_id"]
                    )

                elif name == "notebooklm_clear_sources":
                    result = self.agent.clear_all_sources()

                elif name == "notebooklm_status":
                    result = self.agent.get_status()

                else:
                    result = {
                        "success": False,
                        "error": f"Unknown tool: {name}"
                    }

                # Format response
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )
                ]

            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e),
                    "tool": name
                }
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(error_result, indent=2)
                    )
                ]

    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="notebooklm",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    if not MCP_AVAILABLE:
        print("ERROR: MCP library not available. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = NotebookLMMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
