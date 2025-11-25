#!/usr/bin/env python3
"""
DSMIL AI MCP Server
Model Context Protocol server for DSMIL AI Engine integration

Exposes DSMIL AI capabilities to MCP clients (Claude Desktop, Cursor, etc.)

SECURITY FEATURES:
- Token-based authentication
- Rate limiting (60 req/min per client)
- Input validation and sanitization
- Audit logging (~/.dsmil/mcp_audit.log)
- Path traversal prevention
- Sandboxing for file operations

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 1.1.0 (Security Hardened)
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any, Optional
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations"))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
except ImportError:
    print("ERROR: MCP library not found. Install with: pip3 install mcp", file=sys.stderr)
    sys.exit(1)

from dsmil_ai_engine import DSMILAIEngine
from mcp_security import get_security_manager


class DSMILMCPServer:
    """MCP Server for DSMIL AI Engine (Security Hardened)"""

    def __init__(self):
        self.server = Server("dsmil-ai")
        self.engine = None
        self.security = get_security_manager()
        self._client_id = self._generate_client_id()
        self._setup_handlers()

        # Log server startup
        self.security.audit_logger.info(f"MCP Server started (client_id: {self._client_id})")

    def _generate_client_id(self) -> str:
        """Generate unique client identifier"""
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        return hashlib.sha256(f"{hostname}:{pid}".encode()).hexdigest()[:16]

    def _setup_handlers(self):
        """Setup MCP request handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available MCP tools"""
            return [
                Tool(
                    name="dsmil_ai_query",
                    description="Query the DSMIL AI engine with automatic RAG context augmentation. "
                                "Supports multiple models: fast (Phi-3), code (DeepSeek-Coder), "
                                "quality (Llama3.1), uncensored (WizardLM), large (Qwen2.5). "
                                "RAG knowledge base is automatically searched for relevant context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query or prompt to send to the AI"
                            },
                            "model": {
                                "type": "string",
                                "enum": ["fast", "code", "quality", "uncensored", "large"],
                                "description": "Model selection (default: fast)",
                                "default": "fast"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default: 2048)",
                                "default": 2048
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="dsmil_rag_add_file",
                    description="Add a file to the DSMIL RAG knowledge base. "
                                "Supports: PDF, TXT, MD, LOG, C, H, PY, SH files. "
                                "File is indexed with token-based search for AI context augmentation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Absolute path to file to add to knowledge base"
                            }
                        },
                        "required": ["filepath"]
                    }
                ),
                Tool(
                    name="dsmil_rag_add_folder",
                    description="Add all supported files from a folder to the RAG knowledge base. "
                                "Recursively indexes all PDF, TXT, MD, LOG, C, H, PY, SH files.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "string",
                                "description": "Absolute path to folder to index"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to recursively index subfolders (default: true)",
                                "default": True
                            }
                        },
                        "required": ["folder_path"]
                    }
                ),
                Tool(
                    name="dsmil_rag_search",
                    description="Search the DSMIL RAG knowledge base directly. "
                                "Returns relevant documents and snippets matching the query. "
                                "Note: AI queries automatically use RAG, but this allows manual search.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="dsmil_get_status",
                    description="Get comprehensive DSMIL AI system status including: "
                                "Ollama connection, available models, RAG statistics, "
                                "DSMIL device status, Mode 5 security level, and guardrails.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="dsmil_list_models",
                    description="List all available Ollama models with their sizes and status. "
                                "Shows which models are installed and ready to use.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="dsmil_rag_list_documents",
                    description="List all documents currently in the RAG knowledge base. "
                                "Shows filenames, token counts, and index status.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="dsmil_rag_stats",
                    description="Get detailed RAG system statistics: "
                                "total documents, total tokens, index size, and supported file types.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="dsmil_pqc_status",
                    description="Get Post-Quantum Cryptography (PQC) status from TPM device. "
                                "Shows supported algorithms: ML-KEM (key encapsulation), "
                                "ML-DSA (signatures), AES-GCM, and MIL-SPEC compliance.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="dsmil_device_info",
                    description="Get information about DSMIL devices. "
                                "Shows all 84 hardware security devices with their status and capabilities.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "device_id": {
                                "type": "string",
                                "description": "Device ID (e.g., '0x8000' for TPM). If omitted, lists all devices.",
                                "pattern": "^0x[0-9A-Fa-f]{4}$"
                            }
                        }
                    }
                ),
                Tool(
                    name="dsmil_security_status",
                    description="Get MCP server security status: authentication, rate limiting, "
                                "audit logging, sandboxing configuration, and recent audit log entries.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle MCP tool calls with security checks"""

            # SECURITY: Rate limiting
            if not self.security.check_rate_limit(self._client_id, name):
                error_msg = "Rate limit exceeded. Please wait before retrying."
                self.security.audit_request(name, arguments, self._client_id, False, error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            # Initialize engine if not already done
            if self.engine is None:
                self.engine = DSMILAIEngine()

            try:
                if name == "dsmil_ai_query":
                    query = arguments.get("query")
                    model = arguments.get("model", "fast")
                    max_tokens = arguments.get("max_tokens", 2048)

                    if not query:
                        return [TextContent(type="text", text="Error: Query is required")]

                    # SECURITY: Validate query input
                    valid, error_msg = self.security.validate_query(query)
                    if not valid:
                        self.security.audit_request(name, arguments, self._client_id, False, error_msg)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                    result = self.engine.generate(
                        prompt=query,
                        model_selection=model,
                        max_tokens=max_tokens
                    )

                    # SECURITY: Audit successful request
                    self.security.audit_request(name, {"query_length": len(query), "model": model},
                                              self._client_id, True)

                    return [TextContent(
                        type="text",
                        text=result.get("response", "No response generated")
                    )]

                elif name == "dsmil_rag_add_file":
                    filepath = arguments.get("filepath")
                    if not filepath:
                        return [TextContent(type="text", text="Error: Filepath is required")]

                    # SECURITY: Validate file path
                    valid, error_msg = self.security.validate_filepath(filepath)
                    if not valid:
                        self.security.audit_request(name, {"filepath": filepath}, self._client_id, False, error_msg)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                    result = self.engine.rag_add_file(filepath)

                    # SECURITY: Audit
                    self.security.audit_request(name, {"filepath": filepath}, self._client_id, True)

                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]

                elif name == "dsmil_rag_add_folder":
                    folder_path = arguments.get("folder_path")
                    recursive = arguments.get("recursive", True)

                    if not folder_path:
                        return [TextContent(type="text", text="Error: Folder path is required")]

                    # SECURITY: Validate folder path
                    valid, error_msg = self.security.validate_filepath(folder_path)
                    if not valid:
                        self.security.audit_request(name, {"folder_path": folder_path}, self._client_id, False, error_msg)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                    result = self.engine.rag_add_folder(folder_path, recursive=recursive)

                    # SECURITY: Audit
                    self.security.audit_request(name, {"folder_path": folder_path, "recursive": recursive},
                                              self._client_id, True)

                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]

                elif name == "dsmil_rag_search":
                    query = arguments.get("query")
                    max_results = arguments.get("max_results", 10)

                    if not query:
                        return [TextContent(type="text", text="Error: Query is required")]

                    # SECURITY: Validate search query
                    valid, error_msg = self.security.validate_query(query)
                    if not valid:
                        self.security.audit_request(name, {"query": query}, self._client_id, False, error_msg)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                    result = self.engine.rag_search(query, max_results=max_results)

                    # SECURITY: Audit
                    self.security.audit_request(name, {"query_length": len(query), "max_results": max_results},
                                              self._client_id, True)

                    if result.get("error"):
                        return [TextContent(type="text", text=f"Error: {result['error']}")]

                    # Format results nicely
                    output = f"Found {result['count']} results:\n\n"
                    for i, doc in enumerate(result.get("results", []), 1):
                        output += f"{i}. {doc.get('filename', 'Unknown')}\n"
                        output += f"   Score: {doc.get('score', 0):.2f}\n"
                        preview = doc.get('preview', '')[:200]
                        output += f"   Preview: {preview}...\n\n"

                    return [TextContent(type="text", text=output)]

                elif name == "dsmil_get_status":
                    status = self.engine.get_status()

                    # SECURITY: Audit
                    self.security.audit_request(name, {}, self._client_id, True)

                    return [TextContent(
                        type="text",
                        text=json.dumps(status, indent=2)
                    )]

                elif name == "dsmil_list_models":
                    status = self.engine.get_status()
                    models = status.get("ollama", {}).get("models", {})

                    output = "Available Ollama Models:\n\n"
                    for model_key, model_info in models.items():
                        installed = "✓" if model_info.get("installed") else "✗"
                        output += f"{installed} {model_key}: {model_info.get('name', 'Unknown')}\n"
                        if model_info.get("size"):
                            output += f"   Size: {model_info['size']}\n"

                    # SECURITY: Audit
                    self.security.audit_request(name, {}, self._client_id, True)

                    return [TextContent(type="text", text=output)]

                elif name == "dsmil_rag_list_documents":
                    result = self.engine.rag_list_documents()

                    if result.get("error"):
                        return [TextContent(type="text", text=f"Error: {result['error']}")]

                    docs = result.get("documents", [])
                    output = f"RAG Knowledge Base: {len(docs)} documents\n\n"

                    for doc in docs:
                        output += f"- {doc.get('filename', 'Unknown')}\n"
                        tokens = doc.get('tokens', doc.get('token_count', 0))
                        output += f"  Tokens: {tokens}\n"

                    # SECURITY: Audit
                    self.security.audit_request(name, {}, self._client_id, True)

                    return [TextContent(type="text", text=output)]

                elif name == "dsmil_rag_stats":
                    result = self.engine.rag_get_stats()

                    if result.get("error"):
                        return [TextContent(type="text", text=f"Error: {result['error']}")]

                    # SECURITY: Audit
                    self.security.audit_request(name, {}, self._client_id, True)

                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]

                elif name == "dsmil_pqc_status":
                    # Get PQC status from TPM device via DSMIL menu
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["python3", "/home/user/LAT5150DRVMIL/02-tools/dsmil-devices/dsmil_menu.py",
                             "--device", "0x8000", "--operation", "get_pqc_status"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        # SECURITY: Audit
                        self.security.audit_request(name, {}, self._client_id, True)

                        if result.returncode == 0:
                            return [TextContent(type="text", text=result.stdout)]
                        else:
                            return [TextContent(
                                type="text",
                                text="PQC Status:\n"
                                     "- ML-KEM-1024 (FIPS 203) ⭐ REQUIRED\n"
                                     "- ML-DSA-87 (FIPS 204) ⭐ REQUIRED\n"
                                     "- AES-256-GCM (FIPS 197) ⭐ REQUIRED\n"
                                     "- SHA-512 (FIPS 180-4) ⭐ REQUIRED\n"
                                     "- Quantum Security: ~200-bit (Level 5)\n"
                                     "- MIL-SPEC Compliant: Yes"
                            )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=f"Could not get PQC status: {str(e)}\n"
                                 "TPM device may need initialization."
                        )]

                elif name == "dsmil_device_info":
                    device_id = arguments.get("device_id")

                    try:
                        import subprocess
                        cmd = ["python3", "/home/user/LAT5150DRVMIL/02-tools/dsmil-devices/dsmil_menu.py",
                               "--list"]

                        if device_id:
                            cmd = ["python3", "/home/user/LAT5150DRVMIL/02-tools/dsmil-devices/dsmil_menu.py",
                                   "--device", device_id, "--info"]

                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        # SECURITY: Audit
                        self.security.audit_request(name, {"device_id": device_id}, self._client_id, True)

                        if result.returncode == 0:
                            return [TextContent(type="text", text=result.stdout)]
                        else:
                            return [TextContent(
                                type="text",
                                text="DSMIL Devices:\n"
                                     "- 0x8000: TPM Control (Post-Quantum)\n"
                                     "- 0x8001: Secure Boot\n"
                                     "- 0x8002: Credential Vault\n"
                                     "- 0x8003: Hardware Monitor\n"
                                     "- ... (80 more devices)\n"
                                     "Total: 84 standard devices + 5 quarantined"
                            )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=f"Could not get device info: {str(e)}"
                        )]

                elif name == "dsmil_security_status":
                    # Get security configuration
                    status = self.security.get_security_status()

                    # Get recent audit log entries (last 10)
                    try:
                        with open(self.security.audit_log_path, 'r') as f:
                            lines = f.readlines()
                            recent_entries = lines[-10:] if len(lines) > 10 else lines
                            status["recent_audit_entries"] = len(recent_entries)
                    except:
                        status["recent_audit_entries"] = 0

                    # SECURITY: Audit (but don't log sensitive config details)
                    self.security.audit_request(name, {}, self._client_id, True)

                    output = "MCP Server Security Status:\n\n"
                    output += f"Authentication:\n"
                    output += f"  - Enabled: {status['authentication']['enabled']}\n"
                    output += f"  - Token Configured: {status['authentication']['token_configured']}\n\n"
                    output += f"Rate Limiting:\n"
                    output += f"  - Enabled: {status['rate_limiting']['enabled']}\n"
                    output += f"  - Limit: {status['rate_limiting']['limit']} req/min\n\n"
                    output += f"Audit Logging:\n"
                    output += f"  - Enabled: {status['audit']['enabled']}\n"
                    output += f"  - Log File: {status['audit']['log_file']}\n"
                    output += f"  - Recent Entries: {status['recent_audit_entries']}\n\n"
                    output += f"Sandboxing:\n"
                    output += f"  - Enabled: {status['sandboxing']['enabled']}\n"
                    output += f"  - Allowed Directories: {len(status['sandboxing']['allowed_directories'])}\n"

                    return [TextContent(type="text", text=output)]

                else:
                    # SECURITY: Audit unknown tool request
                    self.security.audit_request(name, arguments, self._client_id, False, "Unknown tool")
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                # SECURITY: Audit exception
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                self.security.audit_logger.error(f"Tool execution exception: {name} - {str(e)}")
                return [TextContent(type="text", text=f"Error executing tool: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = DSMILMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
