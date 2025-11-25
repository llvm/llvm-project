#!/usr/bin/env python3
"""
DSMIL Filesystem MCP Server (Security Hardened)
Based on: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

Provides secure filesystem operations with comprehensive sandboxing.

SECURITY FEATURES:
- Path validation and sandboxing
- Rate limiting
- File size limits
- Extension filtering
- Audit logging
- Symlink protection

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 1.0.0 (Security Hardened)
"""

import asyncio
import json
import sys
import os
import base64
from pathlib import Path
from typing import Any, List
import hashlib
import glob

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP library not found", file=sys.stderr)
    sys.exit(1)

from mcp_security import get_security_manager


class FilesystemServer:
    """MCP Server for Filesystem Operations (Security Hardened)"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_FILES_BULK = 50  # Maximum files in bulk operations

    def __init__(self):
        self.server = Server("filesystem")
        self.security = get_security_manager()
        self._client_id = hashlib.sha256(f"{os.getpid()}:filesystem".encode()).hexdigest()[:16]
        self._setup_handlers()
        self.security.audit_logger.info(f"Filesystem MCP Server started (client_id: {self._client_id})")

    def _setup_handlers(self):
        """Setup MCP handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name="read_file", description="Read text file contents (sandboxed)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                Tool(name="write_file", description="Write file (sandboxed, size limited)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
                Tool(name="list_directory", description="List directory contents (sandboxed)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                Tool(name="create_directory", description="Create directory (sandboxed)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                Tool(name="delete_file", description="Delete file (sandboxed)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                Tool(name="move_file", description="Move/rename file (sandboxed)",
                     inputSchema={"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}),
                Tool(name="get_file_info", description="Get file metadata (sandboxed)",
                     inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
                Tool(name="search_files", description="Search files by glob pattern (sandboxed)",
                     inputSchema={"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if not self.security.check_rate_limit(self._client_id, name):
                return [TextContent(type="text", text="Error: Rate limit exceeded")]

            try:
                if name == "read_file":
                    path = arguments.get("path")
                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        self.security.audit_request(name, {"path": path}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    file_path = Path(path).resolve()
                    if not file_path.exists():
                        return [TextContent(type="text", text=f"Error: File not found: {path}")]

                    if file_path.stat().st_size > self.MAX_FILE_SIZE:
                        return [TextContent(type="text", text=f"Error: File too large (max {self.MAX_FILE_SIZE} bytes)")]

                    content = file_path.read_text()
                    self.security.audit_request(name, {"path": path, "size": len(content)}, self._client_id, True)
                    return [TextContent(type="text", text=content)]

                elif name == "write_file":
                    path = arguments.get("path")
                    content = arguments.get("content", "")

                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        self.security.audit_request(name, {"path": path}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    if len(content) > self.MAX_FILE_SIZE:
                        return [TextContent(type="text", text=f"Error: Content too large")]

                    file_path = Path(path).resolve()
                    file_path.write_text(content)
                    self.security.audit_request(name, {"path": path, "size": len(content)}, self._client_id, True)
                    return [TextContent(type="text", text=f"File written: {path} ({len(content)} bytes)")]

                elif name == "list_directory":
                    path = arguments.get("path", ".")
                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        self.security.audit_request(name, {"path": path}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    dir_path = Path(path).resolve()
                    if not dir_path.is_dir():
                        return [TextContent(type="text", text=f"Error: Not a directory: {path}")]

                    items = []
                    for item in sorted(dir_path.iterdir()):
                        prefix = "📁" if item.is_dir() else "📄"
                        items.append(f"{prefix} {item.name}")

                    self.security.audit_request(name, {"path": path, "count": len(items)}, self._client_id, True)
                    return [TextContent(type="text", text="\n".join(items) if items else "(empty)")]

                elif name == "create_directory":
                    path = arguments.get("path")
                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        self.security.audit_request(name, {"path": path}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    dir_path = Path(path).resolve()
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.security.audit_request(name, {"path": path}, self._client_id, True)
                    return [TextContent(type="text", text=f"Directory created: {path}")]

                elif name == "delete_file":
                    path = arguments.get("path")
                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        self.security.audit_request(name, {"path": path}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    file_path = Path(path).resolve()
                    if not file_path.exists():
                        return [TextContent(type="text", text=f"Error: File not found: {path}")]

                    if file_path.is_dir():
                        file_path.rmdir()
                    else:
                        file_path.unlink()

                    self.security.audit_request(name, {"path": path}, self._client_id, True)
                    return [TextContent(type="text", text=f"Deleted: {path}")]

                elif name == "move_file":
                    source = arguments.get("source")
                    dest = arguments.get("destination")

                    valid_src, error_src = self.security.validate_filepath(source)
                    valid_dst, error_dst = self.security.validate_filepath(dest)

                    if not valid_src:
                        return [TextContent(type="text", text=f"Error: {error_src}")]
                    if not valid_dst:
                        return [TextContent(type="text", text=f"Error: {error_dst}")]

                    src_path = Path(source).resolve()
                    dst_path = Path(dest).resolve()

                    src_path.rename(dst_path)
                    self.security.audit_request(name, {"source": source, "dest": dest}, self._client_id, True)
                    return [TextContent(type="text", text=f"Moved: {source} → {dest}")]

                elif name == "get_file_info":
                    path = arguments.get("path")
                    valid, error = self.security.validate_filepath(path)
                    if not valid:
                        return [TextContent(type="text", text=f"Error: {error}")]

                    file_path = Path(path).resolve()
                    if not file_path.exists():
                        return [TextContent(type="text", text=f"Error: File not found")]

                    stat = file_path.stat()
                    info = {
                        "path": str(file_path),
                        "size": stat.st_size,
                        "type": "directory" if file_path.is_dir() else "file",
                        "modified": stat.st_mtime,
                        "permissions": oct(stat.st_mode)
                    }

                    self.security.audit_request(name, {"path": path}, self._client_id, True)
                    return [TextContent(type="text", text=json.dumps(info, indent=2))]

                elif name == "search_files":
                    pattern = arguments.get("pattern")
                    search_path = arguments.get("path", ".")

                    valid, error = self.security.validate_filepath(search_path)
                    if not valid:
                        return [TextContent(type="text", text=f"Error: {error}")]

                    base_path = Path(search_path).resolve()
                    matches = glob.glob(str(base_path / pattern), recursive=True)
                    matches = [str(Path(m).relative_to(base_path)) for m in matches[:100]]  # Limit to 100

                    self.security.audit_request(name, {"pattern": pattern, "matches": len(matches)}, self._client_id, True)
                    return [TextContent(type="text", text="\n".join(matches) if matches else "(no matches)")]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(FilesystemServer().run())
