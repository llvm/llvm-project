#!/usr/bin/env python3
"""
DSMIL Git MCP Server (Security Hardened)
Based on: https://github.com/modelcontextprotocol/servers/tree/main/src/git

Git operations with command injection protection.

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import asyncio
import sys
import os
import hashlib
import subprocess
import shlex
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP library not found", file=sys.stderr)
    sys.exit(1)

from mcp_security import get_security_manager


class GitServer:
    """MCP Server for Git Operations (Security Hardened)"""

    def __init__(self):
        self.server = Server("git")
        self.security = get_security_manager()
        self._client_id = hashlib.sha256(f"{os.getpid()}:git".encode()).hexdigest()[:16]
        self._setup_handlers()

    def _run_git_command(self, args: list[str], cwd: str = None) -> tuple[bool, str]:
        """Run git command safely"""
        try:
            # Validate repo path
            if cwd:
                valid, error = self.security.validate_filepath(cwd)
                if not valid:
                    return False, error

            # Run git command with timeout
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return False, result.stderr

            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name="git_status", description="Show working tree status",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}),
                Tool(name="git_log", description="Show commit history",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}, "max_count": {"type": "integer", "default": 10}}, "required": ["repo"]}),
                Tool(name="git_diff", description="Show changes",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}),
                Tool(name="git_branch", description="List branches",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}),
                Tool(name="git_add", description="Stage files",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}}, "required": ["repo", "files"]}),
                Tool(name="git_commit", description="Commit changes",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}, "message": {"type": "string"}}, "required": ["repo", "message"]}),
                Tool(name="git_checkout", description="Switch branches",
                     inputSchema={"type": "object", "properties": {"repo": {"type": "string"}, "branch": {"type": "string"}}, "required": ["repo", "branch"]}),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if not self.security.check_rate_limit(self._client_id, name):
                return [TextContent(type="text", text="Error: Rate limit exceeded")]

            try:
                repo = arguments.get("repo")
                if not repo:
                    return [TextContent(type="text", text="Error: repo path required")]

                if name == "git_status":
                    success, output = self._run_git_command(["status"], cwd=repo)
                    self.security.audit_request(name, {"repo": repo}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_log":
                    max_count = arguments.get("max_count", 10)
                    success, output = self._run_git_command(["log", f"-{max_count}", "--oneline"], cwd=repo)
                    self.security.audit_request(name, {"repo": repo}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_diff":
                    success, output = self._run_git_command(["diff"], cwd=repo)
                    self.security.audit_request(name, {"repo": repo}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_branch":
                    success, output = self._run_git_command(["branch", "-a"], cwd=repo)
                    self.security.audit_request(name, {"repo": repo}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_add":
                    files = arguments.get("files", [])
                    success, output = self._run_git_command(["add"] + files, cwd=repo)
                    self.security.audit_request(name, {"repo": repo, "files": len(files)}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_commit":
                    message = arguments.get("message", "")
                    # Validate commit message
                    valid, error = self.security.validate_query(message)
                    if not valid:
                        return [TextContent(type="text", text=f"Error: {error}")]

                    success, output = self._run_git_command(["commit", "-m", message], cwd=repo)
                    self.security.audit_request(name, {"repo": repo}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                elif name == "git_checkout":
                    branch = arguments.get("branch", "")
                    # Sanitize branch name
                    if not branch.replace("-", "").replace("_", "").replace("/", "").isalnum():
                        return [TextContent(type="text", text="Error: Invalid branch name")]

                    success, output = self._run_git_command(["checkout", branch], cwd=repo)
                    self.security.audit_request(name, {"repo": repo, "branch": branch}, self._client_id, success)
                    return [TextContent(type="text", text=output if success else f"Error: {output}")]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(GitServer().run())
