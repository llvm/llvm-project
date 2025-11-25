#!/usr/bin/env python3
"""
Claude Code MCP Server - Python wrapper for Rust claude-code client

High-performance coding agent with improvements from claude-backups:
- Agent orchestration (25+ specialized agents)
- Git intelligence (ShadowGit Phase 3)
- NPU acceleration (Intel AI Boost)
- Binary IPC (50ns-10µs latency)
- AVX2/AVX-512 SIMD optimizations

Author: SWORD Intelligence
License: Apache-2.0
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.models import InitializationOptions
    from mcp.types import Tool, TextContent
    import mcp.types as types
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude-code-mcp")

# Paths
CLAUDE_CODE_CLI_PATH = Path(__file__).parent / "target" / "release" / "claude-code"
if not CLAUDE_CODE_CLI_PATH.exists():
    # Try debug build
    CLAUDE_CODE_CLI_PATH = Path(__file__).parent / "target" / "debug" / "claude-code"


class ClaudeCodeMCPServer:
    """MCP Server for Claude Code integration"""

    def __init__(self):
        self.server = Server("claude-code")
        self.claude_code_path = CLAUDE_CODE_CLI_PATH

        # Register handlers
        self._register_handlers()

        logger.info("Claude Code MCP Server initialized")
        logger.info(f"CLI path: {self.claude_code_path}")

    def _register_handlers(self):
        """Register MCP protocol handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="claude_code_generate",
                    description="Generate code using Claude Code with hardware acceleration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Code generation task description"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language"
                            },
                            "use_npu": {
                                "type": "boolean",
                                "description": "Enable NPU acceleration",
                                "default": False
                            },
                            "use_avx512": {
                                "type": "boolean",
                                "description": "Enable AVX-512 SIMD",
                                "default": False
                            }
                        },
                        "required": ["task"]
                    }
                ),
                Tool(
                    name="claude_git_analyze",
                    description="Analyze Git repository with ShadowGit (7-10x faster with NPU)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Repository path (defaults to current directory)"
                            },
                            "include_intelligence": {
                                "type": "boolean",
                                "description": "Include repository intelligence analysis",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="claude_git_conflicts",
                    description="Predict merge conflicts with AI (sub-50ms diff processing)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "base_branch": {
                                "type": "string",
                                "description": "Base branch name"
                            },
                            "compare_branch": {
                                "type": "string",
                                "description": "Branch to compare"
                            }
                        },
                        "required": ["base_branch", "compare_branch"]
                    }
                ),
                Tool(
                    name="claude_git_diff",
                    description="Fast diff with SIMD acceleration (AVX2/AVX-512)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "commit_a": {
                                "type": "string",
                                "description": "First commit/branch"
                            },
                            "commit_b": {
                                "type": "string",
                                "description": "Second commit/branch"
                            }
                        },
                        "required": ["commit_a", "commit_b"]
                    }
                ),
                Tool(
                    name="claude_agent_execute",
                    description="Execute task with specific agent from 25+ orchestration system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID (use claude_agent_list to see available)"
                            },
                            "task": {
                                "type": "string",
                                "description": "Task for agent to execute"
                            }
                        },
                        "required": ["agent_id", "task"]
                    }
                ),
                Tool(
                    name="claude_agent_list",
                    description="List available agents in orchestration system",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="claude_session_new",
                    description="Create new coding session with auto-save",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Session name"
                            }
                        }
                    }
                ),
                Tool(
                    name="claude_session_resume",
                    description="Resume previous coding session",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID or name"
                            }
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="claude_benchmark",
                    description="Run performance benchmarks (IPC, SIMD, Git, Agents)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "suite": {
                                "type": "string",
                                "description": "Benchmark suite: all, ipc, simd, git, agent",
                                "default": "all"
                            },
                            "iterations": {
                                "type": "integer",
                                "description": "Number of iterations",
                                "default": 1000
                            }
                        }
                    }
                ),
                Tool(
                    name="claude_config",
                    description="Manage Claude Code configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action: show, set, validate",
                                "enum": ["show", "set", "validate"]
                            },
                            "key": {
                                "type": "string",
                                "description": "Config key (for set action)"
                            },
                            "value": {
                                "type": "string",
                                "description": "Config value (for set action)"
                            }
                        },
                        "required": ["action"]
                    }
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            logger.info(f"Tool called: {name} with args: {arguments}")

            try:
                if name == "claude_code_generate":
                    result = await self._code_generate(arguments)
                elif name == "claude_git_analyze":
                    result = await self._git_analyze(arguments)
                elif name == "claude_git_conflicts":
                    result = await self._git_conflicts(arguments)
                elif name == "claude_git_diff":
                    result = await self._git_diff(arguments)
                elif name == "claude_agent_execute":
                    result = await self._agent_execute(arguments)
                elif name == "claude_agent_list":
                    result = await self._agent_list(arguments)
                elif name == "claude_session_new":
                    result = await self._session_new(arguments)
                elif name == "claude_session_resume":
                    result = await self._session_resume(arguments)
                elif name == "claude_benchmark":
                    result = await self._benchmark(arguments)
                elif name == "claude_config":
                    result = await self._config(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=result)]

            except Exception as e:
                logger.error(f"Tool execution error: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    async def _execute_cli(self, args: List[str]) -> str:
        """Execute Claude Code CLI"""
        if not self.claude_code_path.exists():
            return "Error: Claude Code CLI not built. Run: cd 03-mcp-servers/claude-code && cargo build --release"

        try:
            cmd = [str(self.claude_code_path)] + args
            logger.debug(f"Executing: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"CLI error: {error_msg}")
                return f"Error: {error_msg}"

            return stdout.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return f"Execution error: {str(e)}"

    async def _code_generate(self, args: Dict[str, Any]) -> str:
        """Generate code with hardware acceleration"""
        task = args["task"]
        language = args.get("language", "python")
        use_npu = args.get("use_npu", False)
        use_avx512 = args.get("use_avx512", False)

        cli_args = ["exec", f"Generate {language} code: {task}"]
        if use_npu:
            cli_args.insert(0, "--npu")
        if use_avx512:
            cli_args.insert(0, "--avx512")

        return await self._execute_cli(cli_args)

    async def _git_analyze(self, args: Dict[str, Any]) -> str:
        """Analyze Git repository"""
        repo_path = args.get("repo_path", ".")

        cli_args = ["git", "analyze", "--repo", repo_path]

        if args.get("include_intelligence", True):
            intelligence = await self._execute_cli(["git", "intelligence", "--repo", repo_path])
            analysis = await self._execute_cli(cli_args)
            return f"{analysis}\n\nIntelligence:\n{intelligence}"

        return await self._execute_cli(cli_args)

    async def _git_conflicts(self, args: Dict[str, Any]) -> str:
        """Predict merge conflicts"""
        base = args["base_branch"]
        compare = args["compare_branch"]

        return await self._execute_cli(["git", "conflicts", base, compare])

    async def _git_diff(self, args: Dict[str, Any]) -> str:
        """Fast diff with SIMD"""
        commit_a = args["commit_a"]
        commit_b = args["commit_b"]

        return await self._execute_cli(["git", "diff", commit_a, commit_b])

    async def _agent_execute(self, args: Dict[str, Any]) -> str:
        """Execute with specific agent"""
        agent_id = args["agent_id"]
        task = args["task"]

        return await self._execute_cli(["agent", "execute", agent_id, task])

    async def _agent_list(self, args: Dict[str, Any]) -> str:
        """List available agents"""
        return await self._execute_cli(["agent", "list"])

    async def _session_new(self, args: Dict[str, Any]) -> str:
        """Create new session"""
        name = args.get("name")
        if name:
            return await self._execute_cli(["session", "new", name])
        return await self._execute_cli(["session", "new"])

    async def _session_resume(self, args: Dict[str, Any]) -> str:
        """Resume session"""
        session_id = args["session_id"]
        return await self._execute_cli(["session", "resume", session_id])

    async def _benchmark(self, args: Dict[str, Any]) -> str:
        """Run benchmarks"""
        suite = args.get("suite", "all")
        iterations = args.get("iterations", 1000)

        return await self._execute_cli([
            "bench",
            "--suite", suite,
            "--iterations", str(iterations)
        ])

    async def _config(self, args: Dict[str, Any]) -> str:
        """Manage configuration"""
        action = args["action"]

        if action == "show":
            return await self._execute_cli(["config", "show"])
        elif action == "set":
            key = args.get("key")
            value = args.get("value")
            if not key or not value:
                return "Error: key and value required for set action"
            return await self._execute_cli(["config", "set", key, value])
        elif action == "validate":
            return await self._execute_cli(["config", "validate"])

        return f"Unknown config action: {action}"

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Claude Code MCP server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="claude-code",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=types.NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    server = ClaudeCodeMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
