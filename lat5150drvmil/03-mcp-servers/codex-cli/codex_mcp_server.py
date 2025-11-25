#!/usr/bin/env python3
"""
Codex CLI MCP Server - Python wrapper for Rust codex-cli client

This MCP server provides a Python interface to the Rust-based codex CLI,
integrating it with the LAT5150DRVMIL AI platform.

Features:
- Code generation, review, debugging, refactoring, documentation
- Streaming responses for real-time feedback
- Authentication management (ChatGPT + API key)
- Configuration management
- Extended reasoning capabilities

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
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("codex-mcp-server")

# Paths
CODEX_CLI_PATH = Path(__file__).parent / "target" / "release" / "codex-cli"
if not CODEX_CLI_PATH.exists():
    # Try debug build
    CODEX_CLI_PATH = Path(__file__).parent / "target" / "debug" / "codex-cli"


class CodexMCPServer:
    """MCP Server for Codex CLI integration"""

    def __init__(self):
        self.server = Server("codex-cli")
        self.codex_cli_path = CODEX_CLI_PATH
        self.config_path = Path.home() / ".codex" / "config.toml"

        # Register handlers
        self._register_handlers()

        logger.info(f"Codex MCP Server initialized")
        logger.info(f"Codex CLI path: {self.codex_cli_path}")
        logger.info(f"Config path: {self.config_path}")

    def _register_handlers(self):
        """Register MCP protocol handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="codex_generate",
                    description="Generate code based on natural language description using OpenAI Codex",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Natural language description of code to generate"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language (python, rust, javascript, etc.)",
                                "default": "python"
                            },
                            "style": {
                                "type": "string",
                                "description": "Code style (clean, verbose, minimal, production)",
                                "default": "clean"
                            }
                        },
                        "required": ["description"]
                    }
                ),
                Tool(
                    name="codex_review",
                    description="Review code for issues, improvements, and best practices",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to review"
                            },
                            "focus": {
                                "type": "string",
                                "description": "Review focus: security, performance, style, readability, or all",
                                "default": "all"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language for context"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_debug",
                    description="Debug code issues and provide fixes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code with issues"
                            },
                            "error": {
                                "type": "string",
                                "description": "Error message or description of the problem"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about the environment or setup"
                            }
                        },
                        "required": ["code", "error"]
                    }
                ),
                Tool(
                    name="codex_refactor",
                    description="Refactor code for better quality, readability, or performance",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to refactor"
                            },
                            "goal": {
                                "type": "string",
                                "description": "Refactoring goal: readability, performance, maintainability, etc.",
                                "default": "general improvement"
                            },
                            "constraints": {
                                "type": "string",
                                "description": "Constraints to respect (API compatibility, dependencies, etc.)"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_document",
                    description="Generate comprehensive documentation for code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to document"
                            },
                            "format": {
                                "type": "string",
                                "description": "Documentation format: markdown, rst, docstring, inline",
                                "default": "markdown"
                            },
                            "detail_level": {
                                "type": "string",
                                "description": "Detail level: brief, standard, comprehensive",
                                "default": "standard"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_explain",
                    description="Explain how code works in natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to explain"
                            },
                            "audience": {
                                "type": "string",
                                "description": "Target audience: beginner, intermediate, expert",
                                "default": "intermediate"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_optimize",
                    description="Optimize code for performance, memory, or efficiency",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to optimize"
                            },
                            "target": {
                                "type": "string",
                                "description": "Optimization target: speed, memory, size, power",
                                "default": "speed"
                            },
                            "profile_data": {
                                "type": "string",
                                "description": "Profiling data or bottleneck information"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_test",
                    description="Generate unit tests for code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to test"
                            },
                            "framework": {
                                "type": "string",
                                "description": "Test framework: pytest, unittest, jest, etc."
                            },
                            "coverage": {
                                "type": "string",
                                "description": "Coverage goal: basic, standard, comprehensive",
                                "default": "standard"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="codex_convert",
                    description="Convert code between programming languages",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to convert"
                            },
                            "from_language": {
                                "type": "string",
                                "description": "Source language"
                            },
                            "to_language": {
                                "type": "string",
                                "description": "Target language"
                            },
                            "preserve_style": {
                                "type": "boolean",
                                "description": "Try to preserve coding style",
                                "default": True
                            }
                        },
                        "required": ["code", "from_language", "to_language"]
                    }
                ),
                Tool(
                    name="codex_config",
                    description="Configure Codex CLI settings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action: get, set, init, auth_status",
                                "enum": ["get", "set", "init", "auth_status"]
                            },
                            "key": {
                                "type": "string",
                                "description": "Configuration key (for set action)"
                            },
                            "value": {
                                "type": "string",
                                "description": "Configuration value (for set action)"
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
                if name == "codex_generate":
                    result = await self._code_generation(arguments)
                elif name == "codex_review":
                    result = await self._code_review(arguments)
                elif name == "codex_debug":
                    result = await self._code_debug(arguments)
                elif name == "codex_refactor":
                    result = await self._code_refactor(arguments)
                elif name == "codex_document":
                    result = await self._code_document(arguments)
                elif name == "codex_explain":
                    result = await self._code_explain(arguments)
                elif name == "codex_optimize":
                    result = await self._code_optimize(arguments)
                elif name == "codex_test":
                    result = await self._code_test(arguments)
                elif name == "codex_convert":
                    result = await self._code_convert(arguments)
                elif name == "codex_config":
                    result = await self._config_management(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=result)]

            except Exception as e:
                logger.error(f"Tool execution error: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    async def _execute_codex(self, prompt: str, format: str = "text") -> str:
        """Execute codex CLI with given prompt"""

        if not self.codex_cli_path.exists():
            return "Error: Codex CLI binary not found. Please build it first with: cargo build --release"

        try:
            cmd = [
                str(self.codex_cli_path),
                "exec",
                prompt,
                "--format", format
            ]

            logger.debug(f"Executing: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"Codex CLI error: {error_msg}")
                return f"Error: {error_msg}"

            return stdout.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return f"Execution error: {str(e)}"

    async def _code_generation(self, args: Dict[str, Any]) -> str:
        """Generate code based on description"""
        description = args["description"]
        language = args.get("language", "python")
        style = args.get("style", "clean")

        prompt = f"""Generate {style} {language} code for the following:

{description}

Requirements:
- Production-ready code
- Proper error handling
- Clear comments
- Follow best practices for {language}

Provide only the code with minimal explanation."""

        return await self._execute_codex(prompt)

    async def _code_review(self, args: Dict[str, Any]) -> str:
        """Review code for issues"""
        code = args["code"]
        focus = args.get("focus", "all")
        language = args.get("language", "auto-detect")

        prompt = f"""Review this {language} code with focus on {focus}:

```
{code}
```

Provide:
1. Issues found (bugs, security, performance, style)
2. Severity of each issue (critical, high, medium, low)
3. Specific recommendations for fixes
4. Overall code quality score (1-10)

Be specific and actionable."""

        return await self._execute_codex(prompt)

    async def _code_debug(self, args: Dict[str, Any]) -> str:
        """Debug code issues"""
        code = args["code"]
        error = args["error"]
        context = args.get("context", "")

        prompt = f"""Debug this code issue:

Code:
```
{code}
```

Error: {error}

{f"Context: {context}" if context else ""}

Provide:
1. Root cause analysis
2. Step-by-step explanation of why the error occurs
3. Fixed code
4. Prevention tips to avoid similar issues

Be thorough and educational."""

        return await self._execute_codex(prompt)

    async def _code_refactor(self, args: Dict[str, Any]) -> str:
        """Refactor code"""
        code = args["code"]
        goal = args.get("goal", "general improvement")
        constraints = args.get("constraints", "")

        prompt = f"""Refactor this code for {goal}:

```
{code}
```

{f"Constraints: {constraints}" if constraints else ""}

Provide:
1. Refactored code
2. Explanation of changes made
3. Benefits of the refactoring
4. Any trade-offs introduced

Maintain functionality while improving quality."""

        return await self._execute_codex(prompt)

    async def _code_document(self, args: Dict[str, Any]) -> str:
        """Generate documentation"""
        code = args["code"]
        format_type = args.get("format", "markdown")
        detail_level = args.get("detail_level", "standard")

        prompt = f"""Generate {detail_level} {format_type} documentation for:

```
{code}
```

Include:
- Overview and purpose
- Parameters and return values
- Usage examples
- Edge cases and limitations
- Related functions/classes

Make it clear and comprehensive."""

        return await self._execute_codex(prompt)

    async def _code_explain(self, args: Dict[str, Any]) -> str:
        """Explain code"""
        code = args["code"]
        audience = args.get("audience", "intermediate")

        prompt = f"""Explain this code for a {audience} audience:

```
{code}
```

Provide:
1. High-level overview
2. Step-by-step breakdown
3. Key concepts used
4. Real-world analogy (if complex)

Make it educational and clear."""

        return await self._execute_codex(prompt)

    async def _code_optimize(self, args: Dict[str, Any]) -> str:
        """Optimize code"""
        code = args["code"]
        target = args.get("target", "speed")
        profile_data = args.get("profile_data", "")

        prompt = f"""Optimize this code for {target}:

```
{code}
```

{f"Profiling data: {profile_data}" if profile_data else ""}

Provide:
1. Optimized code
2. Performance improvements (quantify if possible)
3. Explanation of optimization techniques used
4. Potential trade-offs

Focus on {target} optimization."""

        return await self._execute_codex(prompt)

    async def _code_test(self, args: Dict[str, Any]) -> str:
        """Generate tests"""
        code = args["code"]
        framework = args.get("framework", "pytest")
        coverage = args.get("coverage", "standard")

        prompt = f"""Generate {coverage} {framework} tests for:

```
{code}
```

Include:
- Happy path tests
- Edge cases
- Error cases
- Integration tests (if applicable)
- Mocks/fixtures as needed

Aim for high code coverage with meaningful tests."""

        return await self._execute_codex(prompt)

    async def _code_convert(self, args: Dict[str, Any]) -> str:
        """Convert code between languages"""
        code = args["code"]
        from_lang = args["from_language"]
        to_lang = args["to_language"]
        preserve_style = args.get("preserve_style", True)

        style_note = "preserving coding style" if preserve_style else "using idiomatic style"

        prompt = f"""Convert this {from_lang} code to {to_lang} ({style_note}):

```{from_lang}
{code}
```

Provide:
1. Converted {to_lang} code
2. Notes on translation decisions
3. Differences in behavior or capabilities
4. Required dependencies in {to_lang}

Make it idiomatic and correct."""

        return await self._execute_codex(prompt)

    async def _config_management(self, args: Dict[str, Any]) -> str:
        """Manage configuration"""
        action = args["action"]

        if action == "auth_status":
            if not self.codex_cli_path.exists():
                return "Codex CLI not built"

            try:
                cmd = [str(self.codex_cli_path), "auth", "status"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return stdout.decode('utf-8', errors='ignore')
            except Exception as e:
                return f"Error checking auth status: {e}"

        elif action == "init":
            try:
                cmd = [str(self.codex_cli_path), "config", "init"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return stdout.decode('utf-8', errors='ignore')
            except Exception as e:
                return f"Error initializing config: {e}"

        elif action == "get":
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return f.read()
            return "Configuration file not found"

        elif action == "set":
            key = args.get("key")
            value = args.get("value")

            if not key or not value:
                return "Error: key and value required for set action"

            try:
                cmd = [str(self.codex_cli_path), "config", "set", key, value]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return stdout.decode('utf-8', errors='ignore')
            except Exception as e:
                return f"Error setting config: {e}"

        return f"Unknown action: {action}"

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Codex MCP server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="codex-cli",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    server = CodexMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
