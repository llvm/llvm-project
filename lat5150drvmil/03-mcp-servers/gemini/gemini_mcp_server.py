#!/usr/bin/env python3
"""
Gemini MCP Server
-----------------
MCP 2024-11-05 protocol server for Google Gemini API.

Features:
- Multimodal support (text, images, videos, audio)
- Function calling
- Code execution
- Google Search grounding
- Thinking mode (extended reasoning)
- Long context (up to 2M tokens)
- Session management with conversation history
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MCP Protocol Version
MCP_VERSION = "2024-11-05"

# Get Gemini CLI path from environment or use default
GEMINI_PATH = os.environ.get(
    "GEMINI_PATH",
    str(Path(__file__).parent / "target" / "release" / "gemini")
)


class GeminiMCPServer:
    """MCP server for Gemini integration."""

    def __init__(self):
        self.tools = self._register_tools()
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register all available MCP tools."""
        return [
            {
                "name": "gemini_generate",
                "description": "Generate text using Google Gemini with optional thinking mode and grounding",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to Gemini"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for conversation continuity"
                        },
                        "thinking_mode": {
                            "type": "boolean",
                            "description": "Enable thinking mode for extended reasoning (uses gemini-2.0-flash-thinking-exp)",
                            "default": False
                        },
                        "grounding": {
                            "type": "boolean",
                            "description": "Enable Google Search grounding for factual accuracy",
                            "default": False
                        },
                        "code_execution": {
                            "type": "boolean",
                            "description": "Enable code execution capabilities",
                            "default": False
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0.0-2.0)",
                            "default": 1.0
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 8192
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "gemini_multimodal",
                "description": "Analyze multimodal content (images, videos, audio) with Gemini",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Question or instruction about the media"
                        },
                        "media_path": {
                            "type": "string",
                            "description": "Path to image, video, or audio file"
                        },
                        "media_type": {
                            "type": "string",
                            "enum": ["image", "video", "audio"],
                            "description": "Type of media file"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for conversation continuity"
                        }
                    },
                    "required": ["prompt", "media_path", "media_type"]
                }
            },
            {
                "name": "gemini_function_call",
                "description": "Execute function calling with Gemini using JSON schema definitions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to Gemini"
                        },
                        "functions": {
                            "type": "array",
                            "description": "List of function declarations (JSON schema format)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "parameters": {"type": "object"}
                                },
                                "required": ["name", "description", "parameters"]
                            }
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for conversation continuity"
                        }
                    },
                    "required": ["prompt", "functions"]
                }
            },
            {
                "name": "gemini_code_execute",
                "description": "Generate and execute code using Gemini's built-in code execution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Description of the code to generate and execute"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for conversation continuity"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "gemini_config",
                "description": "Get or update Gemini configuration settings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["get", "set"],
                            "description": "Get or set configuration"
                        },
                        "config": {
                            "type": "object",
                            "description": "Configuration object (for 'set' action)",
                            "properties": {
                                "model": {"type": "string"},
                                "api_key": {"type": "string"},
                                "temperature": {"type": "number"},
                                "top_p": {"type": "number"},
                                "top_k": {"type": "integer"},
                                "max_output_tokens": {"type": "integer"},
                                "default_thinking": {"type": "boolean"},
                                "default_grounding": {"type": "boolean"},
                                "default_code_execution": {"type": "boolean"}
                            }
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "gemini_session_new",
                "description": "Create a new Gemini conversation session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Unique identifier for the session"
                        },
                        "thinking_mode": {
                            "type": "boolean",
                            "description": "Enable thinking mode for this session",
                            "default": False
                        },
                        "grounding": {
                            "type": "boolean",
                            "description": "Enable grounding for this session",
                            "default": False
                        }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "gemini_session_resume",
                "description": "Resume an existing Gemini conversation session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to resume"
                        }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "gemini_session_stats",
                "description": "Get statistics for a Gemini session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to get stats for"
                        }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "gemini_session_clear",
                "description": "Clear conversation history for a session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to clear"
                        }
                    },
                    "required": ["session_id"]
                }
            }
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return await self._handle_initialize(params)
        elif method == "tools/list":
            return await self._handle_tools_list()
        elif method == "tools/call":
            return await self._handle_tools_call(params)
        else:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "gemini-mcp-server",
                "version": "1.0.0"
            }
        }

    async def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": self.tools}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "gemini_generate":
                result = await self._gemini_generate(arguments)
            elif tool_name == "gemini_multimodal":
                result = await self._gemini_multimodal(arguments)
            elif tool_name == "gemini_function_call":
                result = await self._gemini_function_call(arguments)
            elif tool_name == "gemini_code_execute":
                result = await self._gemini_code_execute(arguments)
            elif tool_name == "gemini_config":
                result = await self._gemini_config(arguments)
            elif tool_name == "gemini_session_new":
                result = await self._gemini_session_new(arguments)
            elif tool_name == "gemini_session_resume":
                result = await self._gemini_session_resume(arguments)
            elif tool_name == "gemini_session_stats":
                result = await self._gemini_session_stats(arguments)
            elif tool_name == "gemini_session_clear":
                result = await self._gemini_session_clear(arguments)
            else:
                return {
                    "error": {
                        "code": -32602,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    async def _gemini_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text with Gemini."""
        prompt = args["prompt"]
        session_id = args.get("session_id")
        thinking_mode = args.get("thinking_mode", False)
        grounding = args.get("grounding", False)
        code_execution = args.get("code_execution", False)

        cmd = [GEMINI_PATH, "exec", prompt]

        if session_id:
            cmd.extend(["--session", session_id])
        if thinking_mode:
            cmd.append("--thinking")
        if grounding:
            cmd.append("--grounding")
        if code_execution:
            cmd.append("--code-exec")

        # Add optional parameters
        if "temperature" in args:
            cmd.extend(["--temperature", str(args["temperature"])])
        if "max_tokens" in args:
            cmd.extend(["--max-tokens", str(args["max_tokens"])])

        result = await self._run_command(cmd)

        return {
            "success": result["success"],
            "response": result["stdout"] if result["success"] else result["stderr"],
            "thinking_mode": thinking_mode,
            "grounding": grounding,
            "code_execution": code_execution
        }

    async def _gemini_multimodal(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze multimodal content."""
        prompt = args["prompt"]
        media_path = args["media_path"]
        media_type = args["media_type"]
        session_id = args.get("session_id")

        cmd = [GEMINI_PATH, "multimodal", media_type, media_path, prompt]

        if session_id:
            cmd.extend(["--session", session_id])

        result = await self._run_command(cmd)

        return {
            "success": result["success"],
            "response": result["stdout"] if result["success"] else result["stderr"],
            "media_type": media_type,
            "media_path": media_path
        }

    async def _gemini_function_call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function calling."""
        prompt = args["prompt"]
        functions = args["functions"]
        session_id = args.get("session_id")

        # Write functions to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(functions, f)
            functions_file = f.name

        try:
            cmd = [GEMINI_PATH, "functions", prompt, "--functions-file", functions_file]

            if session_id:
                cmd.extend(["--session", session_id])

            result = await self._run_command(cmd)

            return {
                "success": result["success"],
                "response": result["stdout"] if result["success"] else result["stderr"],
                "function_count": len(functions)
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(functions_file)
            except:
                pass

    async def _gemini_code_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation and execution."""
        prompt = args["prompt"]
        session_id = args.get("session_id")

        cmd = [GEMINI_PATH, "code", prompt]

        if session_id:
            cmd.extend(["--session", session_id])

        result = await self._run_command(cmd)

        return {
            "success": result["success"],
            "response": result["stdout"] if result["success"] else result["stderr"]
        }

    async def _gemini_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get or update configuration."""
        action = args["action"]

        if action == "get":
            cmd = [GEMINI_PATH, "config", "show"]
            result = await self._run_command(cmd)

            return {
                "success": result["success"],
                "config": result["stdout"] if result["success"] else None,
                "error": result["stderr"] if not result["success"] else None
            }
        elif action == "set":
            config = args.get("config", {})
            results = []

            for key, value in config.items():
                cmd = [GEMINI_PATH, "config", "set", key, str(value)]
                result = await self._run_command(cmd)
                results.append({
                    "key": key,
                    "success": result["success"],
                    "message": result["stdout"] if result["success"] else result["stderr"]
                })

            return {
                "success": all(r["success"] for r in results),
                "results": results
            }

    async def _gemini_session_new(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session."""
        session_id = args["session_id"]
        thinking_mode = args.get("thinking_mode", False)
        grounding = args.get("grounding", False)

        self.sessions[session_id] = {
            "thinking_mode": thinking_mode,
            "grounding": grounding,
            "message_count": 0,
            "total_tokens": 0
        }

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Created new session: {session_id}"
        }

    async def _gemini_session_resume(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resume an existing session."""
        session_id = args["session_id"]

        if session_id in self.sessions:
            return {
                "success": True,
                "session_id": session_id,
                "session_info": self.sessions[session_id]
            }
        else:
            return {
                "success": False,
                "error": f"Session not found: {session_id}"
            }

    async def _gemini_session_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get session statistics."""
        session_id = args["session_id"]

        cmd = [GEMINI_PATH, "session", "stats", session_id]
        result = await self._run_command(cmd)

        return {
            "success": result["success"],
            "stats": result["stdout"] if result["success"] else None,
            "error": result["stderr"] if not result["success"] else None
        }

    async def _gemini_session_clear(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clear session history."""
        session_id = args["session_id"]

        cmd = [GEMINI_PATH, "session", "clear", session_id]
        result = await self._run_command(cmd)

        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] = 0
            self.sessions[session_id]["total_tokens"] = 0

        return {
            "success": result["success"],
            "message": result["stdout"] if result["success"] else result["stderr"]
        }

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run Gemini CLI command."""
        try:
            logger.info(f"Running command: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8').strip(),
                "stderr": stderr.decode('utf-8').strip(),
                "returncode": process.returncode
            }
        except Exception as e:
            logger.error(f"Command failed: {e}", exc_info=True)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }


async def main():
    """Main server loop using stdin/stdout for MCP protocol."""
    server = GeminiMCPServer()

    logger.info("Gemini MCP Server started")

    # Read from stdin, write to stdout (MCP protocol)
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )

            if not line:
                break

            request = json.loads(line)
            response = await server.handle_request(request)

            # Write response to stdout
            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            error_response = {
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            error_response = {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
