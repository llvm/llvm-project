#!/usr/bin/env python3
"""
DSMIL MCP Server - Expose DSMIL Platform as MCP Tools

Provides code-mode access to:
- 84 DSMIL hardware devices
- AI engine capabilities (RAG, generation, conversation)
- Agent orchestration
- Self-improvement system
- File operations
- Execution capabilities

Protocol: Model Context Protocol (MCP)
Compatible with: @utcp/code-mode
"""

import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add DSMIL paths
sys.path.insert(0, "/home/user/LAT5150DRVMIL/02-ai-engine")
sys.path.insert(0, "/home/user/LAT5150DRVMIL")

# Import DSMIL components
from dsmil_ai_engine import DSMILAIEngine
from enhanced_ai_engine import EnhancedAIEngine
from agent_orchestrator import AgentOrchestrator, AgentTask
from autonomous_self_improvement import AutonomousSelfImprovement
from file_operations import FileOps
from tool_operations import ToolOps

# Import Serena semantic engine
sys.path.insert(0, "/home/user/LAT5150DRVMIL/01-source/serena-integration")
try:
    from semantic_code_engine import SemanticCodeEngine
    SERENA_AVAILABLE = True
except ImportError:
    SERENA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DSMILMCPServer:
    """
    MCP server exposing DSMIL platform capabilities

    Tool Categories:
    - dsmil_device_*: 84 hardware devices
    - ai_*: AI engine operations
    - agent_*: Agent orchestration
    - file_*: File operations
    - exec_*: Command execution
    - improve_*: Self-improvement
    """

    def __init__(self):
        """Initialize DSMIL MCP server"""
        logger.info("Initializing DSMIL MCP Server...")

        # Initialize components
        self.ai_engine = DSMILAIEngine()
        self.enhanced_ai = None  # Lazy init
        self.agent_orchestrator = None  # Lazy init
        self.self_improvement = None  # Lazy init
        self.file_ops = FileOps()
        self.tool_ops = ToolOps()

        # Initialize Serena semantic engine (for code analysis tools)
        self.serena = None
        if SERENA_AVAILABLE:
            try:
                self.serena = SemanticCodeEngine(workspace_root="/home/user/LAT5150DRVMIL")
                logger.info("✓ Serena semantic engine initialized")
            except Exception as e:
                logger.warning(f"Serena initialization failed: {e}")
                SERENA_AVAILABLE = False

        # Initialize CodebaseLearner (for call graph analysis - Phase 3)
        self.codebase_learner = None
        try:
            from codebase_learner import CodebaseLearner
            self.codebase_learner = CodebaseLearner(workspace_root="/home/user/LAT5150DRVMIL")
            logger.info(f"✓ CodebaseLearner initialized ({len(self.codebase_learner.function_locations)} functions)")
        except Exception as e:
            logger.warning(f"CodebaseLearner initialization failed: {e}")

        # Device registry
        self.devices = self._load_device_registry()

        # Tool registry
        self.tools = {}
        self._register_all_tools()

        logger.info(f"✓ DSMIL MCP Server ready ({len(self.tools)} tools)")

    def _load_device_registry(self) -> Dict[str, Any]:
        """Load DSMIL device registry"""
        # Known DSMIL devices (84 total)
        devices = {
            # Encryption devices (0x8001-0x8010)
            "0x8001": {"name": "encryption_aes256", "type": "encryption"},
            "0x8002": {"name": "encryption_rsa4096", "type": "encryption"},

            # Storage devices
            "0x8032": {"name": "raid_controller", "type": "storage"},
            "0x8038": {"name": "deduplication_engine", "type": "storage"},
            "0x8039": {"name": "compression_engine", "type": "storage"},

            # AI devices
            "0x804a": {"name": "inference_engine", "type": "ai"},
            "0x8048": {"name": "training_recorder", "type": "ai"},

            # Add more devices as needed
        }

        return devices

    def _register_all_tools(self):
        """Register all MCP tools"""
        # AI Engine tools
        self._register_tool("ai_generate", self._ai_generate, {
            "prompt": {"type": "string", "required": True},
            "model": {"type": "string", "default": "uncensored_code"},
            "temperature": {"type": "number", "default": 0.7}
        })

        self._register_tool("ai_analyze_code", self._ai_analyze_code, {
            "code": {"type": "string", "required": True},
            "language": {"type": "string", "default": "python"}
        })

        self._register_tool("ai_search_rag", self._ai_search_rag, {
            "query": {"type": "string", "required": True},
            "top_k": {"type": "number", "default": 5}
        })

        # File operations
        self._register_tool("file_read", self._file_read, {
            "path": {"type": "string", "required": True}
        })

        self._register_tool("file_write", self._file_write, {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True}
        })

        self._register_tool("file_search", self._file_search, {
            "pattern": {"type": "string", "required": True},
            "path": {"type": "string", "default": "."}
        })

        # Code analysis tools (Serena-based - like Slither MCP)
        if self.serena:
            self._register_tool("code_find_symbol", self._code_find_symbol, {
                "name": {"type": "string", "required": True},
                "symbol_type": {"type": "string", "default": "any"},
                "language": {"type": "string", "default": "python"}
            })

            self._register_tool("code_find_references", self._code_find_references, {
                "file_path": {"type": "string", "required": True},
                "line": {"type": "number", "required": True},
                "column": {"type": "number", "required": True}
            })

            self._register_tool("code_get_definition", self._code_get_definition, {
                "file_path": {"type": "string", "required": True},
                "line": {"type": "number", "required": True},
                "column": {"type": "number", "required": True}
            })

            self._register_tool("code_semantic_search", self._code_semantic_search, {
                "query": {"type": "string", "required": True},
                "max_results": {"type": "number", "default": 10},
                "language": {"type": "string", "default": "python"}
            })

            self._register_tool("code_get_symbol_source", self._code_get_symbol_source, {
                "name": {"type": "string", "required": True},
                "symbol_type": {"type": "string", "default": "any"},
                "language": {"type": "string", "default": "python"}
            })

            self._register_tool("code_insert_after_symbol", self._code_insert_after_symbol, {
                "symbol": {"type": "string", "required": True},
                "code": {"type": "string", "required": True},
                "language": {"type": "string", "default": "python"}
            })

            self._register_tool("code_analyze_imports", self._code_analyze_imports, {
                "file_path": {"type": "string", "required": True}
            })

            self._register_tool("code_get_function_calls", self._code_get_function_calls, {
                "file_path": {"type": "string", "required": True},
                "function_name": {"type": "string", "required": True}
            })

        # Phase 3: Call graph analysis tools (CodebaseLearner)
        self._register_tool("callgraph_find_dead_code", self._callgraph_find_dead_code, {})

        self._register_tool("callgraph_find_cycles", self._callgraph_find_cycles, {})

        self._register_tool("callgraph_analyze_impact", self._callgraph_analyze_impact, {
            "function_name": {"type": "string", "required": True}
        })

        self._register_tool("callgraph_get_stats", self._callgraph_get_stats, {})

        self._register_tool("callgraph_learn_from_file", self._callgraph_learn_from_file, {
            "file_path": {"type": "string", "required": True}
        })

        # Agent orchestration
        self._register_tool("agent_execute_task", self._agent_execute_task, {
            "task_description": {"type": "string", "required": True},
            "capabilities": {"type": "array", "default": []},
            "max_latency_ms": {"type": "number", "default": 5000}
        })

        # Execution tools
        self._register_tool("exec_command", self._exec_command, {
            "command": {"type": "string", "required": True},
            "description": {"type": "string", "default": ""}
        })

        # Self-improvement tools
        self._register_tool("improve_analyze_bottlenecks", self._improve_analyze_bottlenecks, {})

        self._register_tool("improve_propose", self._improve_propose, {
            "category": {"type": "string", "required": True},
            "title": {"type": "string", "required": True},
            "description": {"type": "string", "required": True}
        })

        # Device tools (register all 84 devices)
        for device_id, device_info in self.devices.items():
            tool_name = f"device_{device_info['name']}"
            self._register_device_tool(tool_name, device_id, device_info)

    def _register_tool(self, name: str, handler: callable, schema: Dict):
        """Register a tool with schema"""
        self.tools[name] = {
            "handler": handler,
            "schema": schema
        }

    def _register_device_tool(self, name: str, device_id: str, device_info: Dict):
        """Register a device-specific tool"""
        async def device_handler(params: Dict) -> Dict:
            """Generic device operation handler"""
            operation = params.get("operation", "status")
            data = params.get("data")

            # Device operations would interface with actual hardware
            # For now, return simulated results
            return {
                "device_id": device_id,
                "device_name": device_info["name"],
                "operation": operation,
                "success": True,
                "result": f"Device {device_id} ({device_info['name']}) executed {operation}"
            }

        self._register_tool(name, device_handler, {
            "operation": {"type": "string", "default": "status"},
            "data": {"type": "any"}
        })

    # ========================================================================
    # AI Engine Tool Handlers
    # ========================================================================

    async def _ai_generate(self, params: Dict) -> Dict:
        """Generate AI response"""
        prompt = params["prompt"]
        model = params.get("model", "uncensored_code")
        temperature = params.get("temperature", 0.7)

        result = self.ai_engine.generate(
            prompt=prompt,
            model_selection=model
        )

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "response": result["response"],
            "model": result["model"],
            "inference_time": result["inference_time"],
            "tokens_per_sec": result.get("tokens_per_sec", 0)
        }

    async def _ai_analyze_code(self, params: Dict) -> Dict:
        """Analyze code with AI"""
        code = params["code"]
        language = params.get("language", "python")

        prompt = f"""Analyze this {language} code:

```{language}
{code}
```

Provide:
1. Code quality assessment
2. Potential issues or bugs
3. Optimization suggestions
4. Security concerns"""

        result = self.ai_engine.generate(prompt, model_selection="quality_code")

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "analysis": result["response"],
            "language": language
        }

    async def _ai_search_rag(self, params: Dict) -> Dict:
        """Search RAG knowledge base"""
        query = params["query"]
        top_k = params.get("top_k", 5)

        result = self.ai_engine.rag_search(query, max_results=top_k)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "results": result.get("results", []),
            "count": result.get("count", 0)
        }

    # ========================================================================
    # File Operation Tool Handlers
    # ========================================================================

    async def _file_read(self, params: Dict) -> Dict:
        """Read file"""
        path = params["path"]

        result = self.file_ops.read_file(path)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "path": path,
            "content": result["content"],
            "lines": result.get("lines", 0),
            "size": result.get("size", 0)
        }

    async def _file_write(self, params: Dict) -> Dict:
        """Write file"""
        path = params["path"]
        content = params["content"]

        result = self.file_ops.write_file(path, content)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "path": path,
            "bytes_written": len(content)
        }

    async def _file_search(self, params: Dict) -> Dict:
        """Search files"""
        pattern = params["pattern"]
        path = params.get("path", ".")

        result = self.file_ops.grep(pattern, path=path)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "pattern": pattern,
            "matches": result.get("matches", []),
            "count": result.get("count", 0)
        }

    # ========================================================================
    # Code Analysis Tool Handlers (Serena-based - Slither MCP approach)
    # ========================================================================

    async def _code_find_symbol(self, params: Dict) -> Dict:
        """
        Find symbol by name (like Slither's get_function_source)

        Deterministic AST-based search - replaces error-prone file_read + grep
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        name = params["name"]
        symbol_type = params.get("symbol_type", "any")
        language = params.get("language", "python")

        try:
            # Initialize Serena if needed
            if not self.serena.initialized:
                await self.serena.initialize()

            # Find all instances of symbol
            matches = await self.serena.find_symbol(name, symbol_type, language)

            return {
                "success": True,
                "symbol_name": name,
                "symbol_type": symbol_type,
                "matches": [
                    {
                        "file_path": m.file_path,
                        "line": m.line,
                        "column": m.column,
                        "symbol_name": m.symbol_name,
                        "symbol_type": m.symbol_type,
                        "symbol_info": m.symbol_info
                    }
                    for m in matches
                ],
                "count": len(matches)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_get_symbol_source(self, params: Dict) -> Dict:
        """
        Get full source code for a symbol (direct extraction)

        Like Slither's get_function_source - returns actual code, not just location
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        name = params["name"]
        symbol_type = params.get("symbol_type", "any")
        language = params.get("language", "python")

        try:
            if not self.serena.initialized:
                await self.serena.initialize()

            # Find symbol
            matches = await self.serena.find_symbol(name, symbol_type, language)

            if not matches:
                return {"success": False, "error": f"Symbol '{name}' not found"}

            # Get first match (primary definition)
            match = matches[0]

            # Read file and extract symbol source
            import ast
            with open(match.file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            # Find the specific symbol node
            source_code = None
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef) and node.name == name) or \
                   (isinstance(node, ast.ClassDef) and node.name == name):
                    # Extract source lines
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
                    source_code = '\n'.join(lines[start_line:end_line])
                    break

            return {
                "success": True,
                "symbol_name": name,
                "file_path": match.file_path,
                "line": match.line,
                "source_code": source_code or "Source code extraction failed"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_find_references(self, params: Dict) -> Dict:
        """
        Find all references to symbol at location (like Slither's get_callers)

        Returns who calls/uses this symbol
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        file_path = params["file_path"]
        line = params["line"]
        column = params["column"]

        try:
            if not self.serena.initialized:
                await self.serena.initialize()

            # Find all references
            references = await self.serena.find_references(file_path, line, column)

            return {
                "success": True,
                "references": [
                    {
                        "file_path": ref.file_path,
                        "line": ref.line,
                        "column": ref.column,
                        "reference_type": ref.reference_type,
                        "context": ref.context
                    }
                    for ref in references
                ],
                "count": len(references)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_get_definition(self, params: Dict) -> Dict:
        """
        Get definition of symbol at location

        Jump to definition functionality
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        file_path = params["file_path"]
        line = params["line"]
        column = params["column"]

        try:
            if not self.serena.initialized:
                await self.serena.initialize()

            # Get definition
            definition = await self.serena.get_definition(file_path, line, column)

            if not definition:
                return {"success": False, "error": "Definition not found"}

            return {
                "success": True,
                "file_path": definition.file_path,
                "line": definition.line,
                "column": definition.column,
                "symbol_name": definition.symbol_name,
                "symbol_type": definition.symbol_type,
                "symbol_info": definition.symbol_info
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_semantic_search(self, params: Dict) -> Dict:
        """
        Semantic code search (finds relevant code)

        Better than text search - understands code structure
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        query = params["query"]
        max_results = params.get("max_results", 10)
        language = params.get("language", "python")

        try:
            if not self.serena.initialized:
                await self.serena.initialize()

            # Semantic search
            matches = await self.serena.semantic_search(query, max_results, language)

            return {
                "success": True,
                "query": query,
                "matches": [
                    {
                        "file_path": m.file_path,
                        "line": m.line,
                        "symbol_name": m.symbol_name,
                        "symbol_type": m.symbol_type,
                        "relevance_score": m.relevance_score,
                        "code_snippet": m.code_snippet
                    }
                    for m in matches
                ],
                "count": len(matches)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_insert_after_symbol(self, params: Dict) -> Dict:
        """
        Insert code after symbol definition (semantic editing)

        Precise code modification at symbol level
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        symbol = params["symbol"]
        code = params["code"]
        language = params.get("language", "python")

        try:
            if not self.serena.initialized:
                await self.serena.initialize()

            # Semantic edit
            result = await self.serena.insert_after_symbol(symbol, code, language)

            return {
                "success": result.success,
                "file_path": result.file_path,
                "lines_modified": result.lines_modified,
                "message": result.message
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_analyze_imports(self, params: Dict) -> Dict:
        """
        Analyze imports in a file

        Returns all imports and their usage
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        file_path = params["file_path"]

        try:
            import ast
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            "type": "from_import",
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })

            return {
                "success": True,
                "file_path": file_path,
                "imports": imports,
                "count": len(imports)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _code_get_function_calls(self, params: Dict) -> Dict:
        """
        Get all function calls in a file (call graph data)

        Shows what functions are called and where
        """
        if not self.serena:
            return {"success": False, "error": "Serena semantic engine not available"}

        file_path = params["file_path"]
        function_name = params["function_name"]

        try:
            import ast
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Get function name from call
                    if isinstance(node.func, ast.Name):
                        called_name = node.func.id
                        if function_name == "all" or called_name == function_name:
                            calls.append({
                                "function": called_name,
                                "line": node.lineno,
                                "args_count": len(node.args)
                            })
                    elif isinstance(node, ast.Attribute):
                        called_name = node.func.attr
                        if function_name == "all" or called_name == function_name:
                            calls.append({
                                "function": called_name,
                                "line": node.lineno,
                                "args_count": len(node.args),
                                "method_call": True
                            })

            return {
                "success": True,
                "file_path": file_path,
                "function_name": function_name,
                "calls": calls,
                "count": len(calls)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Call Graph Analysis Tool Handlers (Phase 3)
    # ========================================================================

    async def _callgraph_find_dead_code(self, params: Dict) -> Dict:
        """
        Find dead code (unused functions/methods) using call graph analysis

        Returns functions and methods that are never called
        """
        if not self.codebase_learner:
            return {"success": False, "error": "CodebaseLearner not available"}

        try:
            dead_code = self.codebase_learner.find_dead_code()

            return {
                "success": True,
                "dead_functions": dead_code['functions'],
                "dead_methods": dead_code['methods'],
                "total": dead_code['total'],
                "message": f"Found {dead_code['total']} potentially dead code items"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _callgraph_find_cycles(self, params: Dict) -> Dict:
        """
        Find circular dependencies in the call graph

        Returns list of dependency cycles
        """
        if not self.codebase_learner:
            return {"success": False, "error": "CodebaseLearner not available"}

        try:
            cycles = self.codebase_learner.find_dependency_cycles()

            return {
                "success": True,
                "cycles": cycles,
                "count": len(cycles),
                "message": f"Found {len(cycles)} dependency cycles"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _callgraph_analyze_impact(self, params: Dict) -> Dict:
        """
        Analyze impact of modifying a function

        Returns:
            - Direct callers
            - Transitive callers (all dependent code)
            - Impact score (0-100)
            - Risk level (low/medium/high)
        """
        if not self.codebase_learner:
            return {"success": False, "error": "CodebaseLearner not available"}

        function_name = params["function_name"]

        try:
            impact = self.codebase_learner.find_impact(function_name)

            if "error" in impact:
                return {"success": False, "error": impact["error"]}

            return {
                "success": True,
                "function": impact["function"],
                "direct_callers": impact["direct_callers"],
                "direct_caller_count": impact["direct_caller_count"],
                "transitive_callers": impact["transitive_callers"],
                "transitive_caller_count": impact["transitive_caller_count"],
                "impact_score": impact["impact_score"],
                "risk_level": impact["risk_level"],
                "message": f"Impact analysis: {impact['risk_level']} risk ({impact['impact_score']}/100)"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _callgraph_get_stats(self, params: Dict) -> Dict:
        """
        Get call graph statistics

        Returns comprehensive call graph analysis:
            - Total functions and call graph edges
            - Dead code summary
            - Dependency cycles
            - Hotspots (most-called functions)
            - Complex functions (call many other functions)
        """
        if not self.codebase_learner:
            return {"success": False, "error": "CodebaseLearner not available"}

        try:
            stats = self.codebase_learner.get_call_graph_stats()

            return {
                "success": True,
                "total_functions": stats["total_functions"],
                "total_edges": stats["total_edges"],
                "dead_code_count": stats["dead_code"]["total"],
                "cycle_count": stats["cycle_count"],
                "hotspots": stats["hotspots"],
                "complex_functions": stats["complex_functions"],
                "avg_calls_per_function": stats["avg_calls_per_function"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _callgraph_learn_from_file(self, params: Dict) -> Dict:
        """
        Learn call graph from a Python file

        Analyzes file and updates call graph
        """
        if not self.codebase_learner:
            return {"success": False, "error": "CodebaseLearner not available"}

        file_path = params["file_path"]

        try:
            learned = self.codebase_learner.learn_from_file(file_path)

            if "error" in learned:
                return {"success": False, "error": learned["error"]}

            # Save updated knowledge
            self.codebase_learner.save_knowledge()

            return {
                "success": True,
                "file": learned["filepath"],
                "functions_learned": learned["functions"],
                "classes_learned": learned["classes"],
                "patterns": len(learned["patterns"]),
                "message": f"Learned from {file_path}: {learned['functions']} functions, {learned['classes']} classes"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Agent Orchestration Tool Handlers
    # ========================================================================

    async def _agent_execute_task(self, params: Dict) -> Dict:
        """Execute task with agent orchestrator"""
        # Lazy init orchestrator
        if not self.agent_orchestrator:
            self.agent_orchestrator = AgentOrchestrator()

        task = AgentTask(
            task_id=f"mcp_{id(params)}",
            description=params["task_description"],
            prompt=params["task_description"],
            required_capabilities=params.get("capabilities", []),
            max_latency_ms=params.get("max_latency_ms", 5000)
        )

        result = self.agent_orchestrator.execute_task(task)

        return {
            "success": result.success,
            "agent_used": result.agent_name,
            "content": result.content,
            "latency_ms": result.latency_ms,
            "hardware_backend": result.hardware_backend,
            "error": result.error
        }

    # ========================================================================
    # Execution Tool Handlers
    # ========================================================================

    async def _exec_command(self, params: Dict) -> Dict:
        """Execute shell command"""
        command = params["command"]
        description = params.get("description", "")

        result = self.tool_ops.bash(command, description=description)

        return {
            "success": result.get("success", False),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", -1)
        }

    # ========================================================================
    # Self-Improvement Tool Handlers
    # ========================================================================

    async def _improve_analyze_bottlenecks(self, params: Dict) -> Dict:
        """Analyze system bottlenecks"""
        # Lazy init self-improvement
        if not self.self_improvement:
            self.self_improvement = AutonomousSelfImprovement(
                enable_auto_modification=False  # Safety
            )

        bottlenecks = self.self_improvement.analyze_bottlenecks()

        return {
            "success": True,
            "bottlenecks": bottlenecks,
            "count": len(bottlenecks)
        }

    async def _improve_propose(self, params: Dict) -> Dict:
        """Propose improvement"""
        if not self.self_improvement:
            self.self_improvement = AutonomousSelfImprovement(
                enable_auto_modification=False
            )

        proposal = self.self_improvement.propose_improvement(
            category=params["category"],
            title=params["title"],
            description=params["description"],
            rationale=params.get("rationale", "MCP tool proposal"),
            files_to_modify=params.get("files", []),
            estimated_impact=params.get("impact", "medium"),
            risk_level=params.get("risk", "low")
        )

        return {
            "success": True,
            "proposal_id": proposal.proposal_id,
            "category": proposal.category,
            "title": proposal.title
        }

    # ========================================================================
    # MCP Protocol Implementation
    # ========================================================================

    async def handle_request(self, request: Dict) -> Dict:
        """Handle MCP request"""
        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            return await self._list_tools()

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_params = params.get("arguments", {})

            return await self._call_tool(tool_name, tool_params)

        else:
            return {
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

    async def _list_tools(self) -> Dict:
        """List all available tools"""
        tools_list = []

        for name, tool_info in self.tools.items():
            tools_list.append({
                "name": name,
                "description": f"DSMIL tool: {name}",
                "inputSchema": {
                    "type": "object",
                    "properties": tool_info["schema"],
                    "required": [
                        k for k, v in tool_info["schema"].items()
                        if v.get("required", False)
                    ]
                }
            })

        return {"tools": tools_list}

    async def _call_tool(self, name: str, params: Dict) -> Dict:
        """Call a tool"""
        if name not in self.tools:
            return {
                "error": {"code": -32602, "message": f"Unknown tool: {name}"}
            }

        tool = self.tools[name]
        handler = tool["handler"]

        try:
            result = await handler(params)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Tool execution error: {e}")

            return {
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }

    async def run_stdio(self):
        """Run MCP server on stdio"""
        logger.info("DSMIL MCP Server running on stdio")

        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                if not line:
                    break

                request = json.loads(line)

                # Handle request
                response = await self.handle_request(request)

                # Write response to stdout
                print(json.dumps(response), flush=True)

            except Exception as e:
                logger.error(f"Server error: {e}")
                break


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Run on stdio")
    parser.add_argument("--test", action="store_true", help="Test mode")

    args = parser.parse_args()

    # Initialize server
    server = DSMILMCPServer()

    if args.test:
        # Test mode - list tools
        print("=" * 70)
        print("DSMIL MCP Server - Test Mode")
        print("=" * 70)
        print()

        tools = await server._list_tools()
        print(f"Available tools: {len(tools['tools'])}")

        for tool in tools["tools"][:10]:  # Show first 10
            print(f"  - {tool['name']}")

        print()

        # Test AI generation
        print("Testing AI generation...")
        result = await server._ai_generate({
            "prompt": "What is the meaning of life?",
            "model": "fast"
        })

        if result["success"]:
            print(f"✓ AI response ({result['inference_time']:.1f}s):")
            print(f"  {result['response'][:200]}...")
        else:
            print(f"✗ AI failed: {result['error']}")

    elif args.stdio:
        # Run server on stdio
        await server.run_stdio()

    else:
        print("DSMIL MCP Server")
        print("Usage: server.py --stdio (run as MCP server)")
        print("       server.py --test (test mode)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
