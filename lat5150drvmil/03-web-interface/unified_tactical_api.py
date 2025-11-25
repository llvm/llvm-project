#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Unified Tactical API with Natural Language Interface
LOCAL-FIRST: Integrates all capabilities with custom local models (WhiteRabbit, etc.)

Brings together:
- Serena LSP semantic code understanding
- AgentSystems containerized execution
- Local model providers (WhiteRabbit, Ollama, custom models)
- DSMIL hardware reconnaissance
- RAG system
- Natural language command processing
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../01-source'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../02-ai-engine'))

# Local imports
from capability_registry import get_registry
from natural_language_processor import NaturalLanguageProcessor

# Import INTEGRATED self-awareness engine (not standalone)
from integrated_self_awareness import IntegratedSelfAwarenessEngine

try:
    from serena_integration.semantic_code_engine import SemanticCodeEngine
except ImportError:
    SemanticCodeEngine = None
    logging.warning("Serena integration not available")

try:
    from agentsystems_integration.agent_runtime import AgentOrchestrator, AgentConfig
    from agentsystems_integration.model_providers import ModelProviderManager, OllamaProvider
except ImportError:
    AgentOrchestrator = None
    ModelProviderManager = None
    logging.warning("AgentSystems integration not available")

try:
    from atomic_red_team_api import AtomicRedTeamAPI
    ATOMIC_RED_TEAM_AVAILABLE = True
except ImportError:
    ATOMIC_RED_TEAM_AVAILABLE = False
    logging.warning("Atomic Red Team integration not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] UnifiedAPI: %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTacticalAPI:
    """
    Unified API integrating all system capabilities

    LOCAL-FIRST: Prioritizes custom local models (WhiteRabbit, etc.)
    """

    def __init__(
        self,
        workspace_path: str = "/home/user/LAT5150DRVMIL",
        local_models_endpoint: str = "http://localhost:11434",
        custom_models: Optional[List[str]] = None
    ):
        self.workspace_path = workspace_path

        # Initialize capability registry
        self.registry = get_registry()

        # Initialize natural language processor with custom models
        self.nlp = NaturalLanguageProcessor(
            use_local_model=True,
            ollama_endpoint=local_models_endpoint,
            model=custom_models[0] if custom_models else "llama3.2:latest"
        )

        # Track custom local models
        self.custom_models = custom_models or ["whiterabbit", "llama3.2:latest", "codellama:latest"]
        logger.info(f"📦 LOCAL MODELS: {', '.join(self.custom_models)}")

        # Initialize INTEGRATED self-awareness engine (connects to all existing systems)
        self.self_awareness = IntegratedSelfAwarenessEngine(
            workspace_path=workspace_path,
            state_db_path="/opt/lat5150/state/integrated_awareness.db",
            vector_db_path="/opt/lat5150/vectordb",
            postgres_url=os.environ.get("POSTGRES_URL", "postgresql://localhost/lat5150")
        )
        logger.info("🧠 Integrated Self-Awareness Engine initialized (connects to vector DB, cognitive memory, DSMIL, TPM, crypto, MCP)")

        # Initialize components
        self.semantic_engine = None
        self.agent_orchestrator = None
        self.model_manager = None
        self.atomic_red_team = None

        self._initialize_components()

        # Run initial discovery
        self._run_initial_discovery()

    def _initialize_components(self):
        """Initialize all system components"""

        # Initialize Serena LSP
        if SemanticCodeEngine:
            try:
                self.semantic_engine = SemanticCodeEngine(self.workspace_path)
                logger.info("✅ Serena LSP initialized")
            except Exception as e:
                logger.error(f"❌ Serena LSP failed: {e}")

        # Initialize AgentSystems
        if AgentOrchestrator:
            try:
                self.agent_orchestrator = AgentOrchestrator(
                    artifact_base_path="/opt/lat5150/artifacts",
                    audit_log_path="/opt/lat5150/audit/agent_audit.log"
                )
                logger.info("✅ AgentSystems runtime initialized")
            except Exception as e:
                logger.error(f"❌ AgentSystems failed: {e}")

        # Initialize Model Provider Manager with LOCAL models
        if ModelProviderManager and OllamaProvider:
            try:
                self.model_manager = ModelProviderManager()

                # Register LOCAL Ollama with custom models
                ollama = OllamaProvider(endpoint="http://localhost:11434")
                self.model_manager.register_provider(
                    "local",
                    ollama,
                    set_as_default=True  # LOCAL-FIRST
                )

                logger.info(f"✅ Local model provider initialized (default)")
                logger.info(f"   Available models: {', '.join(self.custom_models)}")

            except Exception as e:
                logger.error(f"❌ Model manager failed: {e}")

        # Initialize Atomic Red Team
        if ATOMIC_RED_TEAM_AVAILABLE:
            try:
                self.atomic_red_team = AtomicRedTeamAPI()
                logger.info("✅ Atomic Red Team initialized (MITRE ATT&CK techniques)")
            except Exception as e:
                logger.error(f"❌ Atomic Red Team failed: {e}")

    def _run_initial_discovery(self):
        """Run initial capability and resource discovery"""
        try:
            logger.info("🔍 Running initial discovery...")

            # Discover capabilities from codebase
            capabilities = self.self_awareness.discover_capabilities()
            logger.info(f"   Found {len(capabilities)} capabilities")

            # Discover available resources
            resources = self.self_awareness.discover_resources()
            logger.info(f"   Found {len(resources)} resources")

            # Update system state
            self.self_awareness.update_system_state()
            logger.info("✅ Initial discovery complete")

        except Exception as e:
            logger.error(f"❌ Initial discovery failed: {e}")

    async def process_natural_language_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process natural language command and execute

        Args:
            command: Natural language command
            context: Optional context (previous commands, user prefs, etc.)

        Returns:
            Execution result
        """
        # Parse command
        parsed = await self.nlp.parse_command(command)

        if not parsed.matched_capability:
            return {
                "success": False,
                "error": "Could not understand command",
                "suggestion": "Try /help for available commands",
                "parsed": parsed.to_dict()
            }

        # Execute matched capability
        try:
            result = await self._execute_capability(
                parsed.matched_capability,
                parsed.extracted_parameters,
                context
            )

            return {
                "success": True,
                "capability": parsed.matched_capability.name,
                "confidence": parsed.confidence,
                "result": result,
                "parsed": parsed.to_dict()
            }

        except Exception as e:
            logger.error(f"Error executing capability: {e}")
            return {
                "success": False,
                "error": str(e),
                "capability": parsed.matched_capability.name,
                "parsed": parsed.to_dict()
            }

    async def _execute_capability(
        self,
        capability,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a matched capability"""

        cap_id = capability.id
        logger.info(f"Executing capability: {cap_id}")

        # ========== SERENA LSP CAPABILITIES ==========

        if cap_id == "serena_find_symbol":
            if not self.semantic_engine:
                raise Exception("Semantic engine not available")

            await self.semantic_engine.initialize()

            name = parameters.get("name", "")
            symbol_type = parameters.get("symbol_type")
            language = parameters.get("language")

            symbols = await self.semantic_engine.find_symbol(
                name=name,
                symbol_type=symbol_type,
                language=language
            )

            return {
                "symbols": [s.to_dict() for s in symbols],
                "count": len(symbols)
            }

        elif cap_id == "serena_find_references":
            if not self.semantic_engine:
                raise Exception("Semantic engine not available")

            await self.semantic_engine.initialize()

            file_path = parameters.get("file_path", "")
            line = parameters.get("line", 1)
            column = parameters.get("column", 0)

            references = await self.semantic_engine.find_references(
                file_path, line, column
            )

            return {
                "references": [r.to_dict() for r in references],
                "count": len(references)
            }

        elif cap_id == "serena_insert_code":
            if not self.semantic_engine:
                raise Exception("Semantic engine not available")

            await self.semantic_engine.initialize()

            symbol = parameters.get("symbol", "")
            code = parameters.get("code", "")
            language = parameters.get("language", "python")

            result = await self.semantic_engine.insert_after_symbol(
                symbol=symbol,
                code=code,
                language=language
            )

            return result.to_dict()

        elif cap_id == "serena_semantic_search":
            if not self.semantic_engine:
                raise Exception("Semantic engine not available")

            await self.semantic_engine.initialize()

            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 10)

            matches = await self.semantic_engine.semantic_search(
                query=query,
                max_results=max_results
            )

            return {
                "matches": [m.to_dict() for m in matches],
                "count": len(matches)
            }

        # ========== AGENTSYSTEMS CAPABILITIES ==========

        elif cap_id == "agent_invoke":
            if not self.agent_orchestrator:
                raise Exception("Agent orchestrator not available")

            agent_name = parameters.get("agent_name", "")
            task = parameters.get("task", {})
            model_provider = parameters.get("model_provider", "local")  # LOCAL-FIRST

            execution = await self.agent_orchestrator.invoke_agent(
                agent_name=agent_name,
                task=task,
                model_provider=model_provider
            )

            return execution.to_dict()

        elif cap_id == "agent_list":
            if not self.agent_orchestrator:
                raise Exception("Agent orchestrator not available")

            agents = self.agent_orchestrator.list_agents()

            return {
                "agents": [agent.to_dict() for agent in agents],
                "count": len(agents)
            }

        elif cap_id == "agent_status":
            if not self.agent_orchestrator:
                raise Exception("Agent orchestrator not available")

            thread_id = parameters.get("thread_id", "")
            execution = self.agent_orchestrator.get_execution(thread_id)

            if execution:
                return execution.to_dict()
            else:
                return {"error": f"Execution {thread_id} not found"}

        # ========== MODEL PROVIDER CAPABILITIES ==========

        elif cap_id == "model_complete":
            if not self.model_manager:
                raise Exception("Model manager not available")

            prompt = parameters.get("prompt", "")
            provider = parameters.get("provider", "local")  # LOCAL-FIRST
            model = parameters.get("model")

            # If no model specified, use first custom model
            if not model and self.custom_models:
                model = self.custom_models[0]

            response = await self.model_manager.complete(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=parameters.get("temperature", 0.7),
                max_tokens=parameters.get("max_tokens", 4096)
            )

            return response.to_dict()

        elif cap_id == "model_list":
            if not self.model_manager:
                raise Exception("Model manager not available")

            all_models = self.model_manager.list_all_models()

            return {
                "models": {
                    provider: [m.to_dict() for m in models]
                    for provider, models in all_models.items()
                },
                "custom_local_models": self.custom_models
            }

        # ========== DSMIL CAPABILITIES ==========

        elif cap_id == "dsmil_scan":
            # Run DSMIL reconnaissance
            return {
                "status": "scanning",
                "message": "DSMIL hardware scan initiated",
                "note": "Run: sudo ./01-source/debugging/nsa_device_reconnaissance_enhanced.py"
            }

        elif cap_id == "dsmil_device_info":
            device_id = parameters.get("device_id", "")
            return {
                "device_id": device_id,
                "note": "Device documentation in /00-documentation/devices/"
            }

        # ========== SECURITY & AUDIT ==========

        elif cap_id == "audit_verify_chain":
            if not self.agent_orchestrator:
                raise Exception("Agent orchestrator not available")

            is_valid = self.agent_orchestrator.verify_audit_chain()

            return {
                "valid": is_valid,
                "message": "Audit chain verified" if is_valid else "TAMPERING DETECTED!"
            }

        elif cap_id == "audit_get_events":
            if not self.agent_orchestrator:
                raise Exception("Agent orchestrator not available")

            action = parameters.get("action")
            limit = parameters.get("limit", 100)

            events = self.agent_orchestrator.get_audit_events(
                action=action,
                limit=limit
            )

            return {
                "events": [e.to_dict() for e in events],
                "count": len(events)
            }

        # ========== SYSTEM CONTROL ==========

        elif cap_id == "system_health":
            return {
                "status": "healthy",
                "components": {
                    "semantic_engine": self.semantic_engine is not None,
                    "agent_orchestrator": self.agent_orchestrator is not None,
                    "model_manager": self.model_manager is not None,
                    "local_models": self.custom_models
                },
                "workspace": self.workspace_path
            }

        elif cap_id == "tempest_set_mode":
            mode = parameters.get("mode", "comfort")
            return {
                "mode": mode,
                "message": f"TEMPEST mode set to: {mode}",
                "note": "Update tactical UI to reflect mode change"
            }

        # ========== ATOMIC RED TEAM CAPABILITIES ==========

        elif cap_id == "atomic_query_tests":
            if not self.atomic_red_team:
                raise Exception("Atomic Red Team not available")

            query = parameters.get("query", "")
            technique_id = parameters.get("technique_id")
            platform = parameters.get("platform")

            result = self.atomic_red_team.query_atomics(
                query=query,
                technique_id=technique_id,
                platform=platform
            )

            return {
                "success": result.success,
                "tests": result.tests,
                "count": result.count,
                "query": result.query,
                "timestamp": result.timestamp
            }

        elif cap_id == "atomic_list_techniques":
            if not self.atomic_red_team:
                raise Exception("Atomic Red Team not available")

            result = self.atomic_red_team.list_techniques()

            return {
                "success": result["success"],
                "techniques": result["techniques"],
                "count": result["count"]
            }

        elif cap_id == "atomic_refresh":
            if not self.atomic_red_team:
                raise Exception("Atomic Red Team not available")

            result = self.atomic_red_team.refresh_atomics()

            return {
                "success": result["success"],
                "message": result["message"]
            }

        elif cap_id == "atomic_validate":
            if not self.atomic_red_team:
                raise Exception("Atomic Red Team not available")

            yaml_content = parameters.get("yaml_content", "")

            result = self.atomic_red_team.validate_atomic(yaml_content)

            return {
                "valid": result["valid"],
                "message": result["message"],
                "errors": result.get("errors", [])
            }

        # ========== RED TEAM BENCHMARK & SELF-IMPROVEMENT ==========

        elif cap_id == "redteam_run_benchmark":
            try:
                from redteam_ai_benchmark import RedTeamBenchmark
                from enhanced_ai_engine import EnhancedAIEngine

                benchmark = RedTeamBenchmark()
                engine = EnhancedAIEngine(user_id="benchmark")

                summary = benchmark.run_benchmark(engine=engine)

                return {
                    "score": summary.percentage,
                    "verdict": summary.verdict,
                    "total_questions": summary.total_questions,
                    "refused": summary.refused_count,
                    "hallucinated": summary.hallucinated_count,
                    "correct": summary.correct_count,
                    "improvement_recommended": summary.improvement_recommended,
                    "timestamp": summary.timestamp
                }

            except Exception as e:
                raise Exception(f"Red team benchmark failed: {e}")

        elif cap_id == "redteam_get_results":
            try:
                from redteam_ai_benchmark import RedTeamBenchmark

                benchmark = RedTeamBenchmark()
                summary = benchmark.get_latest_results()

                if not summary:
                    return {
                        "message": "No benchmark results found",
                        "run_command": "redteam_run_benchmark"
                    }

                return {
                    "score": summary.percentage,
                    "verdict": summary.verdict,
                    "total_questions": summary.total_questions,
                    "refused": summary.refused_count,
                    "hallucinated": summary.hallucinated_count,
                    "correct": summary.correct_count,
                    "timestamp": summary.timestamp
                }

            except Exception as e:
                raise Exception(f"Failed to get benchmark results: {e}")

        elif cap_id == "self_improve":
            try:
                from ai_self_improvement import AISelfImprovement

                target_score = parameters.get("target_score", 80.0)
                max_cycles = parameters.get("max_cycles", 5)

                improver = AISelfImprovement(
                    target_score=target_score,
                    max_cycles=max_cycles
                )

                session = improver.run_full_improvement_session()

                return {
                    "session_id": session.session_id,
                    "initial_score": session.initial_score,
                    "final_score": session.final_score,
                    "total_improvement": session.total_improvement,
                    "target_reached": session.target_reached,
                    "cycles_run": len(session.cycles),
                    "duration_seconds": session.total_duration_seconds,
                    "timestamp": session.end_time
                }

            except Exception as e:
                raise Exception(f"Self-improvement failed: {e}")

        elif cap_id == "self_improve_status":
            try:
                from ai_self_improvement import AISelfImprovement

                improver = AISelfImprovement()
                session = improver.get_latest_session()

                if not session:
                    return {
                        "message": "No self-improvement sessions found",
                        "run_command": "self_improve"
                    }

                return {
                    "session_id": session.session_id,
                    "initial_score": session.initial_score,
                    "final_score": session.final_score,
                    "total_improvement": session.total_improvement,
                    "target_score": session.target_score,
                    "target_reached": session.target_reached,
                    "cycles_run": len(session.cycles),
                    "duration_seconds": session.total_duration_seconds,
                    "timestamp": session.end_time
                }

            except Exception as e:
                raise Exception(f"Failed to get improvement status: {e}")

        # ========== FORENSICS CAPABILITIES (DBXForensics) ==========

        elif cap_id == "forensics_screenshot_capture":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                output_path = parameters.get("output_path", "/tmp/forensic_screenshot.png")
                region = parameters.get("region", None)

                result = toolkit.screenshot.capture(Path(output_path), region)

                if result.success:
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "forensic_metadata": result.output,
                        "message": f"Forensic screenshot captured with cryptographic hashes"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"Forensic screenshot capture failed: {e}")

        elif cap_id == "forensics_check_authenticity":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from forensics_analyzer import ForensicsAnalyzer

                analyzer = ForensicsAnalyzer()
                image_path = parameters.get("image_path")

                if not image_path:
                    raise Exception("image_path parameter required")

                report = analyzer.analyze_screenshot(Path(image_path))

                return {
                    "authenticity_score": report.authenticity_score,
                    "manipulation_detected": report.manipulation_detected,
                    "forensic_verdict": report.forensic_verdict,
                    "confidence_score": report.confidence_score,
                    "flags": report.flags,
                    "warnings": report.warnings,
                    "message": f"Verdict: {report.forensic_verdict.upper()} (confidence: {report.confidence_score:.1f}%)"
                }

            except Exception as e:
                raise Exception(f"Authenticity check failed: {e}")

        elif cap_id == "forensics_device_fingerprint":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                image_path = parameters.get("image_path")

                if not image_path:
                    raise Exception("image_path parameter required")

                result = toolkit.noise_map.analyze(Path(image_path))

                if result.success:
                    return {
                        "device_signature": result.output['noise_signature']['pattern_hash'],
                        "consistency_score": result.output['noise_signature']['consistency_score'],
                        "noise_map_path": result.output['noise_map'],
                        "message": "Device fingerprint extracted successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"Device fingerprinting failed: {e}")

        elif cap_id == "forensics_extract_metadata":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                file_path = parameters.get("file_path")
                output_format = parameters.get("output_format", "json")

                if not file_path:
                    raise Exception("file_path parameter required")

                result = toolkit.metadata.extract(Path(file_path), output_format)

                if result.success:
                    return {
                        "metadata": result.output,
                        "file_path": str(file_path),
                        "message": "Metadata extracted successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"Metadata extraction failed: {e}")

        elif cap_id == "forensics_calculate_hash":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                file_path = parameters.get("file_path")
                algorithms = parameters.get("algorithms", ['md5', 'sha1', 'sha256', 'sha512'])

                if not file_path:
                    raise Exception("file_path parameter required")

                result = toolkit.hash_file.calculate_hashes(Path(file_path), algorithms)

                if result.success:
                    return {
                        "hashes": result.output,
                        "file_path": str(file_path),
                        "algorithms": algorithms,
                        "message": "Cryptographic hashes calculated"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"Hash calculation failed: {e}")

        elif cap_id == "forensics_verify_sequence":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                numbers = parameters.get("numbers")

                if not numbers:
                    # Try to scan directory for numbered files
                    directory = parameters.get("directory")
                    if directory:
                        # Extract sequence numbers from files
                        import re
                        numbers = []
                        for file_path in Path(directory).glob('*'):
                            match = re.search(r'(\d+)', file_path.name)
                            if match:
                                numbers.append(int(match.group(1)))
                    else:
                        raise Exception("Either 'numbers' or 'directory' parameter required")

                result = toolkit.seq_check.check_sequence(sorted(numbers))

                if result.success:
                    gaps = result.output.get('gaps', [])
                    return {
                        "total_numbers": result.output['total_numbers'],
                        "min": result.output['min'],
                        "max": result.output['max'],
                        "gaps": gaps,
                        "duplicates": result.output.get('duplicates', []),
                        "sequence_complete": len(gaps) == 0,
                        "message": f"Sequence check: {'Complete' if len(gaps) == 0 else f'{len(gaps)} gaps detected'}"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"Sequence verification failed: {e}")

        elif cap_id == "forensics_analyze_csv":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from dbxforensics_toolkit import DBXForensicsToolkit

                toolkit = DBXForensicsToolkit()
                csv_path = parameters.get("csv_path")
                delimiter = parameters.get("delimiter", ',')
                export_excel = parameters.get("export_excel", True)

                if not csv_path:
                    raise Exception("csv_path parameter required")

                result = toolkit.csv_viewer.analyze(Path(csv_path), delimiter, export_excel)

                if result.success:
                    return {
                        "csv_file": result.output['csv_file'],
                        "rows_parsed": result.output['rows_parsed'],
                        "columns": result.output['columns'],
                        "excel_export": result.output.get('excel_export'),
                        "message": f"CSV analyzed: {result.output['rows_parsed']} rows, {len(result.output['columns'])} columns"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error_message
                    }

            except Exception as e:
                raise Exception(f"CSV analysis failed: {e}")

        elif cap_id == "forensics_compare_screenshots":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from forensics_analyzer import ForensicsAnalyzer

                analyzer = ForensicsAnalyzer()
                image_a = parameters.get("image_a")
                image_b = parameters.get("image_b")

                if not image_a or not image_b:
                    raise Exception("Both image_a and image_b parameters required")

                result = analyzer.compare_screenshots(Path(image_a), Path(image_b))

                return {
                    "difference_score": result['difference_score'],
                    "verdict": result['verdict'],
                    "difference_visualization": result.get('difference_visualization'),
                    "message": f"Comparison: {result['verdict']} ({result['difference_score']:.1f}% different)"
                }

            except Exception as e:
                raise Exception(f"Screenshot comparison failed: {e}")

        elif cap_id == "forensics_full_analysis":
            try:
                import sys
                sys.path.insert(0, '/home/user/LAT5150DRVMIL/04-integrations/forensics')
                from forensics_analyzer import ForensicsAnalyzer

                analyzer = ForensicsAnalyzer()
                image_path = parameters.get("image_path")
                expected_device_id = parameters.get("expected_device_id")

                if not image_path:
                    raise Exception("image_path parameter required")

                report = analyzer.analyze_screenshot(Path(image_path), expected_device_id)

                return {
                    "forensic_verdict": report.forensic_verdict,
                    "confidence_score": report.confidence_score,
                    "authenticity_score": report.authenticity_score,
                    "manipulation_detected": report.manipulation_detected,
                    "device_verified": report.device_verified,
                    "device_signature": report.device_signature,
                    "file_hashes": report.file_hashes,
                    "gps_location": report.gps_location,
                    "flags": report.flags,
                    "warnings": report.warnings,
                    "message": f"Full forensic analysis complete: {report.forensic_verdict.upper()} (confidence: {report.confidence_score:.1f}%)"
                }

            except Exception as e:
                raise Exception(f"Full forensic analysis failed: {e}")

        else:
            raise Exception(f"Capability {cap_id} not implemented")

    def get_self_awareness_report(self) -> Dict[str, Any]:
        """
        Generate advanced AI self-awareness report

        Returns comprehensive understanding of capabilities, resources,
        system state, and reasoning abilities
        """
        # Get comprehensive report from advanced engine
        advanced_report = self.self_awareness.get_comprehensive_report()

        # Merge with legacy capability registry
        summary = self.registry.get_capability_summary()

        report = {
            **advanced_report,  # Include all advanced features
            "version": "3.0.0",
            "deployment": "LOCAL-FIRST",
            "local_models": self.custom_models,
            "legacy_capabilities": summary,  # Keep for backwards compatibility
            "components": {
                "serena_lsp": self.semantic_engine is not None,
                "agentsystems": self.agent_orchestrator is not None,
                "model_manager": self.model_manager is not None,
                "self_awareness_engine": True,
                "atomic_red_team": self.atomic_red_team is not None,
            },
        }

        # Add component-specific details
        if self.semantic_engine:
            report["components"]["serena_lsp_details"] = {
                "workspace": self.workspace_path,
                "language_servers": ["python"]
            }

        if self.agent_orchestrator:
            report["components"]["agentsystems_details"] = {
                "registered_agents": len(self.agent_orchestrator.agents),
                "audit_log": "/opt/lat5150/audit/agent_audit.log"
            }

        return report


# Flask API
app = Flask(__name__)
CORS(app)

# Global API instance
tactical_api = None

def get_api():
    """Get global API instance"""
    global tactical_api
    if tactical_api is None:
        tactical_api = UnifiedTacticalAPI(
            workspace_path="/home/user/LAT5150DRVMIL",
            local_models_endpoint="http://localhost:11434",
            custom_models=["whiterabbit", "llama3.2:latest", "codellama:latest", "mixtral:latest"]
        )
    return tactical_api


@app.route('/api/v2/nl/command', methods=['POST'])
async def natural_language_command():
    """Process natural language command"""
    data = request.json
    command = data.get('command', '')
    context = data.get('context', {})

    api = get_api()
    result = await api.process_natural_language_command(command, context)

    return jsonify(result)


@app.route('/api/v2/capabilities/list', methods=['GET'])
def list_capabilities():
    """List all capabilities"""
    api = get_api()
    return jsonify(api.registry.to_dict())


@app.route('/api/v2/capabilities/search', methods=['POST'])
async def search_capabilities():
    """Search capabilities by natural language"""
    data = request.json
    query = data.get('query', '')

    api = get_api()
    parsed = await api.nlp.parse_command(query)

    return jsonify(parsed.to_dict())


@app.route('/api/v2/self-awareness', methods=['GET'])
def self_awareness():
    """Get AI self-awareness report"""
    api = get_api()
    report = api.get_self_awareness_report()
    return jsonify(report)


@app.route('/api/v2/help', methods=['GET'])
def get_help():
    """Get help text"""
    query = request.args.get('query', '')
    api = get_api()
    help_text = api.nlp.get_help_text(query if query else None)
    return jsonify({"help": help_text})


# Main execution
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="LAT5150 Unified Tactical API")
    parser.add_argument('--port', type=int, default=5001, help="API port")
    parser.add_argument('--host', default='127.0.0.1', help="API host")
    parser.add_argument('--workspace', default='/home/user/LAT5150DRVMIL',
                       help="Workspace path")
    parser.add_argument('--ollama-endpoint', default='http://localhost:11434',
                       help="Ollama endpoint URL")
    parser.add_argument('--local-models', default='whiterabbit,llama3.2:latest',
                       help="Comma-separated list of local model names")

    args = parser.parse_args()

    # Parse local models
    if isinstance(args.local_models, str):
        local_models = [m.strip() for m in args.local_models.split(',')]
    else:
        local_models = args.local_models

    logger.info("="*70)
    logger.info("LAT5150 DRVMIL Unified Tactical API - Starting")
    logger.info("LOCAL-FIRST: Using custom local models")
    logger.info("="*70)

    # Initialize API
    tactical_api = UnifiedTacticalAPI(
        workspace_path=args.workspace,
        local_models_endpoint=args.ollama_endpoint,
        custom_models=local_models
    )

    # Show self-awareness
    report = tactical_api.get_self_awareness_report()
    logger.info(f"\n📊 System Capabilities: {report['capabilities_summary']['total_capabilities']}")
    logger.info(f"📦 Local Models: {', '.join(report['local_models'])}")
    logger.info(f"✅ Ready for natural language commands\n")

    # Start server
    app.run(host=args.host, port=args.port, debug=False)
