#!/usr/bin/env python3
"""
DSMIL AI - Graphical User Interface Dashboard

Web-based dashboard for AI system management and monitoring.

Features:
- CLI command launcher (query, reason, benchmark, etc.)
- Real-time benchmark visualization
- Test runner with progress tracking
- Script launcher (setup, MCP, etc.)
- System status monitoring
- Hephaestus workflow management
- MCP server status
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_system_integrator import AISystemIntegrator
    from security_hardening import SecurityHardening
    from hephaestus_integration import HephaestusIntegrator
    from local_agent_loader import LocalAgentLoader
    from comprehensive_98_agent_system import create_98_agent_coordinator
    from ramdisk_database import RAMDiskDatabase
    from dsmil_subsystem_controller import DSMILSubsystemController, SubsystemType
    from tpm_crypto_integration import TPMCryptoIntegration
    from ai_benchmarking import EnhancedAIBenchmark
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

# Import Heretic integration modules
try:
    from heretic_hook import register_heretic_hooks
    from hook_system import HookManager, create_default_hooks
    from heretic_web_api import register_heretic_routes
    HERETIC_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Heretic integration not available: {e}")
    HERETIC_INTEGRATION_AVAILABLE = False

# Import Atomic Red Team API
try:
    from atomic_red_team_api import AtomicRedTeamAPI
    ATOMIC_RED_TEAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Atomic Red Team API not available: {e}")
    ATOMIC_RED_TEAM_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Global state
integrator = None
security = None
hephaestus = None
agent_loader = None
agent_coordinator = None
database = None
dsmil_controller = None
tpm_crypto = None
benchmark = None
hook_manager = None
atomic_red_team = None
voice_ui_active = False
active_processes = {}
benchmark_results = []
current_session_id = None


def initialize_components():
    """Initialize AI components"""
    global integrator, security, hephaestus, agent_loader, agent_coordinator, database, current_session_id, dsmil_controller, tpm_crypto, benchmark, hook_manager, atomic_red_team

    try:
        integrator = AISystemIntegrator(
            enable_engine=True,
            enable_reasoning=True,
            enable_benchmarking=True
        )
    except Exception as e:
        print(f"Warning: Could not initialize integrator: {e}")

    try:
        security = SecurityHardening()
    except Exception as e:
        print(f"Warning: Could not initialize security: {e}")

    try:
        hephaestus = HephaestusIntegrator()
    except Exception as e:
        print(f"Warning: Could not initialize Hephaestus: {e}")

    try:
        agent_loader = LocalAgentLoader()
        print("✓ Agent loader initialized")
    except Exception as e:
        print(f"Warning: Could not initialize agent loader: {e}")

    try:
        agent_coordinator = create_98_agent_coordinator()
        print("✓ 98-agent coordinator initialized")
    except Exception as e:
        print(f"Warning: Could not initialize agent coordinator: {e}")

    try:
        # Initialize RAM disk database with auto-sync
        database = RAMDiskDatabase(
            auto_sync=True,
            sync_interval_seconds=60
        )
        current_session_id = f"gui_session_{int(time.time())}"
        print(f"✓ RAM disk database initialized (session: {current_session_id})")
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")

    try:
        # Initialize DSMIL subsystem controller
        dsmil_controller = DSMILSubsystemController()
        print("✓ DSMIL subsystem controller initialized")
    except Exception as e:
        print(f"Warning: Could not initialize DSMIL controller: {e}")

    try:
        # Initialize TPM cryptography (hardware-backed security)
        tpm_crypto = TPMCryptoIntegration()
        if tpm_crypto.tpm_available:
            print(f"✓ TPM 2.0 available - {tpm_crypto.get_algorithm_count()} algorithms detected")
        else:
            print("ℹ TPM not available (expected in Docker, available on Dell MIL-SPEC)")
    except Exception as e:
        print(f"Warning: Could not initialize TPM crypto: {e}")

    try:
        # Initialize benchmark framework
        benchmark = EnhancedAIBenchmark()
        print(f"✓ Benchmark framework initialized - {len(benchmark.tasks)} test tasks loaded")
    except Exception as e:
        print(f"Warning: Could not initialize benchmark framework: {e}")

    # ========================================================================
    # HERETIC INTEGRATION - Abliteration Hook System & Web API
    # ========================================================================
    if HERETIC_INTEGRATION_AVAILABLE:
        try:
            # 1. Initialize hook manager
            hook_manager = create_default_hooks()
            print("✓ Hook system initialized")

            # 2. Register Heretic hooks (automatic monitoring & refusal detection)
            register_heretic_hooks(
                hook_manager,
                enable_refusal_detection=True,      # Detect refusals in responses
                enable_safety_monitor=True,         # Track refusal rates over time
                enable_abliteration_trigger=True,   # Trigger abliteration aggressively
                auto_abliterate=True,               # Automatic abliteration enabled
                refusal_threshold=0.05              # Trigger at 5% refusal rate (highly aggressive)
            )

            # 3. Register Heretic Web API routes (manual control via dashboard)
            register_heretic_routes(app)

            print("✓ Heretic fully integrated (hooks + web API)")
            print("  - Automatic refusal detection enabled")
            print("  - Automatic abliteration enabled (threshold: 5% - HIGHLY AGGRESSIVE)")
            print("  - Web interface available at /api/heretic/*")

        except Exception as e:
            print(f"Warning: Could not initialize Heretic integration: {e}")

    # ========================================================================
    # ATOMIC RED TEAM INTEGRATION - Security Testing Framework
    # ========================================================================
    if ATOMIC_RED_TEAM_AVAILABLE:
        try:
            atomic_red_team = AtomicRedTeamAPI()
            print("✓ Atomic Red Team API initialized")
            print("  - Query MITRE ATT&CK techniques")
            print("  - Validate test YAML structure")
            print("  - Test execution DISABLED (safety)")

        except Exception as e:
            print(f"Warning: Could not initialize Atomic Red Team: {e}")


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get system status"""

    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "integrator": integrator is not None,
            "security": security is not None,
            "hephaestus": hephaestus is not None
        },
        "services": {},
        "stats": {}
    }

    # Check Ollama
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/tags'],
            capture_output=True,
            timeout=2
        )
        status["services"]["ollama"] = result.returncode == 0
    except:
        status["services"]["ollama"] = False

    # Check vLLM
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:8000/health'],
            capture_output=True,
            timeout=2
        )
        status["services"]["vllm"] = result.returncode == 0
    except:
        status["services"]["vllm"] = False

    # Get stats from integrator
    if integrator:
        try:
            status["stats"] = integrator.get_stats()
        except:
            pass

    # Get Hephaestus stats
    if hephaestus:
        try:
            status["hephaestus_stats"] = hephaestus.get_statistics()
        except:
            pass

    return jsonify(status)


@app.route('/api/query', methods=['POST'])
def query():
    """Execute AI query"""

    if not integrator:
        return jsonify({"error": "AI System not initialized"}), 500

    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'uncensored_code')
    mode = data.get('mode', 'auto')

    # Security check
    if security:
        valid, reason = security.validate_input(prompt)
        if not valid:
            return jsonify({"error": f"Security check failed: {reason}"}), 400

    try:
        # Store user message in database
        if database and current_session_id:
            database.store_message(
                session_id=current_session_id,
                role="user",
                content=prompt,
                model=model,
                latency_ms=0,
                hardware_backend="GUI"
            )

        response = integrator.query(
            prompt=prompt,
            model=model,
            mode=mode
        )

        # Store assistant response in database
        if database and current_session_id:
            database.store_message(
                session_id=current_session_id,
                role="assistant",
                content=response.content,
                model=response.model,
                latency_ms=response.latency_ms,
                hardware_backend=getattr(response, 'hardware_backend', 'CPU')
            )

        return jsonify({
            "content": response.content,
            "latency_ms": response.latency_ms,
            "mode": response.mode,
            "model": response.model,
            "cached": response.cached,
            "reasoning_steps": response.reasoning_steps,
            "success": response.success
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run benchmarks"""

    if not integrator:
        return jsonify({"error": "AI System not initialized"}), 500

    data = request.json
    task_ids = data.get('task_ids')
    num_runs = data.get('num_runs', 3)

    # Run in background thread
    def run_async():
        global benchmark_results
        try:
            results = integrator.benchmark(
                task_ids=task_ids,
                num_runs=num_runs
            )
            benchmark_results.append({
                "timestamp": datetime.now().isoformat(),
                "results": results
            })
        except Exception as e:
            benchmark_results.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

    thread = threading.Thread(target=run_async)
    thread.start()

    return jsonify({"status": "started", "message": "Benchmark running in background"})


@app.route('/api/benchmark/results')
def get_benchmark_results():
    """Get benchmark results"""
    return jsonify(benchmark_results)


@app.route('/api/security/report')
def security_report():
    """Get security report"""

    if not security:
        return jsonify({"error": "Security system not initialized"}), 500

    try:
        report = security.get_security_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/scripts')
def list_scripts():
    """List available scripts"""

    scripts_dir = Path(__file__).parent
    scripts = []

    # Find all shell scripts
    for script_path in scripts_dir.glob("*.sh"):
        scripts.append({
            "name": script_path.name,
            "path": str(script_path),
            "size": script_path.stat().st_size,
            "modified": datetime.fromtimestamp(script_path.stat().st_mtime).isoformat()
        })

    return jsonify(scripts)


@app.route('/api/scripts/run', methods=['POST'])
def run_script():
    """Run a script"""

    data = request.json
    script_path = data.get('script_path', '')
    args = data.get('args', [])

    # Security check - only allow scripts in 02-ai-engine directory
    scripts_dir = Path(__file__).parent
    script_path = Path(script_path)

    if not script_path.is_relative_to(scripts_dir):
        return jsonify({"error": "Script must be in AI engine directory"}), 400

    if not script_path.exists():
        return jsonify({"error": "Script not found"}), 404

    # Run in background
    process_id = f"script_{int(time.time())}"

    def run_async():
        global active_processes
        try:
            result = subprocess.run(
                [str(script_path)] + args,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            active_processes[process_id] = {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "completed_at": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            active_processes[process_id] = {
                "status": "timeout",
                "error": "Script execution timed out (5 minutes)"
            }
        except Exception as e:
            active_processes[process_id] = {
                "status": "error",
                "error": str(e)
            }

    active_processes[process_id] = {
        "status": "running",
        "script": script_path.name,
        "started_at": datetime.now().isoformat()
    }

    thread = threading.Thread(target=run_async)
    thread.start()

    return jsonify({
        "process_id": process_id,
        "status": "started"
    })


@app.route('/api/scripts/status/<process_id>')
def script_status(process_id):
    """Get script execution status"""

    if process_id not in active_processes:
        return jsonify({"error": "Process not found"}), 404

    return jsonify(active_processes[process_id])


@app.route('/api/hephaestus/workflows')
def list_workflows():
    """List Hephaestus workflows"""

    if not hephaestus:
        return jsonify({"error": "Hephaestus not initialized"}), 500

    workflows = []
    for wf_id, workflow in hephaestus.workflows.items():
        status = hephaestus.get_workflow_status(wf_id)
        workflows.append(status)

    return jsonify(workflows)


@app.route('/api/hephaestus/workflow/create', methods=['POST'])
def create_workflow():
    """Create new Hephaestus workflow"""

    if not hephaestus:
        return jsonify({"error": "Hephaestus not initialized"}), 500

    data = request.json
    project_name = data.get('project_name', 'unnamed')
    goal = data.get('goal', '')
    initial_tasks = data.get('initial_tasks', [])

    try:
        workflow_id = hephaestus.create_workflow(
            project_name=project_name,
            goal=goal,
            initial_tasks=initial_tasks
        )

        return jsonify({
            "workflow_id": workflow_id,
            "status": "created"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/hephaestus/workflow/<workflow_id>/execute', methods=['POST'])
def execute_workflow(workflow_id):
    """Execute Hephaestus workflow"""

    if not hephaestus:
        return jsonify({"error": "Hephaestus not initialized"}), 500

    # Run in background
    def run_async():
        try:
            hephaestus.execute_workflow(workflow_id, max_iterations=100)
        except Exception as e:
            print(f"Workflow execution error: {e}")

    thread = threading.Thread(target=run_async)
    thread.start()

    return jsonify({
        "status": "started",
        "workflow_id": workflow_id
    })


@app.route('/api/mcp/status')
def mcp_status():
    """Get MCP server status"""

    config_path = Path.home() / ".config" / "mcp_servers_config.json"

    if not config_path.exists():
        return jsonify({
            "configured": False,
            "servers": []
        })

    try:
        with open(config_path) as f:
            config = json.load(f)

        servers = []
        for name, server_config in config.get("mcpServers", {}).items():
            servers.append({
                "name": name,
                "command": server_config.get("command", ""),
                "configured": True
            })

        return jsonify({
            "configured": True,
            "total_servers": len(servers),
            "servers": servers
        })

    except Exception as e:
        return jsonify({
            "configured": False,
            "error": str(e)
        })


@app.route('/api/agents/stats')
def agents_stats():
    """Get agent system statistics"""
    if not agent_loader:
        return jsonify({"error": "Agent loader not initialized"}), 500

    try:
        stats = agent_loader.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/agents/list')
def agents_list():
    """List all loaded agents"""
    if not agent_loader:
        return jsonify({"error": "Agent loader not initialized"}), 500

    try:
        agents = []
        for agent_id, agent in agent_loader.agents.items():
            agents.append({
                "id": agent.id,
                "name": agent.name,
                "category": agent.category.value,
                "hardware": agent.preferred_hardware,
                "model": agent.local_model,
                "capabilities_count": len(agent.capabilities)
            })

        return jsonify({
            "total": len(agents),
            "agents": agents
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/history')
def conversation_history():
    """Get conversation history for current session"""
    if not database:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        limit = request.args.get('limit', 100, type=int)
        session_id = request.args.get('session_id', current_session_id)

        messages = database.get_conversation_history(session_id, limit=limit)

        return jsonify({
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "timestamp": msg.timestamp.isoformat(),
                    "role": msg.role,
                    "content": msg.content,
                    "model": msg.model,
                    "latency_ms": msg.latency_ms,
                    "hardware_backend": msg.hardware_backend
                }
                for msg in messages
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/recent')
def conversation_recent():
    """Get recent messages across all sessions"""
    if not database:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        limit = request.args.get('limit', 50, type=int)
        messages = database.get_recent_messages(limit=limit)

        return jsonify({
            "message_count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "session_id": msg.session_id,
                    "timestamp": msg.timestamp.isoformat(),
                    "role": msg.role,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    "model": msg.model,
                    "latency_ms": msg.latency_ms,
                    "hardware_backend": msg.hardware_backend
                }
                for msg in messages
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/stats')
def conversation_stats():
    """Get database statistics"""
    if not database:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        stats = database.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/sync', methods=['POST'])
def conversation_sync():
    """Manually sync database to backup"""
    if not database:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        success = database.sync_to_backup()
        if success:
            return jsonify({
                "success": True,
                "message": "Database synced to backup"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Sync failed"
            }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/voice/toggle', methods=['POST'])
def voice_toggle():
    """Toggle voice UI"""
    global voice_ui_active

    data = request.json
    enable = data.get('enable', not voice_ui_active)

    try:
        if enable and not voice_ui_active:
            # Start voice UI in background thread
            def run_voice_ui():
                try:
                    from voice_ui_npu import VoiceUI
                    voice_ui = VoiceUI(ai_system=integrator, enable_wake_word=True)
                    voice_ui.text_mode()  # Text mode for GUI integration
                except Exception as e:
                    print(f"Voice UI error: {e}")

            voice_thread = threading.Thread(target=run_voice_ui, daemon=True)
            voice_thread.start()
            voice_ui_active = True

            return jsonify({
                "success": True,
                "status": "active",
                "message": "Voice UI started in text mode"
            })

        elif not enable and voice_ui_active:
            voice_ui_active = False
            return jsonify({
                "success": True,
                "status": "inactive",
                "message": "Voice UI stopped"
            })

        return jsonify({
            "success": True,
            "status": "active" if voice_ui_active else "inactive"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/voice/query', methods=['POST'])
def voice_query():
    """Process voice input as query"""
    data = request.json
    voice_text = data.get('text', '')

    if not voice_text:
        return jsonify({"error": "No voice text provided"}), 400

    # Route to regular query endpoint
    if not integrator:
        return jsonify({"error": "AI System not initialized"}), 500

    try:
        response = integrator.query(
            prompt=voice_text,
            model="uncensored_code",
            mode="auto"
        )

        return jsonify({
            "content": response.content,
            "latency_ms": response.latency_ms,
            "mode": response.mode,
            "success": True,
            "voice_input": voice_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# DSMIL SUBSYSTEM CONTROL ENDPOINTS
# ============================================================================

@app.route('/api/dsmil/health')
def dsmil_health():
    """Get DSMIL system health"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        health = dsmil_controller.get_system_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/subsystems')
def dsmil_subsystems():
    """Get all subsystems status"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        subsystems = dsmil_controller.get_all_subsystems_status()
        # Convert SubsystemStatus objects to dictionaries
        subsystems_dict = {}
        for subsystem_type, status in subsystems.items():
            subsystems_dict[subsystem_type.value] = {
                "available": status.available,
                "operational": status.operational,
                "status_message": status.status_message,
                "last_check": status.last_check.isoformat() if status.last_check else None
            }
        return jsonify(subsystems_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/devices/safe')
def dsmil_devices_safe():
    """List safe devices"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        devices = dsmil_controller.list_safe_devices()
        return jsonify({
            "count": len(devices),
            "devices": [
                {
                    "id": f"0x{d.device_id:04X}",
                    "name": d.name,
                    "description": d.description,
                    "status": d.status.value,
                    "safe": d.safe_to_activate
                }
                for d in devices
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/devices/quarantined')
def dsmil_devices_quarantined():
    """List quarantined devices (READ-ONLY)"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        devices = dsmil_controller.list_quarantined_devices()
        return jsonify({
            "count": len(devices),
            "warning": "These devices are QUARANTINED and cannot be activated",
            "devices": [
                {
                    "id": f"0x{d.device_id:04X}",
                    "name": d.name,
                    "description": d.description,
                    "reason": "SAFETY - Destructive capability"
                }
                for d in devices
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/device/activate', methods=['POST'])
def dsmil_device_activate():
    """
    Activate a DSMIL device (SAFETY ENFORCED)

    POST data:
    {
        "device_id": "0x8003",  # Hex string
        "value": 1
    }
    """
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    data = request.json

    try:
        # Parse device ID (hex string)
        device_id_str = data.get('device_id', '').replace('0x', '')
        device_id = int(device_id_str, 16)
        value = int(data.get('value', 0))

        # Attempt activation (safety-enforced in controller)
        success, message = dsmil_controller.activate_device(device_id, value)

        if success:
            return jsonify({
                "success": True,
                "message": message,
                "device_id": f"0x{device_id:04X}",
                "value": value
            })
        else:
            return jsonify({
                "success": False,
                "error": message,
                "device_id": f"0x{device_id:04X}"
            }), 403  # Forbidden

    except ValueError as e:
        return jsonify({"error": f"Invalid device ID or value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/tpm/quote')
def dsmil_tpm_quote():
    """Get TPM 2.0 quote for attestation"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        quote = dsmil_controller.get_tpm_quote()

        if quote:
            return jsonify({
                "available": True,
                "quote": quote,
                "timestamp": time.time()
            })
        else:
            return jsonify({
                "available": False,
                "message": "TPM attestation not available"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/metrics')
def dsmil_metrics():
    """Get comprehensive DSMIL metrics"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        metrics = dsmil_controller.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# EASY WINS API ENDPOINTS
# ============================================================================

@app.route('/api/dsmil/thermal-enhanced')
def thermal_enhanced():
    """Easy Win #1: Enhanced thermal monitoring with per-core readings"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        thermal = dsmil_controller.get_thermal_status_enhanced()
        return jsonify({"success": True, "thermal": thermal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/tpm-pcr-state')
def tpm_pcr_state():
    """Easy Win #2: TPM PCR state tracking"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        pcr_state = dsmil_controller.get_tpm_pcr_state()
        event_log = dsmil_controller.get_tpm_event_log()
        return jsonify({"success": True, "pcr_state": pcr_state, "event_log": event_log})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/device-status-cached/<device_id>')
def device_status_cached(device_id):
    """Easy Win #3: Device status with caching"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        device_id_int = int(device_id, 16) if device_id.startswith('0x') else int(device_id)
        status = dsmil_controller.get_device_status_cached(device_id_int)
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/operation-history')
def operation_history():
    """Easy Win #4: Operation history logging"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        device_id = request.args.get('device_id')
        limit = int(request.args.get('limit', 100))
        operation_type = request.args.get('operation_type')

        if device_id:
            device_id = int(device_id, 16) if device_id.startswith('0x') else int(device_id)

        history = dsmil_controller.get_operation_history(
            device_id=device_id, limit=limit, operation_type=operation_type
        )
        return jsonify({"success": True, "history": history, "count": len(history)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/operation-stats')
def operation_stats():
    """Easy Win #4: Operation statistics"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        stats = dsmil_controller.get_operation_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dsmil/health-score')
def health_score():
    """Easy Win #5: Subsystem health scores"""
    if not dsmil_controller:
        return jsonify({"error": "DSMIL controller not initialized"}), 500

    try:
        health_scores = dsmil_controller.get_subsystem_health_score()
        return jsonify({"success": True, "health_scores": health_scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/events')
def audit_events():
    """Get audit events with filtering"""
    if not dsmil_controller or not dsmil_controller.audit_storage:
        return jsonify({"error": "Audit storage not initialized"}), 500

    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        device_id = request.args.get('device_id')
        if device_id:
            device_id = int(device_id, 16) if device_id.startswith('0x') else int(device_id)
        events = dsmil_controller.audit_storage.get_events(limit=limit, offset=offset, device_id=device_id)
        return jsonify({"success": True, "events": events, "count": len(events)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/statistics')
def audit_statistics():
    """Get audit statistics summary"""
    if not dsmil_controller or not dsmil_controller.audit_storage:
        return jsonify({"error": "Audit storage not initialized"}), 500
    try:
        stats = dsmil_controller.audit_storage.get_statistics()
        return jsonify({"success": True, "statistics": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/database-info')
def audit_database_info():
    """Get audit database information"""
    if not dsmil_controller or not dsmil_controller.audit_storage:
        return jsonify({"error": "Audit storage not initialized"}), 500
    try:
        info = dsmil_controller.audit_storage.get_database_size()
        return jsonify({"success": True, "database_info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tpm/status')
def tpm_status():
    """Get TPM 2.0 status and capabilities (formerly audit_tpm_capabilities.py)"""
    try:
        if not tpm_crypto:
            return jsonify({
                "available": False,
                "message": "TPM not available (expected in Docker)",
                "expected_on_hardware": {
                    "manufacturer": "STMicroelectronics or Infineon",
                    "algorithms": "88+",
                    "features": ["Hardware RNG", "AES Acceleration", "Attestation", "NVRAM Storage"]
                }
            })

        status = {
            "available": tpm_crypto.tpm_available,
            "algorithm_count": tpm_crypto.get_algorithm_count(),
            "statistics": tpm_crypto.get_statistics()
        }

        if tpm_crypto.tpm_available:
            # Get detailed capabilities
            try:
                result = subprocess.run(
                    ["tpm2_getcap", "properties-fixed"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    status["detailed_capabilities"] = result.stdout
            except:
                pass

        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/benchmark/run', methods=['POST'])
def run_benchmark():
    """Run benchmark suite (formerly test_dsmil_api.py + test_integration.py + ai_benchmarking.py)"""
    if not benchmark:
        return jsonify({"error": "Benchmark framework not initialized"}), 500

    try:
        data = request.json or {}
        task_ids = data.get('task_ids')  # None = all tasks
        num_runs = data.get('num_runs', 1)  # Default 1 run for quick tests
        models = data.get('models')  # None = default model

        # Run in background thread to avoid blocking
        def run_benchmark_async():
            global benchmark_results
            try:
                summary = benchmark.run_benchmark(
                    task_ids=task_ids,
                    num_runs=num_runs,
                    models=models
                )
                benchmark_results.append({
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary
                })
            except Exception as e:
                print(f"Benchmark error: {e}")

        thread = threading.Thread(target=run_benchmark_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "started",
            "message": "Benchmark running in background",
            "task_count": len(benchmark.tasks) if not task_ids else len(task_ids),
            "estimated_time_seconds": (len(benchmark.tasks) if not task_ids else len(task_ids)) * num_runs * 2
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/benchmark/results')
def get_benchmark_results():
    """Get benchmark results"""
    try:
        if not benchmark_results:
            return jsonify({
                "message": "No benchmark results available",
                "results": []
            })

        # Return latest result
        latest = benchmark_results[-1]
        return jsonify({
            "message": "Latest benchmark results",
            "result": {
                "timestamp": latest["timestamp"],
                "total_tasks": latest["summary"].total_tasks,
                "total_runs": latest["summary"].total_runs,
                "avg_latency_ms": latest["summary"].avg_latency_ms,
                "avg_accuracy": latest["summary"].avg_accuracy,
                "goal_completion_rate": latest["summary"].goal_completion_rate,
                "recommendations": latest["summary"].recommendations
            },
            "history_count": len(benchmark_results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/benchmark/tasks')
def get_benchmark_tasks():
    """List available benchmark tasks"""
    if not benchmark:
        return jsonify({"error": "Benchmark framework not initialized"}), 500

    try:
        tasks = []
        for task in benchmark.tasks:
            tasks.append({
                "task_id": task.task_id,
                "category": task.category,
                "description": task.description,
                "difficulty": task.difficulty,
                "max_latency_ms": task.max_latency_ms
            })

        # Group by category
        by_category = {}
        for task in tasks:
            cat = task["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(task)

        return jsonify({
            "total": len(tasks),
            "by_category": by_category,
            "categories": list(by_category.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ATOMIC RED TEAM API ENDPOINTS
# ============================================================================

@app.route('/api/atomic-red-team/status')
def atomic_red_team_status():
    """Get Atomic Red Team API status"""
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        status = atomic_red_team.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/atomic-red-team/query', methods=['POST'])
def atomic_red_team_query():
    """
    Query atomic tests with natural language or filters

    POST data:
    {
        "query": "mshta atomics for windows",
        "technique_id": "T1059.002",  # Optional
        "platform": "windows",         # Optional
        "name_filter": "mshta"         # Optional
    }
    """
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        data = request.json or {}
        query = data.get('query')
        technique_id = data.get('technique_id')
        platform = data.get('platform')
        name_filter = data.get('name_filter')

        result = atomic_red_team.query_atomics(
            query=query,
            technique_id=technique_id,
            platform=platform,
            name_filter=name_filter
        )

        return jsonify({
            "success": result.success,
            "tests": result.tests,
            "count": result.count,
            "query": result.query,
            "timestamp": result.timestamp,
            "error": result.error
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/atomic-red-team/refresh', methods=['POST'])
def atomic_red_team_refresh():
    """Refresh atomic tests from GitHub repository"""
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        result = atomic_red_team.refresh_atomics()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/atomic-red-team/validate', methods=['POST'])
def atomic_red_team_validate():
    """
    Validate atomic test YAML structure

    POST data:
    {
        "yaml_content": "..."
    }
    """
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        data = request.json or {}
        yaml_content = data.get('yaml_content', '')

        if not yaml_content:
            return jsonify({"error": "yaml_content required"}), 400

        result = atomic_red_team.validate_atomic(yaml_content)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/atomic-red-team/schema')
def atomic_red_team_schema():
    """Get atomic test validation schema"""
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        result = atomic_red_team.get_validation_schema()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/atomic-red-team/techniques')
def atomic_red_team_techniques():
    """List all available MITRE ATT&CK techniques"""
    if not atomic_red_team:
        return jsonify({"error": "Atomic Red Team not initialized"}), 500

    try:
        result = atomic_red_team.list_techniques()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


def create_dashboard_html():
    """Create dashboard HTML template"""

    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSMIL AI TACTICAL CONTROL SYSTEM</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #0f0;
            min-height: 100vh;
            padding: 10px;
            line-height: 1.5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(180deg, #001100 0%, #003300 100%);
            border: 2px solid #0f0;
            padding: 15px 25px;
            margin-bottom: 15px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }
        header h1 {
            color: #0f0;
            font-size: 24px;
            text-shadow: 0 0 10px #0f0;
            letter-spacing: 3px;
            font-weight: bold;
        }
        .status-bar {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .status-item {
            background: #002200;
            padding: 8px 15px;
            border: 1px solid #0f0;
            font-size: 12px;
            font-weight: bold;
            text-shadow: 0 0 5px #0f0;
        }
        .status-item.active {
            background: #003300;
            color: #0f0;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
        }
        .status-item.inactive {
            background: #220000;
            color: #f00;
            border-color: #f00;
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .card {
            background: linear-gradient(180deg, #001100 0%, #000000 100%);
            border: 2px solid #0f0;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
        }
        .card h2 {
            color: #ff0;
            margin-bottom: 15px;
            font-size: 16px;
            border-bottom: 2px solid #0f0;
            padding-bottom: 8px;
            text-shadow: 0 0 8px #ff0;
            letter-spacing: 2px;
            font-weight: bold;
        }
        .form-group {
            margin-bottom: 12px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #0f0;
            font-size: 12px;
            text-shadow: 0 0 5px #0f0;
        }
        input[type="text"], input[type="number"], textarea, select {
            width: 100%;
            padding: 8px 10px;
            border: 2px solid #0f0;
            background: #001100;
            color: #0f0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            transition: all 0.2s;
        }
        input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #ff0;
            box-shadow: 0 0 10px rgba(255, 255, 0, 0.5);
            background: #002200;
        }
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        button {
            background: #003300;
            color: #0f0;
            border: 2px solid #0f0;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            transition: all 0.2s;
            text-shadow: 0 0 5px #0f0;
            letter-spacing: 1px;
        }
        button:hover {
            background: #005500;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
        }
        button:disabled {
            background: #111;
            color: #555;
            border-color: #555;
            cursor: not-allowed;
            text-shadow: none;
            box-shadow: none;
        }
        .response {
            margin-top: 12px;
            padding: 12px;
            background: #001100;
            border: 2px solid #0f0;
            border-left: 4px solid #0f0;
            max-height: 300px;
            overflow-y: auto;
            font-size: 12px;
            box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.1);
        }
        .error {
            background: #110000;
            border-color: #f00;
            color: #f00;
            border-left-color: #f00;
            box-shadow: inset 0 0 10px rgba(255, 0, 0, 0.2);
        }
        .success {
            background: #001100;
            border-color: #0f0;
            color: #0f0;
            box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.2);
        }
        .script-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .script-item {
            padding: 10px;
            margin-bottom: 8px;
            background: #002200;
            border: 1px solid #0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }
        .script-item:hover {
            background: #003300;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
        }
        .script-info {
            flex: 1;
        }
        .script-name {
            font-weight: bold;
            color: #ff0;
            text-shadow: 0 0 5px #ff0;
        }
        .script-meta {
            font-size: 11px;
            color: #0f0;
            margin-top: 4px;
            opacity: 0.8;
        }
        .btn-small {
            padding: 6px 12px;
            font-size: 11px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #0f0;
            opacity: 0.7;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }
        .metric {
            background: linear-gradient(180deg, #003300 0%, #001100 100%);
            color: #0f0;
            border: 2px solid #0f0;
            padding: 15px;
            text-align: center;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #ff0;
            text-shadow: 0 0 10px #ff0;
        }
        .metric-label {
            font-size: 11px;
            opacity: 0.9;
            text-shadow: 0 0 5px #0f0;
            letter-spacing: 1px;
        }
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
        }
        .tab {
            padding: 8px 16px;
            background: #002200;
            border: 2px solid #0f0;
            cursor: pointer;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            color: #0f0;
            font-size: 12px;
            transition: all 0.2s;
        }
        .tab.active {
            background: #003300;
            color: #ff0;
            box-shadow: 0 0 10px rgba(255, 255, 0, 0.5);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        pre {
            background: #000;
            color: #0f0;
            padding: 10px;
            border: 1px solid #0f0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>╔═══ DSMIL AI TACTICAL CONTROL ═══╗</h1>
            <div class="status-bar" id="statusBar">
                <div class="status-item">INITIALIZING...</div>
            </div>
        </header>

        <div class="metrics" id="metrics"></div>

        <div class="grid">
            <!-- CLI Commands -->
            <div class="card">
                <h2>[ AI QUERY TERMINAL ]</h2>
                <div class="form-group">
                    <label>Mode</label>
                    <select id="queryMode">
                        <option value="auto">Auto-detect</option>
                        <option value="simple">Simple</option>
                        <option value="reasoning">Deep Reasoning</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Model</label>
                    <select id="queryModel">
                        <option value="uncensored_code">Uncensored Code (34B)</option>
                        <option value="fast">Fast (1.5B)</option>
                        <option value="quality">Quality (7B)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Prompt</label>
                    <textarea id="queryPrompt" placeholder="ENTER QUERY..."></textarea>
                </div>
                <button onclick="runQuery()">EXECUTE QUERY</button>
                <div id="queryResponse"></div>
            </div>

            <!-- Benchmarks -->
            <div class="card">
                <h2>[ BENCHMARK SYSTEMS ]</h2>
                <div class="form-group">
                    <label>Number of Runs</label>
                    <input type="number" id="benchmarkRuns" value="3" min="1" max="10">
                </div>
                <button onclick="runBenchmark()">RUN BENCHMARK</button>
                <button onclick="getBenchmarkResults()" class="btn-small" style="margin-left: 10px;">VIEW RESULTS</button>
                <div id="benchmarkResponse"></div>
            </div>

            <!-- Security -->
            <div class="card">
                <h2>[ SECURITY SYSTEMS ]</h2>
                <button onclick="getSecurityReport()">GENERATE SECURITY REPORT</button>
                <div id="securityResponse"></div>
            </div>

            <!-- Hephaestus Workflows -->
            <div class="card">
                <h2>[ HEPHAESTUS WORKFLOWS ]</h2>
                <button onclick="listWorkflows()">LIST WORKFLOWS</button>
                <button onclick="showCreateWorkflow()" class="btn-small" style="margin-left: 10px;">CREATE NEW</button>
                <div id="workflowResponse"></div>
            </div>

            <!-- Script Launcher -->
            <div class="card" style="grid-column: 1 / -1;">
                <h2>[ SCRIPT EXECUTION CONTROL ]</h2>
                <button onclick="loadScripts()">REFRESH SCRIPTS</button>
                <div class="script-list" id="scriptList">
                    <div class="loading">CLICK "REFRESH SCRIPTS" TO LOAD AVAILABLE SCRIPTS</div>
                </div>
            </div>

            <!-- MCP Servers -->
            <div class="card">
                <h2>[ MCP SERVER STATUS ]</h2>
                <button onclick="getMCPStatus()">CHECK STATUS</button>
                <div id="mcpResponse"></div>
            </div>

            <!-- DSMIL Easy Wins Monitoring -->
            <div class="card" style="grid-column: 1 / -1;">
                <h2>[ DSMIL EASY WINS MONITORING ]</h2>
                <div class="tabs">
                    <div class="tab active" onclick="switchEasyWinTab('thermal')">THERMAL</div>
                    <div class="tab" onclick="switchEasyWinTab('tpm')">TPM PCR</div>
                    <div class="tab" onclick="switchEasyWinTab('cache')">CACHE</div>
                    <div class="tab" onclick="switchEasyWinTab('history')">HISTORY</div>
                    <div class="tab" onclick="switchEasyWinTab('stats')">STATS</div>
                    <div class="tab" onclick="switchEasyWinTab('health')">HEALTH</div>
                    <div class="tab" onclick="switchEasyWinTab('audit')">AUDIT [IA3]</div>
                </div>
                <div id="easyWinThermal" class="tab-content active">
                    <button onclick="loadThermalEnhanced()">LOAD THERMAL DATA</button>
                    <div id="thermalEnhancedResponse"></div>
                </div>
                <div id="easyWinTPM" class="tab-content">
                    <button onclick="loadTPMPCRState()">LOAD TPM PCR STATE</button>
                    <div id="tpmPCRResponse"></div>
                </div>
                <div id="easyWinCache" class="tab-content">
                    <div class="form-group">
                        <label>Device ID (hex)</label>
                        <input type="text" id="cacheDeviceId" value="0x8000" placeholder="0x8000">
                    </div>
                    <button onclick="loadDeviceStatusCached()">LOAD CACHED STATUS</button>
                    <div id="cacheResponse"></div>
                </div>
                <div id="easyWinHistory" class="tab-content">
                    <div class="form-group">
                        <label>Device ID (optional, hex)</label>
                        <input type="text" id="historyDeviceId" placeholder="0x8000">
                    </div>
                    <div class="form-group">
                        <label>Limit</label>
                        <input type="number" id="historyLimit" value="50" min="1" max="1000">
                    </div>
                    <button onclick="loadOperationHistory()">LOAD OPERATION HISTORY</button>
                    <div id="historyResponse"></div>
                </div>
                <div id="easyWinStats" class="tab-content">
                    <button onclick="loadOperationStats()">LOAD OPERATION STATISTICS</button>
                    <div id="statsResponse"></div>
                </div>
                <div id="easyWinHealth" class="tab-content">
                    <button onclick="loadHealthScore()">LOAD SUBSYSTEM HEALTH</button>
                    <div id="healthResponse"></div>
                </div>
                <div id="easyWinAudit" class="tab-content">
                    <h3 style="color: #ff0; margin-bottom: 15px;">IA3 COMPLIANCE AUDIT LOG</h3>
                    <div class="form-group">
                        <label>Quick Filters</label>
                        <select id="auditQuickFilter" onchange="applyAuditQuickFilter()">
                            <option value="all">All Events (Last 100)</option>
                            <option value="critical">Critical Events Only</option>
                            <option value="failed">Failed Operations</option>
                            <option value="today">Today's Events</option>
                            <option value="device">By Device ID...</option>
                        </select>
                    </div>
                    <div id="auditDeviceFilter" style="display: none;" class="form-group">
                        <label>Device ID (hex)</label>
                        <input type="text" id="auditDeviceId" placeholder="0x8000">
                    </div>
                    <div class="form-group">
                        <label>Limit</label>
                        <input type="number" id="auditLimit" value="50" min="1" max="1000">
                    </div>
                    <button onclick="loadAuditEvents()">LOAD AUDIT LOG</button>
                    <button onclick="loadAuditStatistics()" class="btn-small" style="margin-left: 10px;">STATISTICS</button>
                    <button onclick="loadDatabaseInfo()" class="btn-small" style="margin-left: 10px;">DB INFO</button>
                    <div id="auditResponse"></div>
                </div>
            </div>

            <!-- AI Query Interface -->
            <div class="card" style="grid-column: 1 / -1;">
                <h2>[ AI-POWERED AUDIT QUERY ]</h2>
                <div class="form-group">
                    <label>Natural Language Query</label>
                    <textarea id="aiAuditQuery" rows="3" placeholder="Examples:
- Show me all critical audit events from today
- What devices have failed operations?
- Show thermal impact of recent activations
- Export audit log for the last 7 days"></textarea>
                </div>
                <button onclick="submitAIAuditQuery()">QUERY AI</button>
                <button onclick="showAuditExamples()" class="btn-small" style="margin-left: 10px;">EXAMPLES</button>
                <div id="aiAuditResponse"></div>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh status
        setInterval(updateStatus, 5000);
        updateStatus();

        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                let html = '';
                html += `<div class="status-item ${data.components.integrator ? 'active' : 'inactive'}">
                    Integrator: ${data.components.integrator ? '✓' : '✗'}
                </div>`;
                html += `<div class="status-item ${data.services.ollama ? 'active' : 'inactive'}">
                    Ollama: ${data.services.ollama ? '✓' : '✗'}
                </div>`;
                html += `<div class="status-item ${data.services.vllm ? 'active' : 'inactive'}">
                    vLLM: ${data.services.vllm ? '✓' : '✗'}
                </div>`;
                html += `<div class="status-item ${data.components.hephaestus ? 'active' : 'inactive'}">
                    Hephaestus: ${data.components.hephaestus ? '✓' : '✗'}
                </div>`;

                document.getElementById('statusBar').innerHTML = html;

                // Update metrics
                if (data.stats && data.stats.history) {
                    const metricsHtml = `
                        <div class="metric">
                            <div class="metric-value">${data.stats.history.total || 0}</div>
                            <div class="metric-label">Total Queries</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${Math.round((data.stats.history.success_rate || 0) * 100)}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${Math.round((data.stats.history.cache_rate || 0) * 100)}%</div>
                            <div class="metric-label">Cache Hit Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.stats.history.reasoning_tasks || 0}</div>
                            <div class="metric-label">Reasoning Tasks</div>
                        </div>
                    `;
                    document.getElementById('metrics').innerHTML = metricsHtml;
                }
            } catch (error) {
                console.error('Status update failed:', error);
            }
        }

        async function runQuery() {
            const prompt = document.getElementById('queryPrompt').value;
            const mode = document.getElementById('queryMode').value;
            const model = document.getElementById('queryModel').value;
            const responseDiv = document.getElementById('queryResponse');

            if (!prompt) {
                responseDiv.innerHTML = '<div class="response error">Please enter a prompt</div>';
                return;
            }

            responseDiv.innerHTML = '<div class="response">Processing...</div>';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, mode, model })
                });

                const data = await response.json();

                if (response.ok) {
                    responseDiv.innerHTML = `
                        <div class="response success">
                            <strong>Response:</strong><br><br>
                            ${data.content}<br><br>
                            <small>
                                ⏱️ ${data.latency_ms}ms |
                                🎯 ${data.mode} mode |
                                ${data.cached ? '⚡ Cached' : ''}
                                ${data.reasoning_steps > 0 ? `🧠 ${data.reasoning_steps} steps` : ''}
                            </small>
                        </div>
                    `;
                } else {
                    responseDiv.innerHTML = `<div class="response error">${data.error}</div>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        async function runBenchmark() {
            const numRuns = document.getElementById('benchmarkRuns').value;
            const responseDiv = document.getElementById('benchmarkResponse');

            responseDiv.innerHTML = '<div class="response">Starting benchmark... (this may take several minutes)</div>';

            try {
                const response = await fetch('/api/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_runs: parseInt(numRuns) })
                });

                const data = await response.json();

                if (response.ok) {
                    responseDiv.innerHTML = `
                        <div class="response success">
                            ${data.message}<br>
                            <small>Check "View Results" in a few minutes</small>
                        </div>
                    `;
                } else {
                    responseDiv.innerHTML = `<div class="response error">${data.error}</div>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        async function getBenchmarkResults() {
            const responseDiv = document.getElementById('benchmarkResponse');

            try {
                const response = await fetch('/api/benchmark/results');
                const data = await response.json();

                if (data.length === 0) {
                    responseDiv.innerHTML = '<div class="response">No benchmark results yet</div>';
                } else {
                    const latest = data[data.length - 1];
                    responseDiv.innerHTML = `
                        <div class="response success">
                            <pre>${JSON.stringify(latest, null, 2)}</pre>
                        </div>
                    `;
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        async function getSecurityReport() {
            const responseDiv = document.getElementById('securityResponse');
            responseDiv.innerHTML = '<div class="response">Generating report...</div>';

            try {
                const response = await fetch('/api/security/report');
                const data = await response.json();

                if (response.ok) {
                    responseDiv.innerHTML = `
                        <div class="response success">
                            <pre>${JSON.stringify(data, null, 2)}</pre>
                        </div>
                    `;
                } else {
                    responseDiv.innerHTML = `<div class="response error">${data.error}</div>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        async function loadScripts() {
            const scriptList = document.getElementById('scriptList');
            scriptList.innerHTML = '<div class="loading">LOADING...</div>';

            try {
                const response = await fetch('/api/scripts');
                const scripts = await response.json();

                if (scripts.length === 0) {
                    scriptList.innerHTML = '<div class="loading">NO SCRIPTS FOUND</div>';
                } else {
                    let html = '';
                    scripts.forEach(script => {
                        html += `
                            <div class="script-item">
                                <div class="script-info">
                                    <div class="script-name">${script.name}</div>
                                    <div class="script-meta">
                                        SIZE: ${Math.round(script.size / 1024)}KB |
                                        MODIFIED: ${new Date(script.modified).toLocaleString()}
                                    </div>
                                </div>
                                <button class="btn-small" onclick="runScript('${script.path}')">EXECUTE</button>
                            </div>
                        `;
                    });
                    scriptList.innerHTML = html;
                }
            } catch (error) {
                scriptList.innerHTML = `<div class="response error">ERROR: ${error.message}</div>`;
            }
        }

        async function runScript(scriptPath) {
            if (!confirm(`Run script: ${scriptPath}?`)) return;

            try {
                const response = await fetch('/api/scripts/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ script_path: scriptPath })
                });

                const data = await response.json();

                if (response.ok) {
                    alert(`Script started! Process ID: ${data.process_id}\n\nCheck status in a few moments.`);
                    // Poll for status
                    setTimeout(() => checkScriptStatus(data.process_id), 3000);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function checkScriptStatus(processId) {
            try {
                const response = await fetch(`/api/scripts/status/${processId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    alert(`Script completed!\n\nReturn code: ${data.returncode}\n\nOutput available in console.`);
                    console.log('Script output:', data.stdout);
                    if (data.stderr) console.error('Script errors:', data.stderr);
                } else if (data.status === 'running') {
                    setTimeout(() => checkScriptStatus(processId), 3000);
                } else {
                    alert(`Script status: ${data.status}\n${data.error || ''}`);
                }
            } catch (error) {
                console.error('Status check failed:', error);
            }
        }

        async function listWorkflows() {
            const responseDiv = document.getElementById('workflowResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';

            try {
                const response = await fetch('/api/hephaestus/workflows');
                const workflows = await response.json();

                if (response.ok) {
                    if (workflows.length === 0) {
                        responseDiv.innerHTML = '<div class="response">No workflows yet</div>';
                    } else {
                        let html = '<div class="response success">';
                        workflows.forEach(wf => {
                            html += `
                                <div style="margin-bottom: 15px; padding: 10px; background: white; border-radius: 5px;">
                                    <strong>${wf.project_name}</strong><br>
                                    <small>Phase: ${wf.current_phase} | Progress: ${Math.round(wf.progress_percentage)}%</small>
                                </div>
                            `;
                        });
                        html += '</div>';
                        responseDiv.innerHTML = html;
                    }
                } else {
                    responseDiv.innerHTML = `<div class="response error">${workflows.error}</div>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        async function getMCPStatus() {
            const responseDiv = document.getElementById('mcpResponse');
            responseDiv.innerHTML = '<div class="response">Checking...</div>';

            try {
                const response = await fetch('/api/mcp/status');
                const data = await response.json();

                if (data.configured) {
                    let html = `<div class="response success">
                        <strong>${data.total_servers} MCP Servers Configured</strong><br><br>`;
                    data.servers.forEach(server => {
                        html += `• ${server.name}<br>`;
                    });
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else {
                    responseDiv.innerHTML = '<div class="response error">MCP servers not configured</div>';
                }
            } catch (error) {
                responseDiv.innerHTML = `<div class="response error">Error: ${error.message}</div>`;
            }
        }

        // DSMIL Easy Wins Functions
        function switchEasyWinTab(tabName) {
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            const tabs = document.querySelectorAll('.tabs .tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            const tabMap = {'thermal':'easyWinThermal','tpm':'easyWinTPM','cache':'easyWinCache','history':'easyWinHistory','stats':'easyWinStats','health':'easyWinHealth','audit':'easyWinAudit'};
            document.getElementById(tabMap[tabName]).classList.add('active');
            event.target.classList.add('active');
        }

        async function loadThermalEnhanced() {
            const responseDiv = document.getElementById('thermalEnhancedResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                const response = await fetch('/api/dsmil/thermal-enhanced');
                const data = await response.json();
                if (response.ok && data.success) {
                    const thermal = data.thermal;
                    let html = '<div class="response success">';
                    html += `<strong>Status: ${thermal.overall_status.toUpperCase()}</strong> | <strong>Max: ${thermal.max_temp}°C</strong><br><br>`;
                    thermal.zones.forEach(zone => {
                        const color = zone.status==='critical'?'#f00':zone.status==='warning'?'#ff0':'#0f0';
                        html += `<span style="color:${color}">● ${zone.type}: ${zone.temp_c}°C [${zone.status}]</span><br>`;
                    });
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadTPMPCRState() {
            const responseDiv = document.getElementById('tpmPCRResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                const response = await fetch('/api/dsmil/tpm-pcr-state');
                const data = await response.json();
                if (response.ok && data.success) {
                    let html = '<div class="response success">';
                    if (data.pcr_state.success) {
                        html += `<strong>PCR Count: ${data.pcr_state.pcr_count}</strong><br><br>`;
                        for (const [pcr, value] of Object.entries(data.pcr_state.pcrs)) {
                            html += `PCR ${pcr}: ${value.substring(0, 32)}...<br>`;
                        }
                    } else { html += `Error: ${data.pcr_state.error}`; }
                    html += `<br><br><strong>Event Log: </strong>${data.event_log.log_available?data.event_log.event_count+' events':'Not available'}</div>`;
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadDeviceStatusCached() {
            const deviceId = document.getElementById('cacheDeviceId').value;
            const responseDiv = document.getElementById('cacheResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                const response = await fetch(`/api/dsmil/device-status-cached/${deviceId}`);
                const data = await response.json();
                if (response.ok && data.success) {
                    responseDiv.innerHTML = `<div class="response success"><pre>${JSON.stringify(data.status, null, 2)}</pre></div>`;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadOperationHistory() {
            const deviceId = document.getElementById('historyDeviceId').value;
            const limit = document.getElementById('historyLimit').value;
            const responseDiv = document.getElementById('historyResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                let url = `/api/dsmil/operation-history?limit=${limit}`;
                if (deviceId) url += `&device_id=${deviceId}`;
                const response = await fetch(url);
                const data = await response.json();
                if (response.ok && data.success) {
                    let html = '<div class="response success">';
                    html += `<strong>Total: ${data.history.length}</strong><br><br>`;
                    data.history.forEach(op => {
                        const color = op.success ? '#0f0' : '#f00';
                        html += `<span style="color:${color}">[${new Date(op.timestamp).toLocaleTimeString()}] ${op.device_name}: ${op.operation} ${op.success?'✓':'✗'}</span><br>`;
                    });
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadOperationStats() {
            const responseDiv = document.getElementById('statsResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                const response = await fetch('/api/dsmil/operation-stats');
                const data = await response.json();
                if (response.ok && data.success) {
                    const stats = data.stats;
                    let html = '<div class="response success">';
                    html += `<strong>Total: ${stats.total_operations}</strong> | <strong>Success: ${stats.successful_operations}</strong> | <strong>Failed: ${stats.failed_operations}</strong><br>`;
                    html += `<strong>Success Rate: ${stats.success_rate}%</strong><br><br>`;
                    if (stats.most_active_device) {
                        html += `<strong>Most Active:</strong> ${stats.most_active_device.device_name} (${stats.most_active_device.operation_count} ops)<br><br>`;
                    }
                    html += '<strong>By Type:</strong><br>';
                    for (const [op, count] of Object.entries(stats.operations_by_type)) { html += `${op}: ${count}<br>`; }
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadHealthScore() {
            const responseDiv = document.getElementById('healthResponse');
            responseDiv.innerHTML = '<div class="response">Loading...</div>';
            try {
                const response = await fetch('/api/dsmil/health-score');
                const data = await response.json();
                if (response.ok && data.success) {
                    const health = data.health_scores;
                    let html = '<div class="response success">';
                    html += `<strong>Overall: ${health.overall_health} - ${health.status.toUpperCase()}</strong><br><br><strong>Subsystems:</strong><br>`;
                    const sorted = Object.entries(health.subsystem_scores).sort((a, b) => b[1] - a[1]);
                    sorted.forEach(([subsystem, score]) => {
                        const color = score>0.9?'#0f0':score>0.7?'#ff0':'#f00';
                        const bar = '█'.repeat(Math.round(score * 10));
                        html += `<span style="color:${color}">${subsystem}: ${score} ${bar}</span><br>`;
                    });
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        // Audit Tab Functions
        function applyAuditQuickFilter() {
            const filter = document.getElementById('auditQuickFilter').value;
            const deviceFilter = document.getElementById('auditDeviceFilter');
            deviceFilter.style.display = filter === 'device' ? 'block' : 'none';
        }

        async function loadAuditEvents() {
            const responseDiv = document.getElementById('auditResponse');
            responseDiv.innerHTML = '<div class="response">Loading audit events...</div>';
            try {
                const filter = document.getElementById('auditQuickFilter').value;
                const limit = document.getElementById('auditLimit').value;
                let url = `/api/audit/events?limit=${limit}`;

                if (filter === 'critical') url += '&risk_level=critical';
                else if (filter === 'failed') url += '&success=false';
                else if (filter === 'today') {
                    const todayStart = new Date().setHours(0,0,0,0) / 1000;
                    url += `&start_time=${todayStart}`;
                } else if (filter === 'device') {
                    const deviceId = document.getElementById('auditDeviceId').value;
                    if (deviceId) url += `&device_id=${deviceId}`;
                }

                const response = await fetch(url);
                const data = await response.json();
                if (response.ok && data.success) {
                    let html = '<div class="response success">';
                    html += `<strong style="color:#ff0">IA3 AUDIT LOG - ${data.count} EVENTS</strong><br><br>`;
                    data.events.forEach(e => {
                        const riskColor = e.risk_level==='critical'?'#f00':e.risk_level==='high'?'#f80':e.risk_level==='medium'?'#ff0':'#0f0';
                        const statusColor = e.success?'#0f0':'#f00';
                        html += `<div style="margin-bottom:10px;padding:8px;border-left:3px solid ${riskColor};background:#001100">`;
                        html += `<span style="color:#888">[${e.datetime_iso}]</span> `;
                        html += `<span style="color:${statusColor}">${e.success?'✓':'✗'}</span> `;
                        html += `<strong style="color:#0f0">${e.device_name||e.device_id}</strong> `;
                        html += `<span style="color:#ff0">${e.operation.toUpperCase()}</span><br>`;
                        html += `<span style="color:#888">User: ${e.user} | Risk: </span>`;
                        html += `<span style="color:${riskColor}">${e.risk_level.toUpperCase()}</span>`;
                        if (e.thermal_impact) html += ` | <span style="color:#f80">Thermal: +${e.thermal_impact}°C</span>`;
                        if (e.details) html += `<br><span style="color:#aaa">${e.details}</span>`;
                        html += '</div>';
                    });
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadAuditStatistics() {
            const responseDiv = document.getElementById('auditResponse');
            responseDiv.innerHTML = '<div class="response">Loading statistics...</div>';
            try {
                const response = await fetch('/api/audit/statistics');
                const data = await response.json();
                if (response.ok && data.success) {
                    const s = data.statistics;
                    let html = '<div class="response success">';
                    html += '<strong style="color:#ff0;font-size:16px">IA3 COMPLIANCE STATISTICS</strong><br><br>';
                    html += `<strong>Total Events:</strong> ${s.total_events}<br>`;
                    html += `<strong>Success Rate:</strong> <span style="color:#0f0">${s.success_rate}%</span><br>`;
                    html += `<strong>Successful:</strong> ${s.successful_events}<br>`;
                    html += `<strong>Failed:</strong> <span style="color:#f00">${s.failed_events}</span><br><br>`;
                    html += '<strong>Risk Breakdown:</strong><br>';
                    for (const [risk, count] of Object.entries(s.risk_level_breakdown)) {
                        const color = risk==='critical'?'#f00':risk==='high'?'#f80':risk==='medium'?'#ff0':'#0f0';
                        html += `<span style="color:${color}">${risk.toUpperCase()}: ${count}</span><br>`;
                    }
                    html += '<br><strong>Operations by Type:</strong><br>';
                    for (const [op, count] of Object.entries(s.operations_by_type)) {
                        html += `${op}: ${count}<br>`;
                    }
                    if (s.most_active_devices.length > 0) {
                        html += '<br><strong>Most Active Devices:</strong><br>';
                        s.most_active_devices.slice(0,5).forEach(d => {
                            html += `${d.device_name||d.device_id}: ${d.count} ops<br>`;
                        });
                    }
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        async function loadDatabaseInfo() {
            const responseDiv = document.getElementById('auditResponse');
            responseDiv.innerHTML = '<div class="response">Loading database info...</div>';
            try {
                const response = await fetch('/api/audit/database-info');
                const data = await response.json();
                if (response.ok && data.success) {
                    const info = data.database_info;
                    let html = '<div class="response success">';
                    html += '<strong style="color:#ff0">AUDIT DATABASE INFORMATION</strong><br><br>';
                    html += `<strong>Path:</strong> ${info.database_path}<br>`;
                    html += `<strong>Size:</strong> ${info.size_mb} MB (${info.size_bytes} bytes)<br>`;
                    html += `<strong>Event Count:</strong> ${info.event_count}<br>`;
                    html += `<strong>Date Range:</strong> ${info.date_range_days} days<br>`;
                    html += `<strong>Oldest Event:</strong> ${info.oldest_event||'N/A'}<br>`;
                    html += `<strong>Newest Event:</strong> ${info.newest_event||'N/A'}<br>`;
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        // AI-Powered Audit Query Functions
        async function submitAIAuditQuery() {
            const query = document.getElementById('aiAuditQuery').value;
            const responseDiv = document.getElementById('aiAuditResponse');
            if (!query.trim()) {
                responseDiv.innerHTML = '<div class="response error">Please enter a query</div>';
                return;
            }
            responseDiv.innerHTML = '<div class="response">Processing AI query...</div>';
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({query: `Audit Query: ${query}`, mode: 'auto', model: 'quality'})
                });
                const data = await response.json();
                if (response.ok) {
                    let html = '<div class="response success">';
                    html += '<strong style="color:#ff0">AI ANALYSIS:</strong><br><br>';
                    html += data.response.replace(/\n/g, '<br>');
                    html += '<br><br><em style="color:#888">Tip: Use natural language like "Show critical events from yesterday"</em>';
                    html += '</div>';
                    responseDiv.innerHTML = html;
                } else { responseDiv.innerHTML = `<div class="response error">${data.error||'Error'}</div>`; }
            } catch (error) { responseDiv.innerHTML = `<div class="response error">${error.message}</div>`; }
        }

        function showAuditExamples() {
            const examples = [
                "Show me all critical audit events from today",
                "What devices have failed operations in the last 24 hours?",
                "List all activate operations with thermal impact > 1°C",
                "Show audit statistics for the last 7 days",
                "Which device has the most operations logged?",
                "Find all operations by user john",
                "Show me quarantined device access attempts"
            ];
            const responseDiv = document.getElementById('aiAuditResponse');
            let html = '<div class="response success">';
            html += '<strong style="color:#ff0">EXAMPLE QUERIES:</strong><br><br>';
            examples.forEach(ex => {
                html += `<span style="color:#0f0">• ${ex}</span><br>`;
            });
            html += '<br><em style="color:#888">Copy any example above into the query box</em>';
            html += '</div>';
            responseDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)


def main():
    """Start dashboard server"""

    print("=" * 70)
    print(" DSMIL AI - GUI Dashboard")
    print("=" * 70)
    print()

    # Create HTML template
    create_dashboard_html()
    print("✓ Dashboard template created")

    # Initialize components
    print("Initializing AI components...")
    initialize_components()

    print()
    print("=" * 70)
    print(" Starting Dashboard Server")
    print("=" * 70)
    print()
    print("Dashboard will be available at: http://localhost:5050")
    print("Press Ctrl+C to stop")
    print()

    # Start Flask app
    app.run(host='0.0.0.0', port=5050, debug=False)


if __name__ == "__main__":
    main()
