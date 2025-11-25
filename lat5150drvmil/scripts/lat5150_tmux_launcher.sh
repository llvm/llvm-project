#!/bin/bash
#
# LAT5150 DRVMIL - tmux Orchestration Launcher
# Creates a multi-window tmux session that groups the most common LAT5150 and
# DSMIL workflows (core control, dashboards, benchmarks, DSMIL tooling) into
# a single entry point that can be spread across two monitors.

set -euo pipefail

SESSION_NAME="${1:-lat5150-suite}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAT_SCRIPT="${PROJECT_ROOT}/lat5150.sh"
DSMIL_CONTROL="${PROJECT_ROOT}/dsmil_control_centre.py"
DSMIL_TOOL="${PROJECT_ROOT}/dsmil.py"
LAT_VENV_DIR="${PROJECT_ROOT}/.venv"
ENSURE_VENV="${PROJECT_ROOT}/scripts/ensure_venv.sh"
GET_API_PORT="${PROJECT_ROOT}/scripts/get_api_port.sh"

# Get API port dynamically
get_api_port() {
    if [ -x "${GET_API_PORT}" ]; then
        "${GET_API_PORT}"
    else
        echo "8765"
    fi
}

API_PORT=$(get_api_port)

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required. Install with: sudo apt install tmux"
    exit 1
fi

if [ -x "$ENSURE_VENV" ]; then
    "$ENSURE_VENV" >/dev/null 2>&1 || true
fi

activate_pane() {
    tmux send-keys -t "$1" "if [ -f \"${LAT_VENV_DIR}/bin/activate\" ]; then source \"${LAT_VENV_DIR}/bin/activate\"; fi" C-m
}

if [ ! -x "${LAT_SCRIPT}" ]; then
    echo "lat5150.sh not found or not executable at ${LAT_SCRIPT}"
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists – attaching."
    tmux set-environment -t "${SESSION_NAME}" LAT5150_VENV "${LAT_VENV_DIR}"
    tmux attach-session -t "${SESSION_NAME}"
    exit 0
fi

pane_message() {
    local target="$1"
    local content="$2"

    tmux send-keys -t "${target}" "clear" C-m
    tmux send-keys -t "${target}" "cat <<'MSG'" C-m
    while IFS= read -r line || [ -n "$line" ]; do
        tmux send-keys -t "${target}" "$line" C-m
    done <<< "${content}
"
    tmux send-keys -t "${target}" "MSG" C-m
}

setup_pane() {
    activate_pane "$1"
    pane_message "$1" "$2"
}

CORE_CONSOLE_TEXT="$(cat <<'EOF'
LAT5150 Core Console
====================
Use this pane for installs, service control, and quick commands.

Suggested boot sequence:
  ./lat5150.sh install
  ./lat5150.sh start-all
  ./lat5150.sh status
  ./lat5150.sh dashboard  (optional UI)

Other helpful commands:
  ./lat5150.sh benchmark
  ./lat5150.sh improve
  ./lat5150.sh logs api
  ./lat5150.sh logs improve

Docs: LAT5150_INTEGRATION_GUIDE.md
Tip: run './lat5150.sh help' for the full command list.
EOF
)"

LOGS_TEXT="$(cat <<'EOF'
LAT5150 Log Pane
================
Use this pane to follow journal logs or shell history.

Examples:
  ./lat5150.sh logs api
  ./lat5150.sh logs improve
  sudo journalctl -fu lat5150-unified-api
  sudo journalctl -fu lat5150-self-improvement

Note: journalctl commands may request sudo for system logs.
Press Ctrl+C to stop a streaming log at any time.
EOF
)"

IMPROVE_TEXT="$(cat <<'EOF'
LAT5150 Self-Improvement
========================
Run iterative self-improvement passes here.

Command:
  ./lat5150.sh improve

Optional flags:
  ./lat5150.sh improve --max-cycles 5

This pane is isolated so you can keep the session running while monitoring status/logs elsewhere.
EOF
)"

BENCHMARK_TEXT="$(cat <<'EOF'
Red Team Benchmark
==================
Launch benchmark runs here.

Command:
  ./lat5150.sh benchmark

Benchmark artifacts:
  02-ai-engine/redteam_benchmark_data/results/
EOF
)"

TEST_TEXT="$(cat <<'EOF'
Integration Tests
=================
Execute the combined test suite in this pane.

Command:
  ./lat5150.sh test

Phases:
  1. Python syntax checks
  2. Import smoke tests
  3. Service health checks
  4. API probe (:80)
  5. Dependency audit
EOF
)"

DASHBOARD_TEXT="$(cat <<'EOF'
Dashboard Launcher
==================
Bring up the AI GUI dashboard on port 5050.

Command:
  ./lat5150.sh dashboard

Access: http://localhost:5050
Stop:   Ctrl+C
EOF
)"

API_TEXT="$(cat <<EOF
Unified Tactical API
====================
Manually run the API or attach to service logs.

Commands:
  ./lat5150.sh api
  curl http://localhost:${API_PORT}/api/self-awareness

API Port: ${API_PORT} (no sudo required)
Use Ctrl+C to stop the foreground API server when running manually.
EOF
)"

DSMIL_CC_TEXT="$(cat <<'EOF'
DSMIL Control Centre
====================
Full-screen guided activation for the 104-device DSMIL stack.

Preferred command:
  ./dsmil_control_centre.py

Automation (multi-pane tmux layout):
  ./scripts/launch-dsmil-control-center.sh

Driver tip:
  sudo ./dsmil.py build-auto   # builds/loads drivers (prefers 104dev)
  sudo ./dsmil.py status       # quick kernel driver status

Note: sudo will be requested automatically when needed for kernel operations.
EOF
)"

DSMIL_DRIVER_TEXT="$(cat <<'EOF'
DSMIL Driver / Activation Tasks
===============================
Run kernel driver workflows and diagnostics here.

Quick start:
  sudo ./dsmil.py build-auto   # build + install (prefers dsmil-104dev)
  sudo ./dsmil.py status       # show which DSMIL driver is loaded

Common commands:
  ./dsmil.py build             # build drivers only
  ./dsmil.py load              # load best driver for this kernel
  ./dsmil.py control           # text-mode control centre

Note: sudo will be requested automatically when needed for kernel operations.
Use this pane for ad-hoc scripts (e.g., test_build_fallback.py).
EOF
)"

echo "Launching tmux session '${SESSION_NAME}' for LAT5150 DRVMIL..."
echo "  Window 0: lat5150-core (console, status monitor, log pane)"
echo "  Window 1: workflows (improve, benchmark, test)"
echo "  Window 2: dashboards (GUI + API launchers)"
echo "  Window 3: dsmil (control centre + driver tasks)"

# Create base session
if ! tmux new-session -d -s "${SESSION_NAME}" -n "lat5150-core" -c "${PROJECT_ROOT}"; then
    echo "Failed to start tmux session (check permissions or existing tmux server)."
    exit 1
fi
tmux set-environment -t "${SESSION_NAME}" LAT5150_VENV "${LAT_VENV_DIR}"

# Window 0 layout
tmux split-window -h -p 38 -t "${SESSION_NAME}":0 -c "${PROJECT_ROOT}"
tmux split-window -v -p 55 -t "${SESSION_NAME}":0.1 -c "${PROJECT_ROOT}"

    setup_pane "${SESSION_NAME}:0.0" "${CORE_CONSOLE_TEXT}"

    activate_pane "${SESSION_NAME}:0.1"
    tmux send-keys -t "${SESSION_NAME}:0.1" "cd '${PROJECT_ROOT}'" C-m
tmux send-keys -t "${SESSION_NAME}:0.1" "./lat5150.sh venv >/dev/null 2>&1 || true; while true; do clear; date '+%Y-%m-%d %H:%M:%S'; echo 'LAT5150 Status Monitor (auto-refresh every 30s)'; echo '================================================'; ./lat5150.sh status || echo 'Status command failed'; echo; echo 'Refreshing in 30s...'; sleep 30; done" C-m

    setup_pane "${SESSION_NAME}:0.2" "${LOGS_TEXT}"

# Window 1: workflows
tmux new-window -t "${SESSION_NAME}":1 -n "workflows" -c "${PROJECT_ROOT}"
tmux split-window -h -p 40 -t "${SESSION_NAME}":1 -c "${PROJECT_ROOT}"
tmux split-window -v -p 55 -t "${SESSION_NAME}":1.1 -c "${PROJECT_ROOT}"

    setup_pane "${SESSION_NAME}:1.0" "${IMPROVE_TEXT}"
    setup_pane "${SESSION_NAME}:1.1" "${BENCHMARK_TEXT}"
    setup_pane "${SESSION_NAME}:1.2" "${TEST_TEXT}"

# Window 2: dashboards/API
tmux new-window -t "${SESSION_NAME}":2 -n "dashboards" -c "${PROJECT_ROOT}"
tmux split-window -h -p 45 -t "${SESSION_NAME}":2 -c "${PROJECT_ROOT}"

    setup_pane "${SESSION_NAME}:2.0" "${DASHBOARD_TEXT}"
    setup_pane "${SESSION_NAME}:2.1" "${API_TEXT}"

# Window 3: DSMIL tools
tmux new-window -t "${SESSION_NAME}":3 -n "dsmil" -c "${PROJECT_ROOT}"
tmux split-window -h -p 42 -t "${SESSION_NAME}":3 -c "${PROJECT_ROOT}"

    setup_pane "${SESSION_NAME}:3.0" "${DSMIL_CC_TEXT}"
    setup_pane "${SESSION_NAME}:3.1" "${DSMIL_DRIVER_TEXT}"

# Session-wide quality of life options
tmux set-option -t "${SESSION_NAME}" mouse on
tmux set-option -t "${SESSION_NAME}" status-interval 5
tmux set-option -t "${SESSION_NAME}" status-left "#[bold]LAT5150#[default] "
tmux set-option -t "${SESSION_NAME}" status-right " #[fg=cyan]%Y-%m-%d %H:%M #[default]"

    tmux select-window -t "${SESSION_NAME}":0
tmux attach-session -t "${SESSION_NAME}"
