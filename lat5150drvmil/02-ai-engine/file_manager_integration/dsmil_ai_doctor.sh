#!/usr/bin/env bash
###############################################################################
# dsmil_ai_doctor.sh - DSMIL AI Environment & Server Diagnostics
###############################################################################
set -o errexit
set -o nounset
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()      { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()       { echo -e "${GREEN}[OK]${NC} $*"; }
warn()     { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()      { echo -e "${RED}[ERROR]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AI_ENGINE_DIR="$PROJECT_ROOT/02-ai-engine"
API_SERVER="$AI_ENGINE_DIR/dsmil_terminal_api.py"
API_CLIENT="$AI_ENGINE_DIR/dsmil_api_client.py"
SOCKET_PATH="${DSMIL_SOCKET_PATH:-/tmp/dsmil-ai-$(id -u).sock}"

export PYTHONPATH="$AI_ENGINE_DIR:$PROJECT_ROOT:${PYTHONPATH-}"

banner() {
  echo -e "${BOLD}${CYAN}"
  cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                        DSMIL AI DOCTOR                           ║
║            Quick health-check for DSMIL AI stack                 ║
╚══════════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"
}

check_python() {
  log "Checking python3..."
  if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found in PATH"
    exit 1
  fi
  PYV="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo "unknown")"
  ok "python3 found (version $PYV)"
}

check_project_layout() {
  log "Checking project layout..."
  if [ ! -d "$AI_ENGINE_DIR" ]; then
    err "AI engine directory missing: $AI_ENGINE_DIR"
    exit 1
  fi
  if [ ! -f "$API_SERVER" ]; then
    err "API server missing: $API_SERVER"
    exit 1
  fi
  if [ ! -f "$API_CLIENT" ]; then
    err "API client missing: $API_CLIENT"
    exit 1
  fi
  ok "Project layout looks valid"
  log "PYTHONPATH=${PYTHONPATH}"
}

check_python_deps() {
  log "Checking Python dependencies (basic import smoke test)..."
  local imports=("json" "pathlib" "asyncio")
  local failed=0

  for m in "${imports[@]}"; do
    if ! python3 -c "import $m" >/dev/null 2>&1; then
      warn "Failed to import '$m'"
      failed=1
    fi
  done

  if [ $failed -ne 0 ]; then
    warn "Some basic modules failed to import (above). Core deps may also be missing."
  else
    ok "Basic imports OK"
  fi
}

check_server_running() {
  log "Checking DSMIL API server status..."
  if [ -S "$SOCKET_PATH" ]; then
    log "Socket exists at $SOCKET_PATH; trying ping..."
    if python3 "$API_CLIENT" ping >/dev/null 2>&1; then
      ok "Server is running and responding to ping"
      return 0
    else
      warn "Socket exists but ping failed; server may be hung or incompatible"
      return 1
    fi
  else
    warn "Socket not found at $SOCKET_PATH"
    return 1
  fi
}

start_server() {
  log "Attempting to start DSMIL API server..."
  if [ -n "${DSMIL_SOCKET_PATH-}" ]; then
    python3 "$API_SERVER" --daemon --socket "$SOCKET_PATH" >/dev/null 2>&1 &
  else
    python3 "$API_SERVER" --daemon >/dev/null 2>&1 &
  fi
  local pid=$!

  for _ in {1..30}; do
    sleep 0.2
    if python3 "$API_CLIENT" ping >/dev/null 2>&1; then
      ok "Server started (PID $pid), ping OK"
      return 0
    fi
  done

  err "Failed to start DSMIL API server (timeout waiting for ping)"
  return 1
}

check_session_flow() {
  log "Running end-to-end session flow test..."
  local tmpdir; tmpdir="$(mktemp -d /tmp/dsmil_test_XXXXXX)"
  local session_id=""

  log "Creating session in $tmpdir..."
  local out
  if ! out="$(python3 "$API_CLIENT" create "$tmpdir" 2>&1)"; then
    err "Session creation failed:"
    echo "$out"
    rm -rf "$tmpdir"
    exit 1
  fi

  session_id="$(echo "$out" | awk '/Session created:/ {print $3; exit}')"
  if [ -z "$session_id" ]; then
    err "Could not parse session ID from output:"
    echo "$out"
    rm -rf "$tmpdir"
    exit 1
  fi
  ok "Session created: $session_id"

  log "Running lightweight analyze..."
  if ! python3 "$API_CLIENT" analyze "$session_id" >/dev/null 2>&1; then
    warn "Analyze call failed; core server is up but analysis pipeline may be misconfigured"
  else
    ok "Analyze call succeeded"
  fi

  log "Closing session..."
  python3 "$API_CLIENT" close "$session_id" >/dev/null 2>&1 || warn "Session close failed (non-critical)"

  rm -rf "$tmpdir"
  ok "Session flow test completed"
}

main() {
  banner
  check_python
  check_project_layout
  check_python_deps

  if ! check_server_running; then
    log "Server not healthy; trying to bootstrap..."
    if ! start_server; then
      err "Server failed to start; fix issues above and re-run doctor."
      exit 1
    fi
  fi

  check_session_flow

  echo ""
  echo -e "${BOLD}${GREEN}All critical checks passed.${NC}"
  echo "If anything was flagged as WARN, review those sections for tuning."
}

main "$@"
