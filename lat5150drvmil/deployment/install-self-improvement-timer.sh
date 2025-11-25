#!/bin/bash
#
# Install LAT5150 AI Self-Improvement Scheduled Timer
# Configures SystemD timer to automatically run self-improvement daily
#
# Features:
# - Automated daily self-improvement runs at 2 AM
# - Runs 5 minutes after boot if missed
# - Integrates with Red Team Benchmark and Heretic
# - Session results saved and tracked
# - Journal logging for monitoring
#
# Usage:
#   sudo ./install-self-improvement-timer.sh install
#   sudo ./install-self-improvement-timer.sh remove
#   sudo ./install-self-improvement-timer.sh status
#   sudo ./install-self-improvement-timer.sh run-now
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AI_ENGINE_DIR="${PROJECT_ROOT}/02-ai-engine"
SERVICE_FILE="${SCRIPT_DIR}/systemd/lat5150-self-improvement.service"
TIMER_FILE="${SCRIPT_DIR}/systemd/lat5150-self-improvement.timer"
SERVICE_NAME="lat5150-self-improvement.service"
TIMER_NAME="lat5150-self-improvement.timer"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
TIMER_PATH="/etc/systemd/system/${TIMER_NAME}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${CYAN}[====]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python3
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi

    # Check self-improvement script exists
    if [ ! -f "${AI_ENGINE_DIR}/ai_self_improvement.py" ]; then
        log_error "AI Self-Improvement script not found"
        exit 1
    fi

    # Check benchmark script exists
    if [ ! -f "${AI_ENGINE_DIR}/redteam_ai_benchmark.py" ]; then
        log_error "Red Team Benchmark script not found"
        exit 1
    fi

    # Create session directories
    log_info "Creating session directories..."
    mkdir -p "${AI_ENGINE_DIR}/self_improvement_sessions"
    mkdir -p "${AI_ENGINE_DIR}/redteam_benchmark_data/results"
    chown -R $(logname):$(logname) "${AI_ENGINE_DIR}/self_improvement_sessions" 2>/dev/null || true
    chown -R $(logname):$(logname) "${AI_ENGINE_DIR}/redteam_benchmark_data" 2>/dev/null || true

    log_info "✓ Dependencies OK"
}

install_timer() {
    check_root
    check_dependencies

    log_section "Installing LAT5150 AI Self-Improvement Timer"
    echo ""

    # Copy service file
    log_info "Installing SystemD service..."
    cp "$SERVICE_FILE" "$SERVICE_PATH"
    chmod 644 "$SERVICE_PATH"

    # Copy timer file
    log_info "Installing SystemD timer..."
    cp "$TIMER_FILE" "$TIMER_PATH"
    chmod 644 "$TIMER_PATH"

    # Reload SystemD
    log_info "Reloading SystemD daemon..."
    systemctl daemon-reload

    # Enable timer (not service - timer will trigger service)
    log_info "Enabling timer..."
    systemctl enable "$TIMER_NAME"

    # Start timer
    log_info "Starting timer..."
    systemctl start "$TIMER_NAME"

    # Wait a moment
    sleep 2

    # Check status
    if systemctl is-active --quiet "$TIMER_NAME"; then
        log_info "✓ Timer started successfully"
    else
        log_error "Timer failed to start"
        log_warn "Check logs with: journalctl -u $TIMER_NAME -n 50"
        exit 1
    fi

    echo ""
    log_section "Installation Complete"
    echo ""
    log_info "Timer: $TIMER_NAME"
    log_info "Status: $(systemctl is-active $TIMER_NAME)"
    log_info "Enabled: $(systemctl is-enabled $TIMER_NAME)"
    echo ""
    log_info "🚀 Self-Improvement Schedule:"
    log_info "   • Daily runs at 2:00 AM (±30 min randomization)"
    log_info "   • Runs 5 minutes after boot if missed"
    log_info "   • Target score: 80%"
    log_info "   • Max cycles: 5"
    log_info "   • Auto-abliteration enabled"
    echo ""
    log_info "Next scheduled run:"
    systemctl list-timers $TIMER_NAME --no-pager
    echo ""
    log_info "Session results: ${AI_ENGINE_DIR}/self_improvement_sessions/"
    echo ""
    log_info "Useful commands:"
    log_info "  sudo systemctl status $TIMER_NAME"
    log_info "  sudo systemctl list-timers $TIMER_NAME"
    log_info "  sudo journalctl -u $SERVICE_NAME -f"
    log_info "  sudo ./install-self-improvement-timer.sh run-now"
    echo ""
}

remove_timer() {
    check_root

    log_section "Removing LAT5150 AI Self-Improvement Timer"
    echo ""

    # Stop timer
    if systemctl is-active --quiet "$TIMER_NAME"; then
        log_info "Stopping timer..."
        systemctl stop "$TIMER_NAME"
    fi

    # Stop service (if running)
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Stopping service..."
        systemctl stop "$SERVICE_NAME"
    fi

    # Disable timer
    if systemctl is-enabled --quiet "$TIMER_NAME" 2>/dev/null; then
        log_info "Disabling timer..."
        systemctl disable "$TIMER_NAME"
    fi

    # Remove files
    if [ -f "$TIMER_PATH" ]; then
        log_info "Removing timer file..."
        rm "$TIMER_PATH"
    fi

    if [ -f "$SERVICE_PATH" ]; then
        log_info "Removing service file..."
        rm "$SERVICE_PATH"
    fi

    # Reload SystemD
    log_info "Reloading SystemD daemon..."
    systemctl daemon-reload
    systemctl reset-failed 2>/dev/null || true

    echo ""
    log_info "✓ Timer removed"
    echo ""
}

show_status() {
    log_section "LAT5150 AI Self-Improvement Timer Status"
    echo ""

    if [ ! -f "$TIMER_PATH" ]; then
        log_warn "Timer not installed"
        return
    fi

    # Timer status
    echo "Timer Status:"
    systemctl status "$TIMER_NAME" --no-pager | head -15
    echo ""

    # Next scheduled run
    echo "Schedule:"
    systemctl list-timers "$TIMER_NAME" --no-pager
    echo ""

    log_info "Enabled: $(systemctl is-enabled $TIMER_NAME 2>/dev/null || echo 'disabled')"
    log_info "Active:  $(systemctl is-active $TIMER_NAME 2>/dev/null || echo 'inactive')"
    echo ""

    # Recent service runs
    echo "Recent Self-Improvement Runs:"
    journalctl -u "$SERVICE_NAME" -n 20 --no-pager --since "7 days ago" || echo "No recent runs"
    echo ""

    # Check latest session
    local latest_session=$(ls -t "${AI_ENGINE_DIR}/self_improvement_sessions"/improvement_*.json 2>/dev/null | head -1)
    if [ -n "$latest_session" ]; then
        log_info "Latest session: $(basename $latest_session)"
        echo ""
        echo "Latest Results:"
        cat "$latest_session" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Session ID:    {data['session_id']}\")
    print(f\"  Initial Score: {data['initial_score']:.1f}%\")
    print(f\"  Final Score:   {data['final_score']:.1f}%\")
    print(f\"  Improvement:   {data['total_improvement']:+.1f}%\")
    print(f\"  Target Met:    {'✓ YES' if data['target_reached'] else '✗ NO'}\")
    print(f\"  Cycles Run:    {len(data['cycles'])}\")
    print(f\"  Duration:      {data['total_duration_seconds']}s\")
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
"
    else
        log_warn "No sessions found yet"
    fi
    echo ""
}

run_now() {
    check_root

    log_section "Running Self-Improvement Now"
    echo ""

    if [ ! -f "$SERVICE_PATH" ]; then
        log_error "Service not installed. Run: sudo $0 install"
        exit 1
    fi

    log_info "Triggering immediate self-improvement run..."
    log_info "This may take several minutes..."
    echo ""

    # Start service (oneshot - will run and exit)
    systemctl start "$SERVICE_NAME"

    echo ""
    log_info "✓ Self-improvement run complete"
    echo ""
    log_info "View results with: sudo $0 status"
    echo ""
}

show_help() {
    cat <<EOF
Install LAT5150 AI Self-Improvement Scheduled Timer

Usage:
    sudo $0 <command>

Commands:
    install     Install and enable self-improvement timer
    remove      Remove self-improvement timer
    status      Show timer status and latest results
    run-now     Trigger immediate self-improvement run
    help        Show this help message

Examples:
    # Install scheduled timer
    sudo $0 install

    # Check status and next run time
    sudo $0 status

    # Run self-improvement immediately
    sudo $0 run-now

    # Remove timer
    sudo $0 remove

Description:
    Configures SystemD timer to automatically run AI self-improvement
    sessions on a schedule. Each session:

    1. Runs Red Team Benchmark (12 offensive security tests)
    2. Analyzes results for refusals and hallucinations
    3. Applies Heretic abliteration if needed
    4. Re-runs benchmark to measure improvement
    5. Repeats until 80% score or improvement plateau
    6. Saves session results to JSON

Schedule:
    - Daily at 2:00 AM (±30 minute randomization)
    - Runs 5 minutes after boot if missed during downtime
    - Persistent across reboots

Configuration:
    Timer:   $TIMER_PATH
    Service: $SERVICE_PATH
    Results: ${AI_ENGINE_DIR}/self_improvement_sessions/
    Target:  80% benchmark score
    Max:     5 improvement cycles per session

Monitoring:
    - Watch logs: sudo journalctl -u $SERVICE_NAME -f
    - List timers: systemctl list-timers $TIMER_NAME
    - View results: sudo $0 status

Integration:
    - Works with existing Heretic abliteration system
    - Uses Red Team Benchmark (12 offensive security tests)
    - Integrates with Enhanced AI Engine
    - Results queryable via Unified Tactical API

Natural Language Access (via API):
    curl -X POST http://localhost/api/query \\
      -H "Content-Type: application/json" \\
      -d '{"query": "show self improvement status"}'

EOF
}

# Main
case "${1:-help}" in
    install)
        install_timer
        ;;
    remove)
        remove_timer
        ;;
    status)
        show_status
        ;;
    run-now)
        run_now
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
