#!/bin/bash
#
# LAT5150 DRVMIL - Unified System Launcher and Control Script
# Version: 2.0.2
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAT5150_ROOT="${SCRIPT_DIR}"
AI_ENGINE="${LAT5150_ROOT}/02-ai-engine"
WEB_INTERFACE="${LAT5150_ROOT}/03-web-interface"
DEPLOYMENT="${LAT5150_ROOT}/deployment"
VENV_DIR="${LAT5150_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"
GET_API_PORT="${LAT5150_ROOT}/scripts/get_api_port.sh"

# Get API port (dynamic allocation to avoid sudo requirement)
get_api_port() {
    if [ -x "${GET_API_PORT}" ]; then
        "${GET_API_PORT}"
    else
        # Fallback if port script not available
        echo "8765"
    fi
}

API_PORT="${LAT5150_API_PORT:-$(get_api_port)}"

# Service names
SERVICE_UNIFIED_API="lat5150-unified-api"
SERVICE_SELF_IMPROVE="lat5150-self-improvement"
TIMER_SELF_IMPROVE="lat5150-self-improvement.timer"

# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------

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
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $1"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
}

log_debug() {
    if [ "${DEBUG:-0}" = "1" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        log_warn "Some operations require sudo. You may be prompted for password."
    fi
}

# ---------------------------------------------------------------------------
# Virtual Environment Helper Functions
# ---------------------------------------------------------------------------

venv_exists() {
    [ -d "${VENV_DIR}" ] && [ -f "${VENV_PYTHON}" ]
}

venv_is_active() {
    [ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" = "${VENV_DIR}" ]
}

venv_create() {
    log_info "Creating virtual environment at ${VENV_DIR}..."
    
    if ! python3 -m venv "${VENV_DIR}"; then
        log_error "Failed to create virtual environment"
        return 1
    fi
    
    # Verify creation
    if [ ! -f "${VENV_PYTHON}" ]; then
        log_error "Virtual environment created but Python executable not found"
        return 1
    fi
    
    log_success "Virtual environment created"
    log_debug "Python: ${VENV_PYTHON}"
    return 0
}

venv_activate() {
    if venv_is_active; then
        log_success "Virtual environment already active"
        return 0
    fi
    
    if ! venv_exists; then
        log_error "Virtual environment does not exist at ${VENV_DIR}"
        return 1
    fi
    
    log_info "Activating virtual environment..."
    
    # shellcheck disable=SC1090
    if source "${VENV_DIR}/bin/activate"; then
        log_success "Virtual environment activated: ${VIRTUAL_ENV}"
        log_debug "Python: $(command -v python3)"
        log_debug "Pip: $(command -v pip)"
        return 0
    else
        log_error "Failed to activate virtual environment"
        return 1
    fi
}

venv_install_deps() {
    log_info "Installing Python dependencies from requirements.txt..."

    if ! venv_exists; then
        log_error "Virtual environment does not exist"
        return 1
    fi

    log_debug "Using pip: ${VENV_PIP}"

    pip_network_ok=true
    if ! "${VENV_PYTHON}" -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple', timeout=3)" >/dev/null 2>&1; then
        pip_network_ok=false
        log_warn "PyPI unreachable; skipping dependency installation (ensure requirements are already cached)"
    fi

    if [ "$pip_network_ok" = true ] && [ -f "${LAT5150_ROOT}/requirements.txt" ]; then
        if ! "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel >/dev/null; then
            log_warn "Failed to upgrade pip (continuing anyway)"
        fi
        if ! "${VENV_PYTHON}" -m pip install -r "${LAT5150_ROOT}/requirements.txt"; then
            log_warn "Failed to install requirements.txt; continuing with existing packages"
        fi
    elif [ "$pip_network_ok" = true ]; then
        log_warn "requirements.txt missing; installing core packages"
        "${VENV_PYTHON}" -m pip install flask flask-cors >/dev/null || log_warn "Failed to install fallback packages; please install manually"
    fi

    local flask_version
    flask_version=$("${VENV_PYTHON}" -c 'import flask; print(flask.__version__)' 2>/dev/null || echo "unknown")
    log_success "Dependencies present: flask ${flask_version}"
    return 0
}

venv_setup_alias() {
    log_info "Setting up 'venv' shell alias..."
    
    local -r venv_alias="alias venv='source ${VENV_DIR}/bin/activate'"
    local -r venv_marker="# LAT5150 Virtual Environment Alias"
    
    # Determine user's home directory
    local real_home
    if [ -n "${SUDO_USER:-}" ]; then
        real_home=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
    else
        real_home="${HOME}"
    fi
    
    local -r bashrc="${real_home}/.bashrc"
    local -r bash_aliases="${real_home}/.bash_aliases"
    
    # Function to add alias to file
    add_alias_to_file() {
        local target_file="$1"
        
        # Create parent directory if needed
        mkdir -p "$(dirname "${target_file}")" 2>/dev/null || true
        
        # Create file if it doesn't exist
        if [ ! -f "${target_file}" ]; then
            touch "${target_file}"
        fi
        
        # Check if alias already exists
        if grep -qF "${venv_marker}" "${target_file}" 2>/dev/null; then
            log_success "'venv' alias already in $(basename "${target_file}")"
            return 0
        fi
        
        # Add alias
        {
            echo ""
            echo "${venv_marker}"
            echo "${venv_alias}"
        } >> "${target_file}"
        
        # Fix ownership if running as sudo
        if [ -n "${SUDO_USER:-}" ]; then
            chown "${SUDO_USER}:${SUDO_USER}" "${target_file}" 2>/dev/null || true
        fi
        
        log_success "Added 'venv' alias to $(basename "${target_file}")"
    }
    
    # Add to both files
    add_alias_to_file "${bashrc}"
    add_alias_to_file "${bash_aliases}"
    
    # Source bashrc to activate alias in current session
    # shellcheck disable=SC1090
    source "${bashrc}" 2>/dev/null || log_warn "Could not source ${bashrc}"
    
    return 0
}

venv_ensure_ready() {
    log_info "Ensuring virtual environment is ready..."
    
    # Create if doesn't exist
    if ! venv_exists; then
        venv_create || return 1
    else
        log_success "Virtual environment exists"
    fi
    
    # Activate
    venv_activate || return 1
    
    # Install dependencies
    venv_install_deps || return 1
    
    # Setup alias
    venv_setup_alias || return 1
    
    # Verify
    echo ""
    log_info "Virtual Environment Verification:"
    log_info "  Location: ${VENV_DIR}"
    log_info "  Python: ${VENV_PYTHON}"
    log_info "  Pip: ${VENV_PIP}"
    
    local flask_version
    flask_version=$("${VENV_PYTHON}" -c 'import flask; print(flask.__version__)' 2>/dev/null || echo "NOT FOUND")
    log_info "  Flask: ${flask_version}"
    
    if [ "${flask_version}" = "NOT FOUND" ]; then
        log_error "Flask not properly installed in venv"
        return 1
    fi
    
    return 0
}

# ---------------------------------------------------------------------------
# Service Installation Helpers
# ---------------------------------------------------------------------------

service_create_unified_api() {
    log_info "Creating Unified API systemd service..."
    
    local service_file="/etc/systemd/system/${SERVICE_UNIFIED_API}.service"
    local api_script="${WEB_INTERFACE}/unified_tactical_api.py"
    
    if [ ! -f "${api_script}" ]; then
        log_error "API script not found: ${api_script}"
        return 1
    fi
    
    # Verify script can be executed with venv Python
    if ! "${VENV_PYTHON}" -c "import sys; sys.path.insert(0, '${WEB_INTERFACE}'); import unified_tactical_api" 2>/dev/null; then
        log_error "Cannot import unified_tactical_api with venv Python"
        log_info "Check: ${VENV_PYTHON} -c 'import flask'"
        return 1
    fi
    
    log_info "Creating systemd service file..."
    
    local api_port
    api_port=$(get_api_port)

    sudo tee "${service_file}" > /dev/null <<EOF
[Unit]
Description=LAT5150 Unified Tactical API
After=network.target

[Service]
Type=simple
User=${SUDO_USER:-${USER}}
WorkingDirectory=${WEB_INTERFACE}
Environment="PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="VIRTUAL_ENV=${VENV_DIR}"
Environment="PYTHONPATH=${WEB_INTERFACE}:${AI_ENGINE}"
Environment="LAT5150_API_PORT=${api_port}"
ExecStart=${VENV_PYTHON} ${api_script} --port ${api_port}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    if [ ! -f "${service_file}" ]; then
        log_error "Failed to create service file"
        return 1
    fi
    
    log_success "Service file created: ${service_file}"
    
    # Reload systemd
    sudo systemctl daemon-reload
    log_success "Systemd daemon reloaded"
    
    # Enable service
    if sudo systemctl enable "${SERVICE_UNIFIED_API}"; then
        log_success "Service enabled for autostart"
    else
        log_warn "Failed to enable service"
    fi
    
    return 0
}

service_create_self_improvement() {
    log_info "Creating Self-Improvement systemd service and timer..."
    
    local service_file="/etc/systemd/system/${SERVICE_SELF_IMPROVE}.service"
    local timer_file="/etc/systemd/system/${TIMER_SELF_IMPROVE}"
    local improve_script="${AI_ENGINE}/ai_self_improvement.py"
    
    if [ ! -f "${improve_script}" ]; then
        log_error "Self-improvement script not found: ${improve_script}"
        return 1
    fi
    
    # Create service file
    log_info "Creating service file..."
    
    sudo tee "${service_file}" > /dev/null <<EOF
[Unit]
Description=LAT5150 Self-Improvement Session
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=${AI_ENGINE}
Environment="PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="VIRTUAL_ENV=${VENV_DIR}"
Environment="PYTHONPATH=${AI_ENGINE}:${WEB_INTERFACE}"
ExecStart=${VENV_PYTHON} ${improve_script} run
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Create timer file
    log_info "Creating timer file..."
    
    sudo tee "${timer_file}" > /dev/null <<EOF
[Unit]
Description=LAT5150 Self-Improvement Timer (Daily at 2 AM)
Requires=${SERVICE_SELF_IMPROVE}.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    if [ ! -f "${service_file}" ] || [ ! -f "${timer_file}" ]; then
        log_error "Failed to create service/timer files"
        return 1
    fi
    
    log_success "Service and timer files created"
    
    # Reload systemd
    sudo systemctl daemon-reload
    log_success "Systemd daemon reloaded"
    
    # Enable timer
    if sudo systemctl enable "${TIMER_SELF_IMPROVE}"; then
        log_success "Timer enabled for autostart"
    else
        log_warn "Failed to enable timer"
    fi
    
    return 0
}

service_start() {
    local service_name="$1"
    
    log_info "Starting ${service_name}..."
    
    if sudo systemctl start "${service_name}" 2>/dev/null; then
        sleep 2
        if sudo systemctl is-active --quiet "${service_name}"; then
            log_success "${service_name} started and running"
            return 0
        else
            log_error "${service_name} started but not running"
            log_info "Check logs: sudo journalctl -u ${service_name} -n 50"
            return 1
        fi
    else
        log_error "${service_name} failed to start"
        log_info "Check logs: sudo journalctl -u ${service_name} -n 50"
        return 1
    fi
}

service_stop() {
    local service_name="$1"
    
    log_info "Stopping ${service_name}..."
    
    if sudo systemctl stop "${service_name}" 2>/dev/null; then
        log_success "${service_name} stopped"
        return 0
    else
        log_warn "${service_name} not running or not installed"
        return 1
    fi
}

service_is_active() {
    local service_name="$1"
    systemctl is-active --quiet "${service_name}" 2>/dev/null
}

service_remove() {
    local service_name="$1"
    local service_file="/etc/systemd/system/${service_name}.service"
    local timer_file="/etc/systemd/system/${service_name}.timer"
    
    log_info "Removing ${service_name}..."
    
    # Stop services
    sudo systemctl stop "${service_name}" 2>/dev/null || true
    sudo systemctl stop "${service_name}.timer" 2>/dev/null || true
    
    # Disable services
    sudo systemctl disable "${service_name}" 2>/dev/null || true
    sudo systemctl disable "${service_name}.timer" 2>/dev/null || true
    
    # Remove files
    sudo rm -f "${service_file}" "${timer_file}"
    
    # Reload
    sudo systemctl daemon-reload
    
    log_success "${service_name} removed"
}

# ---------------------------------------------------------------------------
# Command Implementations
# ---------------------------------------------------------------------------

cmd_install() {
    log_section "Installing LAT5150 DRVMIL Components"
    
    check_sudo

    if [ "${EUID:-0}" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        log_error "Do not execute './lat5150.sh install' under sudo."
        log_info "Run it as your normal user; the script will prompt for elevated commands automatically."
        exit 1
    fi
    
    # Step 1: Ensure virtual environment is ready
    echo ""
    log_section "Step 1/4: Virtual Environment Setup"
    
    if ! venv_ensure_ready; then
        log_error "Virtual environment setup failed"
        exit 1
    fi
    
    # Step 2: Install Unified API service
    echo ""
    log_section "Step 2/4: Unified Tactical API"
    
    if ! service_create_unified_api; then
        log_error "Failed to create Unified API service"
        log_info "Check manually: sudo journalctl -xe"
        exit 1
    fi
    
    # Step 3: Install Self-Improvement service
    echo ""
    log_section "Step 3/4: Self-Improvement Timer"
    
    if ! service_create_self_improvement; then
        log_warn "Self-improvement setup had issues (non-critical)"
    fi
    
    # Step 4: Shell integration
    echo ""
    log_section "Step 4/4: Shell Integration"
    
    if [ -f "${DEPLOYMENT}/setup-shell-integration.sh" ]; then
        log_info "Setting up shell integration..."
        cd "${DEPLOYMENT}"
        
        if [ -n "${SUDO_USER:-}" ]; then
            sudo -u "${SUDO_USER}" -H bash setup-shell-integration.sh 2>/dev/null || log_warn "Shell integration had issues"
        else
            bash setup-shell-integration.sh 2>/dev/null || log_warn "Shell integration had issues"
        fi
        
        log_success "Shell helpers configured"
        cd "${LAT5150_ROOT}"
    fi
    
    # Final verification
    echo ""
    log_section "Installation Complete"
    
    log_success "Virtual environment: ${VENV_DIR}"
    log_success "Python: ${VENV_PYTHON}"
    log_success "Services created and enabled"
    
    if venv_is_active; then
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ Virtual environment is ACTIVE${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    fi
    
    echo ""
    log_info "Next steps:"
    log_info "  1. Start services: ${CYAN}./lat5150.sh start-all${NC}"
    log_info "  2. Check status: ${CYAN}./lat5150.sh status${NC}"
    log_info "  3. Run tests: ${CYAN}./lat5150.sh test${NC}"
    log_info "  4. Run benchmark: ${CYAN}./lat5150.sh benchmark${NC}"
    echo ""
}

cmd_benchmark() {
    log_section "Running Red Team Benchmark"
    
    # Ensure venv is active
    if ! venv_is_active; then
        log_info "Activating virtual environment..."
        venv_activate || {
            log_error "Failed to activate virtual environment"
            log_info "Run: ./lat5150.sh install"
            exit 1
        }
    fi
    
    # Check if benchmark script exists
    local benchmark_script="${AI_ENGINE}/redteam_ai_benchmark.py"
    if [ ! -f "${benchmark_script}" ]; then
        log_error "Benchmark script not found: ${benchmark_script}"
        exit 1
    fi
    
    # Verify Python can import the script
    log_info "Verifying benchmark script..."
    if ! "${VENV_PYTHON}" -c "import sys; sys.path.insert(0, '${AI_ENGINE}'); from redteam_ai_benchmark import RedTeamBenchmark" 2>/dev/null; then
        log_error "Failed to import benchmark module"
        log_info "Check: ${VENV_PYTHON} -m pip list | grep -i flask"
        exit 1
    fi
    
    log_success "Benchmark script verified"
    
    # Run benchmark
    log_info "Running 12 offensive security tests..."
    echo ""
    
    cd "${AI_ENGINE}"
    
    if "${VENV_PYTHON}" redteam_ai_benchmark.py run; then
        echo ""
        log_success "Benchmark completed"
        
        # Show latest results
        local latest_result
        latest_result=$(ls -t redteam_benchmark_data/results/benchmark_*.json 2>/dev/null | head -1)
        
        if [ -n "${latest_result}" ]; then
            echo ""
            log_info "Latest results: $(basename "${latest_result}")"
            "${VENV_PYTHON}" -c "
import json
with open('${latest_result}') as f:
    data = json.load(f)
    score = data.get('percentage', 0)
    print(f'Score: {score:.1f}%')
    print(f'Verdict: {data.get(\"verdict\", \"unknown\")}')
    print(f'Refused: {data.get(\"refused_count\", 0)}/12')
    print(f'Correct: {data.get(\"correct_count\", 0)}/12')
"
        fi
    else
        echo ""
        log_error "Benchmark failed"
        exit 1
    fi
    
    cd "${LAT5150_ROOT}"
}

cmd_improve() {
    log_section "Running Self-Improvement Session"
    
    # Ensure venv is active
    if ! venv_is_active; then
        log_info "Activating virtual environment..."
        venv_activate || {
            log_error "Failed to activate virtual environment"
            exit 1
        }
    fi
    
    # Check script exists
    local improve_script="${AI_ENGINE}/ai_self_improvement.py"
    if [ ! -f "${improve_script}" ]; then
        log_error "Self-improvement script not found: ${improve_script}"
        exit 1
    fi
    
    log_info "Starting automated improvement session..."
    log_info "Target: 80% benchmark score"
    log_info "Max cycles: 5"
    echo ""
    
    cd "${AI_ENGINE}"
    "${VENV_PYTHON}" ai_self_improvement.py run || {
        log_error "Self-improvement failed"
        exit 1
    }
    cd "${LAT5150_ROOT}"
}

cmd_start_all() {
    log_section "Starting All LAT5150 Services"

    local api_port
    api_port=$(get_api_port)

    service_start "${SERVICE_UNIFIED_API}"
    service_start "${TIMER_SELF_IMPROVE}"

    echo ""
    log_info "Access points:"
    log_info "  • API: http://localhost:${api_port}/api/self-awareness"
    log_info "  • Dashboard: ./lat5150.sh dashboard"
    log_info "  • API Port: ${api_port} (stored in .lat5150_api_port)"
    echo ""
}

cmd_stop_all() {
    log_section "Stopping All LAT5150 Services"
    
    check_sudo
    
    service_stop "${SERVICE_UNIFIED_API}"
    service_stop "${TIMER_SELF_IMPROVE}"
    
    log_success "All services stopped"
    echo ""
}

cmd_restart_all() {
    cmd_stop_all
    sleep 2
    cmd_start_all
}

cmd_status() {
    log_section "LAT5150 DRVMIL System Status"
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  LAT5150 DRVMIL - Tactical Intelligence System${NC}"
    echo -e "${BLUE}  Version: 2.0.2${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Virtual environment
    echo -e "${CYAN}Virtual Environment:${NC}"
    echo "─────────────────────────────────────────────────────────"
    
    if venv_exists; then
        echo -e "  ${GREEN}●${NC} Location: ${VENV_DIR}"
        if venv_is_active; then
            echo -e "  ${GREEN}●${NC} Status: ${GREEN}Active${NC}"
        else
            echo -e "  ${YELLOW}●${NC} Status: ${YELLOW}Inactive${NC} (run: venv)"
        fi
        local flask_ver
        flask_ver=$("${VENV_PYTHON}" -c 'import flask; print(flask.__version__)' 2>/dev/null || echo "ERROR")
        echo -e "  ${GREEN}●${NC} Flask: ${flask_ver}"
    else
        echo -e "  ${RED}●${NC} Status: ${RED}Not installed${NC}"
    fi
    
    echo ""
    
    # Services
    echo -e "${CYAN}System Services:${NC}"
    echo "─────────────────────────────────────────────────────────"
    
    if service_is_active "${SERVICE_UNIFIED_API}"; then
        echo -e "  ${GREEN}●${NC} Unified API: ${GREEN}Running${NC}"
    else
        echo -e "  ${RED}●${NC} Unified API: ${RED}Stopped${NC}"
    fi
    
    if service_is_active "${TIMER_SELF_IMPROVE}"; then
        echo -e "  ${GREEN}●${NC} Self-Improvement: ${GREEN}Active${NC}"
    else
        echo -e "  ${RED}●${NC} Self-Improvement: ${RED}Inactive${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

cmd_dashboard() {
    log_section "Launching Web Dashboard"
    
    if ! venv_is_active; then
        venv_activate || exit 1
    fi
    
    local dashboard_script="${AI_ENGINE}/ai_gui_dashboard.py"
    if [ ! -f "${dashboard_script}" ]; then
        log_error "Dashboard not found: ${dashboard_script}"
        exit 1
    fi
    
    log_info "Starting dashboard on port 5050..."
    log_info "Access at: http://localhost:5050"
    echo ""
    
    cd "${AI_ENGINE}"
    "${VENV_PYTHON}" ai_gui_dashboard.py
}

cmd_api() {
    log_section "Starting Unified Tactical API"

    local api_port
    api_port=$(get_api_port)

    if ! venv_is_active; then
        venv_activate || exit 1
    fi

    log_info "Starting API on port ${api_port}..."
    log_info "Access at: http://localhost:${api_port}/api/self-awareness"
    echo ""

    cd "${WEB_INTERFACE}"
    "${VENV_PYTHON}" unified_tactical_api.py --port "${api_port}"
}

cmd_test() {
    log_section "Running Integration Tests"
    
    if ! venv_is_active; then
        venv_activate || exit 1
    fi
    
    echo -e "${CYAN}[1/4] Testing Python syntax...${NC}"
    "${VENV_PYTHON}" -m py_compile "${AI_ENGINE}/enhanced_ai_engine.py" && log_success "enhanced_ai_engine.py" || log_fail "enhanced_ai_engine.py"
    "${VENV_PYTHON}" -m py_compile "${AI_ENGINE}/redteam_ai_benchmark.py" && log_success "redteam_ai_benchmark.py" || log_fail "redteam_ai_benchmark.py"
    
    echo ""
    echo -e "${CYAN}[2/4] Testing imports...${NC}"
    "${VENV_PYTHON}" -c "import sys; sys.path.insert(0, '${AI_ENGINE}'); from enhanced_ai_engine import EnhancedAIEngine" && log_success "EnhancedAIEngine" || log_fail "EnhancedAIEngine"
    
    echo ""
    echo -e "${CYAN}[3/4] Testing venv...${NC}"
    venv_is_active && log_success "Venv active" || log_fail "Venv not active"
    
    echo ""
    echo -e "${CYAN}[4/4] Testing services...${NC}"
    service_is_active "${SERVICE_UNIFIED_API}" && log_success "Unified API" || log_warn "Unified API not running"
    
    echo ""
    log_info "Tests complete!"
    echo ""
}

cmd_logs() {
    check_sudo
    
    if [ "${1:-api}" == "improve" ]; then
        log_info "Following self-improvement logs (Ctrl+C to stop)..."
        echo ""
        sudo journalctl -u "${SERVICE_SELF_IMPROVE}" -f
    else
        log_info "Following API logs (Ctrl+C to stop)..."
        echo ""
        sudo journalctl -u "${SERVICE_UNIFIED_API}" -f
    fi
}

cmd_uninstall() {
    log_section "Uninstalling LAT5150"
    
    check_sudo
    
    read -p "Remove all LAT5150 services? (y/N) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cancelled"
        exit 0
    fi
    
    service_remove "${SERVICE_UNIFIED_API}"
    service_remove "${SERVICE_SELF_IMPROVE}"
    
    log_success "Uninstall complete"
    echo ""
}

cmd_help() {
    cat <<EOF
${CYAN}╔════════════════════════════════════════════════════════════╗${NC}
${CYAN}║${NC} LAT5150 DRVMIL - Unified System Control
${CYAN}║${NC} Version: 2.0.2
${CYAN}╚════════════════════════════════════════════════════════════╝${NC}

${GREEN}Commands:${NC}
    install             Install everything (venv + services)
    start-all           Start all services
    stop-all            Stop all services
    status              Show system status
    benchmark           Run red team tests
    test                Run integration tests
    logs [api|improve]  View logs
    uninstall           Remove everything

${GREEN}Examples:${NC}
    ./lat5150.sh install
    ./lat5150.sh start-all
    ./lat5150.sh benchmark

${CYAN}═══════════════════════════════════════════════════════════${NC}
EOF
}

# Main
case "${1:-help}" in
    install) cmd_install ;;
    start-all|start) cmd_start_all ;;
    stop-all|stop) cmd_stop_all ;;
    restart-all|restart) cmd_restart_all ;;
    status) cmd_status ;;
    benchmark) cmd_benchmark ;;
    improve) cmd_improve ;;
    dashboard) cmd_dashboard ;;
    api) cmd_api ;;
    test) cmd_test ;;
    logs) cmd_logs "${2:-api}" ;;
    uninstall) cmd_uninstall ;;
    *) cmd_help ;;
esac
