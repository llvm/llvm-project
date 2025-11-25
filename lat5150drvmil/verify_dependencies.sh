#!/bin/bash
#
# LAT5150 DRVMIL - Dependency and Build Order Verification Script
# Verifies all dependencies and build order for complete system deployment
#
# Usage:
#   ./verify_dependencies.sh [--fix]
#
# Options:
#   --fix    Attempt to install missing dependencies
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAT5150_ROOT="${SCRIPT_DIR}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

FIX_MODE=false
if [ "$1" == "--fix" ]; then
    FIX_MODE=true
fi

ISSUES_FOUND=0

log_check() {
    echo -e "${CYAN}[CHECK]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}  ✓${NC} $1"
}

log_fail() {
    echo -e "${RED}  ✗${NC} $1"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

log_warn() {
    echo -e "${YELLOW}  ⚠${NC} $1"
}

log_info() {
    echo -e "    $1"
}

log_section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Phase 1: System prerequisites
check_system_prerequisites() {
    log_section "Phase 1: System Prerequisites"

    # Check Python
    log_check "Python 3.10+"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            log_pass "Python $PYTHON_VERSION"
        else
            log_fail "Python $PYTHON_VERSION (need 3.10+)"
            log_info "Install: sudo apt install python3.10"
        fi
    else
        log_fail "Python not found"
        log_info "Install: sudo apt install python3 python3-pip"

        if [ "$FIX_MODE" = true ]; then
            log_info "Attempting to install Python..."
            sudo apt update && sudo apt install -y python3 python3-pip python3-venv
        fi
    fi

    # Check pip
    log_check "pip3"
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | awk '{print $2}')
        log_pass "pip $PIP_VERSION"
    else
        log_fail "pip3 not found"
        log_info "Install: sudo apt install python3-pip"

        if [ "$FIX_MODE" = true ]; then
            sudo apt install -y python3-pip
        fi
    fi

    # Check git
    log_check "Git"
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        log_pass "Git $GIT_VERSION"
    else
        log_fail "Git not found"
        log_info "Install: sudo apt install git"
    fi

    # Check curl
    log_check "curl"
    if command -v curl &> /dev/null; then
        log_pass "curl installed"
    else
        log_fail "curl not found"
        log_info "Install: sudo apt install curl"
    fi
}

# Phase 2: Python dependencies
check_python_dependencies() {
    log_section "Phase 2: Python Dependencies"

    # Core AI dependencies
    log_check "Flask"
    if python3 -c "import flask" 2>/dev/null; then
        FLASK_VERSION=$(python3 -c "import flask; print(flask.__version__)" 2>/dev/null)
        log_pass "Flask $FLASK_VERSION"
    else
        log_fail "Flask not installed"
        log_info "Install: pip3 install flask"

        if [ "$FIX_MODE" = true ]; then
            pip3 install flask
        fi
    fi

    log_check "Flask-CORS"
    if python3 -c "import flask_cors" 2>/dev/null; then
        log_pass "Flask-CORS installed"
    else
        log_fail "Flask-CORS not installed"
        log_info "Install: pip3 install flask-cors"

        if [ "$FIX_MODE" = true ]; then
            pip3 install flask-cors
        fi
    fi

    # Optional but recommended
    log_check "PyTorch (optional)"
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_pass "PyTorch $TORCH_VERSION"
    else
        log_warn "PyTorch not installed (needed for Heretic)"
        log_info "Install: pip3 install torch"
    fi

    log_check "Transformers (optional)"
    if python3 -c "import transformers" 2>/dev/null; then
        log_pass "Transformers installed"
    else
        log_warn "Transformers not installed (needed for Heretic)"
        log_info "Install: pip3 install transformers"
    fi

    log_check "Optuna (optional)"
    if python3 -c "import optuna" 2>/dev/null; then
        log_pass "Optuna installed"
    else
        log_warn "Optuna not installed (needed for Heretic optimization)"
        log_info "Install: pip3 install optuna"
    fi

    log_check "Pydantic Settings (optional)"
    if python3 -c "import pydantic_settings" 2>/dev/null; then
        log_pass "Pydantic Settings installed"
    else
        log_warn "Pydantic Settings not installed (needed for Heretic)"
        log_info "Install: pip3 install pydantic-settings"
    fi
}

# Phase 3: MCP Server dependencies
check_mcp_dependencies() {
    log_section "Phase 3: MCP Server Dependencies"

    # Check Rust
    log_check "Rust (optional)"
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        log_pass "Rust $RUST_VERSION"
    else
        log_warn "Rust not installed (needed for some MCP servers)"
        log_info "Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    fi

    # Check uv/uvx
    log_check "uvx (for Atomic Red Team MCP)"
    if command -v uvx &> /dev/null; then
        log_pass "uvx installed"
    else
        log_fail "uvx not found (needed for Atomic Red Team)"
        log_info "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        log_info "Then: export PATH=\"\$HOME/.local/bin:\$PATH\""

        if [ "$FIX_MODE" = true ]; then
            log_info "Attempting to install uvx..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi

    # Test atomic-red-team-mcp
    if command -v uvx &> /dev/null; then
        log_check "Atomic Red Team MCP"
        if uvx atomic-red-team-mcp --help &> /dev/null; then
            log_pass "atomic-red-team-mcp available"
        else
            log_warn "atomic-red-team-mcp not tested (may need first run)"
        fi
    fi
}

# Phase 4: File structure verification
check_file_structure() {
    log_section "Phase 4: File Structure Verification"

    # Core files
    log_check "Core Python modules"

    [ -f "${LAT5150_ROOT}/02-ai-engine/enhanced_ai_engine.py" ] && log_pass "enhanced_ai_engine.py" || log_fail "enhanced_ai_engine.py missing"
    [ -f "${LAT5150_ROOT}/02-ai-engine/redteam_ai_benchmark.py" ] && log_pass "redteam_ai_benchmark.py" || log_fail "redteam_ai_benchmark.py missing"
    [ -f "${LAT5150_ROOT}/02-ai-engine/ai_self_improvement.py" ] && log_pass "ai_self_improvement.py" || log_fail "ai_self_improvement.py missing"
    [ -f "${LAT5150_ROOT}/02-ai-engine/heretic_abliteration.py" ] && log_pass "heretic_abliteration.py" || log_fail "heretic_abliteration.py missing"
    [ -f "${LAT5150_ROOT}/02-ai-engine/atomic_red_team_api.py" ] && log_pass "atomic_red_team_api.py" || log_fail "atomic_red_team_api.py missing"

    log_check "Web interface"
    [ -f "${LAT5150_ROOT}/03-web-interface/unified_tactical_api.py" ] && log_pass "unified_tactical_api.py" || log_fail "unified_tactical_api.py missing"
    [ -f "${LAT5150_ROOT}/03-web-interface/capability_registry.py" ] && log_pass "capability_registry.py" || log_fail "capability_registry.py missing"

    log_check "Deployment scripts"
    [ -f "${LAT5150_ROOT}/deployment/install-unified-api-autostart.sh" ] && log_pass "install-unified-api-autostart.sh" || log_fail "install-unified-api-autostart.sh missing"
    [ -f "${LAT5150_ROOT}/deployment/install-self-improvement-timer.sh" ] && log_pass "install-self-improvement-timer.sh" || log_fail "install-self-improvement-timer.sh missing"

    log_check "SystemD service files"
    [ -f "${LAT5150_ROOT}/deployment/systemd/lat5150-unified-api.service" ] && log_pass "lat5150-unified-api.service" || log_fail "lat5150-unified-api.service missing"
    [ -f "${LAT5150_ROOT}/deployment/systemd/lat5150-self-improvement.service" ] && log_pass "lat5150-self-improvement.service" || log_fail "lat5150-self-improvement.service missing"
    [ -f "${LAT5150_ROOT}/deployment/systemd/lat5150-self-improvement.timer" ] && log_pass "lat5150-self-improvement.timer" || log_fail "lat5150-self-improvement.timer missing"

    log_check "Root launcher"
    [ -f "${LAT5150_ROOT}/lat5150.sh" ] && log_pass "lat5150.sh" || log_fail "lat5150.sh missing"
    [ -x "${LAT5150_ROOT}/lat5150.sh" ] && log_pass "lat5150.sh executable" || log_fail "lat5150.sh not executable (chmod +x)"

    log_check "Documentation"
    [ -f "${LAT5150_ROOT}/LAT5150_INTEGRATION_GUIDE.md" ] && log_pass "LAT5150_INTEGRATION_GUIDE.md" || log_fail "LAT5150_INTEGRATION_GUIDE.md missing"
    [ -f "${LAT5150_ROOT}/deployment/AUTO_INSTALL_README.md" ] && log_pass "AUTO_INSTALL_README.md" || log_fail "AUTO_INSTALL_README.md missing"
    [ -f "${LAT5150_ROOT}/deployment/AUTO_IMPROVEMENT_README.md" ] && log_pass "AUTO_IMPROVEMENT_README.md" || log_fail "AUTO_IMPROVEMENT_README.md missing"
}

# Phase 5: Python module syntax verification
check_python_syntax() {
    log_section "Phase 5: Python Module Syntax Verification"

    log_check "Compile-checking Python modules"

    python3 -m py_compile "${LAT5150_ROOT}/02-ai-engine/enhanced_ai_engine.py" 2>&1 && log_pass "enhanced_ai_engine.py" || log_fail "enhanced_ai_engine.py syntax error"
    python3 -m py_compile "${LAT5150_ROOT}/02-ai-engine/redteam_ai_benchmark.py" 2>&1 && log_pass "redteam_ai_benchmark.py" || log_fail "redteam_ai_benchmark.py syntax error"
    python3 -m py_compile "${LAT5150_ROOT}/02-ai-engine/ai_self_improvement.py" 2>&1 && log_pass "ai_self_improvement.py" || log_fail "ai_self_improvement.py syntax error"
    python3 -m py_compile "${LAT5150_ROOT}/03-web-interface/unified_tactical_api.py" 2>&1 && log_pass "unified_tactical_api.py" || log_fail "unified_tactical_api.py syntax error"
}

# Phase 6: Import verification
check_python_imports() {
    log_section "Phase 6: Python Import Verification"

    log_check "Testing module imports"

    python3 -c "import sys; sys.path.insert(0, '${LAT5150_ROOT}/02-ai-engine'); from enhanced_ai_engine import EnhancedAIEngine" 2>&1 && log_pass "EnhancedAIEngine" || log_fail "EnhancedAIEngine import failed"
    python3 -c "import sys; sys.path.insert(0, '${LAT5150_ROOT}/02-ai-engine'); from redteam_ai_benchmark import RedTeamBenchmark" 2>&1 && log_pass "RedTeamBenchmark" || log_fail "RedTeamBenchmark import failed"
    python3 -c "import sys; sys.path.insert(0, '${LAT5150_ROOT}/02-ai-engine'); from ai_self_improvement import AISelfImprovement" 2>&1 && log_pass "AISelfImprovement" || log_fail "AISelfImprovement import failed"
    python3 -c "import sys; sys.path.insert(0, '${LAT5150_ROOT}/02-ai-engine'); from atomic_red_team_api import AtomicRedTeamAPI" 2>&1 && log_pass "AtomicRedTeamAPI" || log_fail "AtomicRedTeamAPI import failed"
}

# Phase 7: Service installation status
check_service_status() {
    log_section "Phase 7: Service Installation Status"

    log_check "SystemD services"

    if [ -f "/etc/systemd/system/lat5150-unified-api.service" ]; then
        log_pass "lat5150-unified-api.service installed"

        if systemctl is-enabled lat5150-unified-api.service &> /dev/null; then
            log_pass "  Enabled on boot"
        else
            log_warn "  Not enabled on boot"
        fi

        if systemctl is-active lat5150-unified-api.service &> /dev/null; then
            log_pass "  Currently running"
        else
            log_warn "  Not currently running"
        fi
    else
        log_warn "lat5150-unified-api.service not installed"
        log_info "Install: cd deployment && sudo ./install-unified-api-autostart.sh install"
    fi

    if [ -f "/etc/systemd/system/lat5150-self-improvement.timer" ]; then
        log_pass "lat5150-self-improvement.timer installed"

        if systemctl is-enabled lat5150-self-improvement.timer &> /dev/null; then
            log_pass "  Enabled on boot"
        else
            log_warn "  Not enabled on boot"
        fi

        if systemctl is-active lat5150-self-improvement.timer &> /dev/null; then
            log_pass "  Currently active"
        else
            log_warn "  Not currently active"
        fi
    else
        log_warn "lat5150-self-improvement.timer not installed"
        log_info "Install: cd deployment && sudo ./install-self-improvement-timer.sh install"
    fi
}

# Phase 8: API health check
check_api_health() {
    log_section "Phase 8: API Health Check"

    log_check "Unified Tactical API"

    if curl -s http://localhost:80/api/self-awareness > /dev/null 2>&1; then
        log_pass "API responding on port 80"

        # Check components
        api_response=$(curl -s http://localhost:80/api/self-awareness 2>/dev/null)
        if [ -n "$api_response" ]; then
            log_info "Checking components..."
            echo "$api_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    components = data.get('components', {})
    for name, status in components.items():
        print(f'    {name}: {status}')

    caps = data.get('legacy_capabilities', {})
    if caps:
        total = caps.get('total_capabilities', 0)
        print(f'    Total capabilities: {total}')
except:
    pass
" 2>/dev/null
        fi
    else
        log_warn "API not responding on port 80"
        log_info "Start: sudo systemctl start lat5150-unified-api"
    fi
}

# Build order recommendations
show_build_order() {
    log_section "Build Order Recommendations"

    echo -e "${CYAN}Phase 1: System Prerequisites${NC}"
    echo "  1. Install Python 3.10+"
    echo "  2. Install pip3"
    echo "  3. Install git, curl"
    echo ""

    echo -e "${CYAN}Phase 2: Python Dependencies${NC}"
    echo "  4. Install Flask: pip3 install flask flask-cors"
    echo "  5. Install Heretic deps (optional): pip3 install torch transformers optuna pydantic-settings"
    echo ""

    echo -e "${CYAN}Phase 3: MCP Dependencies${NC}"
    echo "  6. Install Rust (optional): curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  7. Install uvx: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""

    echo -e "${CYAN}Phase 4: Deploy Services${NC}"
    echo "  8. cd deployment && sudo ./install-unified-api-autostart.sh install"
    echo "  9. cd deployment && sudo ./install-self-improvement-timer.sh install"
    echo "  10. cd deployment && ./setup-shell-integration.sh"
    echo ""

    echo -e "${CYAN}Phase 5: Verification${NC}"
    echo "  11. ./lat5150.sh test"
    echo "  12. ./lat5150.sh status"
    echo ""
}

# Main execution
main() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  LAT5150 DRVMIL - Dependency & Build Order Verification${NC}"
    echo -e "${CYAN}  Version: 2.0.0${NC}"
    if [ "$FIX_MODE" = true ]; then
        echo -e "${YELLOW}  Mode: Fix (will attempt to install missing dependencies)${NC}"
    else
        echo -e "${CYAN}  Mode: Check only (use --fix to auto-install)${NC}"
    fi
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    check_system_prerequisites
    check_python_dependencies
    check_mcp_dependencies
    check_file_structure
    check_python_syntax
    check_python_imports
    check_service_status
    check_api_health

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ $ISSUES_FOUND -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! System is ready.${NC}"
        echo ""
        echo "Next steps:"
        echo "  • Start services: ./lat5150.sh start-all"
        echo "  • Check status: ./lat5150.sh status"
        echo "  • Run benchmark: ./lat5150.sh benchmark"
    else
        echo -e "${RED}✗ Found $ISSUES_FOUND issue(s)${NC}"
        echo ""
        if [ "$FIX_MODE" = false ]; then
            echo "Run with --fix to attempt automatic fixes:"
            echo "  ./verify_dependencies.sh --fix"
            echo ""
        fi
        show_build_order
    fi

    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    exit $ISSUES_FOUND
}

main
