#!/bin/bash
#
# LAT5150 DRVMIL Unified Entry Point
# Launches the tmux-based operations suite with comprehensive environment setup
# Handles sudo requirements, path resolution, and dependency validation
#
# Usage:
#   ./lat5150_entrypoint.sh [session-name]
#
# Environment variables:
#   LAT5150_DEBUG=1    - Enable debug output
#   LAT5150_SKIP_VENV=1 - Skip venv setup (for testing)

set -euo pipefail

# ============================================================================
# Color Definitions
# ============================================================================
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Path Resolution (Absolute Paths)
# ============================================================================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LAT5150_ROOT="${SCRIPT_DIR}"
readonly VENV_DIR="${LAT5150_ROOT}/.venv"
readonly LAUNCHER="${LAT5150_ROOT}/scripts/lat5150_tmux_launcher.sh"
readonly ENSURE_VENV="${LAT5150_ROOT}/scripts/ensure_venv.sh"
readonly LAT_SCRIPT="${LAT5150_ROOT}/lat5150.sh"
readonly DSMIL_SCRIPT="${LAT5150_ROOT}/dsmil.py"

# Python paths
readonly VENV_PYTHON="${VENV_DIR}/bin/python3"
readonly VENV_PIP="${VENV_DIR}/bin/pip"

# AI Engine and Web Interface
readonly AI_ENGINE="${LAT5150_ROOT}/02-ai-engine"
readonly WEB_INTERFACE="${LAT5150_ROOT}/03-web-interface"

# ============================================================================
# Logging Functions
# ============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [ "${LAT5150_DEBUG:-0}" = "1" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" >&2
    fi
}

log_success() {
    echo -e "${GREEN}✓${NC} $*" >&2
}

log_fail() {
    echo -e "${RED}✗${NC} $*" >&2
}

# ============================================================================
# Validation Functions
# ============================================================================

# Check if tmux is installed
check_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        log_error "tmux is required but not installed"
        log_info "Install with: sudo apt-get install tmux"
        return 1
    fi
    log_debug "tmux found: $(command -v tmux)"
    return 0
}

# Check Python version
check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "python3 is required but not found"
        return 1
    fi

    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_debug "Python version: ${python_version}"

    # Check minimum version (3.7+)
    local major minor
    major=$(echo "${python_version}" | cut -d. -f1)
    minor=$(echo "${python_version}" | cut -d. -f2)

    if [ "${major}" -lt 3 ] || { [ "${major}" -eq 3 ] && [ "${minor}" -lt 7 ]; }; then
        log_error "Python 3.7+ required, found ${python_version}"
        return 1
    fi

    log_debug "Python version OK: ${python_version}"
    return 0
}

# Validate all required scripts exist and are executable
check_required_files() {
    local missing=0

    log_debug "Checking required files..."

    # Critical scripts
    local -a required_files=(
        "${LAUNCHER}"
        "${LAT_SCRIPT}"
        "${ENSURE_VENV}"
    )

    for file in "${required_files[@]}"; do
        if [ ! -f "${file}" ]; then
            log_fail "Missing: ${file}"
            missing=1
        elif [ ! -x "${file}" ]; then
            log_warn "Not executable: ${file}"
            chmod +x "${file}" 2>/dev/null && log_success "Fixed permissions: ${file}" || {
                log_error "Cannot set executable: ${file}"
                missing=1
            }
        else
            log_debug "Found: ${file}"
        fi
    done

    # Optional but recommended - DSMIL
    if [ ! -f "${DSMIL_SCRIPT}" ]; then
        log_debug "DSMIL script not found (optional): ${DSMIL_SCRIPT}"
    else
        # Verify DSMIL control centre can run (suppress output as it may show optional dependency warnings)
        log_debug "Verifying DSMIL control centre..."
        if python3 "${LAT5150_ROOT}/dsmil_control_centre.py" --check-only >/dev/null 2>&1; then
            log_debug "DSMIL control centre OK"
        else
            log_debug "DSMIL control centre has optional dependency warnings (non-critical)"
        fi
    fi

    return ${missing}
}

# Check if we have necessary permissions
check_permissions() {
    local user_id
    user_id=$(id -u)

    # Store original user if running via sudo
    if [ -n "${SUDO_USER:-}" ]; then
        export LAT5150_REAL_USER="${SUDO_USER}"
        export LAT5150_REAL_HOME=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
        log_debug "Running via sudo as ${SUDO_USER}"
    else
        export LAT5150_REAL_USER="${USER:-$(whoami)}"
        export LAT5150_REAL_HOME="${HOME:-$(eval echo ~$(whoami))}"
        log_debug "Running as ${LAT5150_REAL_USER}"
    fi

    # Warn about sudo requirements
    if [ "${user_id}" -ne 0 ]; then
        log_debug "Running as non-root user (UID ${user_id})"
        log_info "Note: Some operations may request sudo when needed:"
        log_info "  - Starting/stopping systemd services"
        log_info "  - Viewing system logs"
        log_info "  - DSMIL kernel module operations"
    fi

    return 0
}

# Setup virtual environment
setup_venv() {
    if [ "${LAT5150_SKIP_VENV:-0}" = "1" ]; then
        log_warn "Skipping venv setup (LAT5150_SKIP_VENV=1)"
        return 0
    fi

    log_info "Setting up Python virtual environment..."

    # Run ensure_venv script if available
    if [ -x "${ENSURE_VENV}" ]; then
        log_debug "Running: ${ENSURE_VENV}"
        if "${ENSURE_VENV}" 2>&1 | grep -v "Optional package.*could not be installed" >/dev/null; then
            log_success "Virtual environment ready"
        else
            log_debug "Virtual environment setup completed with optional warnings"
        fi
    else
        log_warn "ensure_venv.sh not found or not executable"
    fi

    # Verify venv exists
    if [ ! -d "${VENV_DIR}" ]; then
        log_error "Virtual environment not created at ${VENV_DIR}"
        log_info "Run: ./lat5150.sh install"
        return 1
    fi

    if [ ! -f "${VENV_PYTHON}" ]; then
        log_error "Python not found in venv: ${VENV_PYTHON}"
        return 1
    fi

    # Auto-install optional packages if needed
    log_debug "Checking optional packages..."
    local venv_pip="${VENV_DIR}/bin/pip"
    if [ -x "${venv_pip}" ]; then
        # Check if PyPI is accessible
        if python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple', timeout=3)" >/dev/null 2>&1; then
            # Install optional packages silently (package_name:import_name)
            local packages=(
                "duckduckgo-search:duckduckgo_search"
                "google-generativeai:google.generativeai"
                "langchain:langchain"
            )

            for pkg_pair in "${packages[@]}"; do
                local pkg_name="${pkg_pair%%:*}"
                local import_name="${pkg_pair##*:}"

                if ! "${VENV_PYTHON}" -c "import ${import_name}" >/dev/null 2>&1; then
                    log_debug "Installing optional package: ${pkg_name}"
                    "${venv_pip}" install -q "${pkg_name}" >/dev/null 2>&1 || log_debug "Could not install ${pkg_name} (optional)"
                fi
            done
        else
            log_debug "PyPI unreachable; skipping optional package installation"
        fi
    fi

    log_debug "Virtual environment verified at ${VENV_DIR}"
    return 0
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    log_info "Configuring environment variables..."

    # Core paths
    export LAT5150_ROOT
    export SCRIPT_DIR

    # Virtual environment
    export VIRTUAL_ENV="${VENV_DIR}"
    export LAT5150_VENV="${VENV_DIR}"

    # Python paths - ensure both are in PYTHONPATH
    if [ -z "${PYTHONPATH:-}" ]; then
        export PYTHONPATH="${AI_ENGINE}:${WEB_INTERFACE}"
    else
        export PYTHONPATH="${AI_ENGINE}:${WEB_INTERFACE}:${PYTHONPATH}"
    fi

    # Add venv bin to PATH (prepend to ensure it takes precedence)
    if [ -d "${VENV_DIR}/bin" ]; then
        export PATH="${VENV_DIR}/bin:${PATH}"
        log_debug "Added venv to PATH: ${VENV_DIR}/bin"
    fi

    # Session name
    export LAT5150_SESSION="${1:-lat5150-suite}"

    # Debugging
    if [ "${LAT5150_DEBUG:-0}" = "1" ]; then
        log_debug "Environment configured:"
        log_debug "  LAT5150_ROOT=${LAT5150_ROOT}"
        log_debug "  VIRTUAL_ENV=${VIRTUAL_ENV}"
        log_debug "  PYTHONPATH=${PYTHONPATH}"
        log_debug "  PATH=${PATH}"
        log_debug "  LAT5150_SESSION=${LAT5150_SESSION}"
    fi

    log_success "Environment configured"
    return 0
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

preflight_checks() {
    log_info "Running pre-flight checks..."

    local errors=0

    # Check tmux
    check_tmux || ((errors++))

    # Check Python
    check_python || ((errors++))

    # Check required files
    check_required_files || ((errors++))

    # Check permissions
    check_permissions || ((errors++))

    # Setup venv
    setup_venv || ((errors++))

    if [ ${errors} -gt 0 ]; then
        log_error "Pre-flight checks failed with ${errors} error(s)"
        return 1
    fi

    log_success "All pre-flight checks passed"
    return 0
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} LAT5150 DRVMIL - System Initialization"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Change to project root
    cd "${LAT5150_ROOT}" || {
        log_error "Failed to change to LAT5150_ROOT: ${LAT5150_ROOT}"
        exit 1
    }
    log_debug "Working directory: $(pwd)"

    # Run pre-flight checks
    if ! preflight_checks; then
        log_error "System not ready to launch"
        exit 1
    fi

    # Setup environment
    setup_environment "$@" || {
        log_error "Failed to configure environment"
        exit 1
    }

    # Optional: Build/Install DEB packages
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} DEB Package System (Optional)"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Would you like to build/install DEB packages before launching?${NC}"
    echo -e "  ${GREEN}1)${NC} Build DEB packages (4 packages)"
    echo -e "  ${GREEN}2)${NC} Build and Install DEB packages ${YELLOW}(requires root)${NC}"
    echo -e "  ${GREEN}3)${NC} Skip - Launch environment now"
    echo ""
    read -p "Select option [1-3, default=3]: " deb_choice
    deb_choice=${deb_choice:-3}

    case "$deb_choice" in
        1)
            echo ""
            log_info "Building DEB packages..."
            if [ -x "${LAT5150_ROOT}/packaging/build-all-debs.sh" ]; then
                cd "${LAT5150_ROOT}/packaging" && ./build-all-debs.sh
                log_success "DEB packages built successfully!"
                echo ""
                read -p "Press ENTER to continue..."
            else
                log_error "Build script not found: ${LAT5150_ROOT}/packaging/build-all-debs.sh"
            fi
            cd "${LAT5150_ROOT}"
            ;;
        2)
            echo ""
            log_info "Building and installing DEB packages..."
            if [ -x "${LAT5150_ROOT}/packaging/build-all-debs.sh" ]; then
                cd "${LAT5150_ROOT}/packaging" && ./build-all-debs.sh
                if [ $? -eq 0 ]; then
                    log_info "Build complete. Installing packages (requires root)..."
                    sudo ./install-all-debs.sh
                    if [ $? -eq 0 ]; then
                        log_success "DEB packages installed successfully!"
                        echo ""
                        log_info "Running verification..."
                        ./verify-installation.sh
                    else
                        log_error "Installation failed"
                    fi
                else
                    log_error "Build failed"
                fi
                echo ""
                read -p "Press ENTER to continue..."
            else
                log_error "Build script not found: ${LAT5150_ROOT}/packaging/build-all-debs.sh"
            fi
            cd "${LAT5150_ROOT}"
            ;;
        3|*)
            log_info "Skipping DEB package operations"
            ;;
    esac

    echo ""
    log_info "Launching LAT5150 tmux session: ${LAT5150_SESSION}"
    log_info "Press Ctrl+B, then D to detach from session"
    echo ""

    # Execute launcher with all environment variables
    log_debug "Executing: ${LAUNCHER} ${LAT5150_SESSION}"

    # Use exec to replace this process with the launcher
    exec "${LAUNCHER}" "${LAT5150_SESSION}"
}

# ============================================================================
# Execute
# ============================================================================

# Catch errors and provide helpful messages
trap 'log_error "Entrypoint failed at line $LINENO. Enable debug with: LAT5150_DEBUG=1 $0"' ERR

# Run main
main "$@"
