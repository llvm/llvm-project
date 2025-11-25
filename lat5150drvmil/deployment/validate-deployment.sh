#!/bin/bash
###############################################################################
# LAT5150 DRVMIL Tactical AI Sub-Engine - Deployment Validation Script
# Version: 1.0.0
# Purpose: Comprehensive validation of all system components before deployment
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation results
PASSED=0
FAILED=0
WARNINGS=0

# Logging
LOG_FILE="/tmp/lat5150_validation_$(date +%Y%m%d_%H%M%S).log"

###############################################################################
# Helper Functions
###############################################################################

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

success() {
    log "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    log "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    log "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

info() {
    log "${BLUE}ℹ️  INFO${NC}: $1"
}

section() {
    log "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

###############################################################################
# Validation Tests
###############################################################################

validate_environment() {
    section "1. ENVIRONMENT VALIDATION"

    # Check OS
    if [ -f /etc/os-release ]; then
        OS_NAME=$(grep ^NAME= /etc/os-release | cut -d'"' -f2)
        success "Operating System: $OS_NAME"
    else
        fail "Cannot determine operating system"
    fi

    # Check kernel version
    KERNEL=$(uname -r)
    if [[ "$KERNEL" =~ ^[4-9]\. ]] || [[ "$KERNEL" =~ ^[1-9][0-9]\. ]]; then
        success "Kernel version: $KERNEL (>= 4.0)"
    else
        fail "Kernel version too old: $KERNEL (need >= 4.0)"
    fi

    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        success "Running with root privileges"
    else
        warn "Not running as root - some checks will be limited"
    fi

    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -ge 16 ]; then
        success "Total RAM: ${TOTAL_MEM}GB (>= 16GB)"
    elif [ "$TOTAL_MEM" -ge 8 ]; then
        warn "Total RAM: ${TOTAL_MEM}GB (recommended: >= 16GB)"
    else
        fail "Total RAM: ${TOTAL_MEM}GB (insufficient, need >= 16GB)"
    fi

    # Check disk space
    AVAIL_DISK=$(df -BG /home/user | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAIL_DISK" -ge 50 ]; then
        success "Available disk space: ${AVAIL_DISK}GB (>= 50GB)"
    else
        fail "Available disk space: ${AVAIL_DISK}GB (need >= 50GB)"
    fi
}

validate_dependencies() {
    section "2. DEPENDENCY VALIDATION"

    # Python 3
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        if [[ "$PYTHON_VERSION" =~ ^3\.[8-9]\. ]] || [[ "$PYTHON_VERSION" =~ ^3\.1[0-9]\. ]]; then
            success "Python: $PYTHON_VERSION (>= 3.8)"
        else
            fail "Python version too old: $PYTHON_VERSION (need >= 3.8)"
        fi
    else
        fail "Python 3 not found"
    fi

    # Flask
    if python3 -c "import flask" 2>/dev/null; then
        FLASK_VERSION=$(python3 -c "import flask; print(flask.__version__)")
        success "Flask: $FLASK_VERSION"
    else
        fail "Flask not installed (pip3 install flask)"
    fi

    # NumPy
    if python3 -c "import numpy" 2>/dev/null; then
        success "NumPy: installed"
    else
        warn "NumPy not installed (recommended for performance)"
    fi

    # Jina
    if python3 -c "import jina" 2>/dev/null; then
        success "Jina: installed (RAG support)"
    else
        warn "Jina not installed (RAG features disabled)"
    fi

    # Git
    if command -v git &> /dev/null; then
        success "Git: installed"
    else
        warn "Git not installed (version control unavailable)"
    fi

    # SystemD
    if command -v systemctl &> /dev/null; then
        success "SystemD: installed"
    else
        fail "SystemD not found (required for auto-start)"
    fi

    # SSH
    if command -v ssh &> /dev/null; then
        success "OpenSSH client: installed"
    else
        fail "SSH not found (required for VM tunneling)"
    fi

    if command -v sshd &> /dev/null; then
        success "OpenSSH server: installed"
    else
        warn "SSH server not found (required for VM access)"
    fi
}

validate_directory_structure() {
    section "3. DIRECTORY STRUCTURE VALIDATION"

    BASE_DIR="/home/user/LAT5150DRVMIL"

    REQUIRED_DIRS=(
        "$BASE_DIR"
        "$BASE_DIR/00-documentation"
        "$BASE_DIR/01-source"
        "$BASE_DIR/02-rag-embeddings-unified"
        "$BASE_DIR/03-web-interface"
        "$BASE_DIR/deployment"
    )

    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            success "Directory exists: $dir"
        else
            fail "Missing directory: $dir"
        fi
    done

    REQUIRED_FILES=(
        "$BASE_DIR/03-web-interface/secured_self_coding_api.py"
        "$BASE_DIR/03-web-interface/tactical_self_coding_ui.html"
        "$BASE_DIR/deployment/install-autostart.sh"
        "$BASE_DIR/deployment/deploy-vm-shortcuts.sh"
        "$BASE_DIR/01-source/debugging/nsa_device_reconnaissance_enhanced.py"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            if [ -x "$file" ] || [[ "$file" =~ \.(html|md)$ ]]; then
                success "File exists: $(basename $file)"
            else
                warn "File exists but not executable: $(basename $file)"
            fi
        else
            fail "Missing file: $file"
        fi
    done
}

validate_permissions() {
    section "4. PERMISSIONS VALIDATION"

    BASE_DIR="/home/user/LAT5150DRVMIL"

    # Check script executability
    SCRIPTS=(
        "$BASE_DIR/deployment/install-autostart.sh"
        "$BASE_DIR/deployment/deploy-vm-shortcuts.sh"
        "$BASE_DIR/deployment/configure_xen_bridge.sh"
        "$BASE_DIR/deployment/xen-vm-ssh-tunnel.sh"
        "$BASE_DIR/deployment/xen-vm-desktop/tactical-tunnel-autostart.sh"
        "$BASE_DIR/01-source/debugging/nsa_device_reconnaissance_enhanced.py"
    )

    for script in "${SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                success "Executable: $(basename $script)"
            else
                warn "Not executable: $(basename $script)"
                info "Fix with: chmod +x $script"
            fi
        fi
    done

    # Check ownership
    OWNER=$(stat -c '%U' "$BASE_DIR")
    if [ "$OWNER" = "user" ] || [ "$OWNER" = "root" ]; then
        success "Directory ownership: $OWNER"
    else
        warn "Unexpected ownership: $OWNER (expected: user or root)"
    fi
}

validate_systemd_service() {
    section "5. SYSTEMD SERVICE VALIDATION"

    SERVICE_FILE="/etc/systemd/system/lat5150-tactical.service"

    if [ -f "$SERVICE_FILE" ]; then
        success "Service file exists: $SERVICE_FILE"

        # Check if service is enabled
        if systemctl is-enabled lat5150-tactical.service &>/dev/null; then
            success "Service is enabled (auto-start on boot)"
        else
            warn "Service not enabled (run: sudo systemctl enable lat5150-tactical.service)"
        fi

        # Check if service is active
        if systemctl is-active lat5150-tactical.service &>/dev/null; then
            success "Service is running"

            # Check if port 5001 is listening
            if netstat -tln 2>/dev/null | grep -q ":5001"; then
                success "API listening on port 5001"
            else
                fail "Port 5001 not listening (service may have crashed)"
            fi
        else
            warn "Service not running (start with: sudo systemctl start lat5150-tactical.service)"
        fi
    else
        fail "Service file not installed: $SERVICE_FILE"
        info "Install with: cd deployment && sudo ./install-autostart.sh install"
    fi
}

validate_api_health() {
    section "6. API HEALTH VALIDATION"

    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        warn "curl not installed - skipping API health check"
        return
    fi

    # Check API health endpoint
    if systemctl is-active lat5150-tactical.service &>/dev/null; then
        HEALTH_RESPONSE=$(curl -s http://127.0.0.1:5001/health 2>/dev/null || echo "")

        if [ -n "$HEALTH_RESPONSE" ]; then
            success "API responding on http://127.0.0.1:5001"

            # Parse JSON response (if jq available)
            if command -v jq &> /dev/null; then
                STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.status' 2>/dev/null || echo "unknown")
                if [ "$STATUS" = "healthy" ]; then
                    success "API health status: $STATUS"

                    # Check feature flags
                    RAG=$(echo "$HEALTH_RESPONSE" | jq -r '.rag_enabled' 2>/dev/null || echo "unknown")
                    INT8=$(echo "$HEALTH_RESPONSE" | jq -r '.int8_enabled' 2>/dev/null || echo "unknown")
                    LEARNING=$(echo "$HEALTH_RESPONSE" | jq -r '.learning_enabled' 2>/dev/null || echo "unknown")

                    [ "$RAG" = "true" ] && success "RAG enabled" || warn "RAG disabled"
                    [ "$INT8" = "true" ] && success "INT8 enabled" || warn "INT8 disabled"
                    [ "$LEARNING" = "true" ] && success "Learning enabled" || warn "Learning disabled"
                else
                    fail "API health status: $STATUS"
                fi
            else
                info "Response: $HEALTH_RESPONSE"
            fi
        else
            fail "No response from API on http://127.0.0.1:5001"
        fi
    else
        warn "Service not running - skipping API health check"
    fi
}

validate_xen_integration() {
    section "7. XEN INTEGRATION VALIDATION"

    # Check if Xen is installed
    if command -v xl &> /dev/null; then
        success "Xen toolstack (xl) installed"

        # Check if running in Dom0
        if [ -f /proc/xen/capabilities ]; then
            CAPS=$(cat /proc/xen/capabilities)
            if echo "$CAPS" | grep -q "control_d"; then
                success "Running in Xen Dom0"
            else
                warn "Not running in Xen Dom0 (DomU or non-Xen system)"
            fi
        else
            warn "Cannot detect Xen capabilities"
        fi

        # Check for xenbr0
        if ip addr show xenbr0 &>/dev/null; then
            BRIDGE_IP=$(ip addr show xenbr0 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
            success "Xen bridge (xenbr0) configured: $BRIDGE_IP"
        else
            warn "Xen bridge (xenbr0) not found"
            info "VMs may need manual network configuration"
        fi
    else
        warn "Xen not installed (VM integration unavailable)"
    fi

    # Check VM desktop integration files
    VM_DESKTOP_DIR="/home/user/LAT5150DRVMIL/deployment/xen-vm-desktop"
    if [ -d "$VM_DESKTOP_DIR" ]; then
        success "VM desktop integration files present"

        if [ -f "$VM_DESKTOP_DIR/LAT5150-Tactical.desktop" ]; then
            success "Desktop launcher: LAT5150-Tactical.desktop"
        else
            fail "Missing desktop launcher"
        fi

        if [ -f "$VM_DESKTOP_DIR/tactical-tunnel-autostart.sh" ]; then
            success "Tunnel autostart script present"
        else
            fail "Missing tunnel autostart script"
        fi
    else
        fail "VM desktop integration directory missing"
    fi
}

validate_dsmil_system() {
    section "8. DSMIL DEVICE SYSTEM VALIDATION"

    # Check for /dev/dsmil
    if [ -e /dev/dsmil ]; then
        success "DSMIL device node exists: /dev/dsmil"

        # Check permissions
        DSMIL_PERMS=$(stat -c '%a' /dev/dsmil)
        if [ "$DSMIL_PERMS" = "666" ] || [ "$DSMIL_PERMS" = "660" ]; then
            success "DSMIL permissions: $DSMIL_PERMS (accessible)"
        else
            warn "DSMIL permissions: $DSMIL_PERMS (may need 666 or 660)"
        fi
    else
        warn "/dev/dsmil not found (hardware reconnaissance limited)"
        info "Load kernel module: sudo modprobe dsmil"
    fi

    # Check reconnaissance script
    RECON_SCRIPT="/home/user/LAT5150DRVMIL/01-source/debugging/nsa_device_reconnaissance_enhanced.py"
    if [ -f "$RECON_SCRIPT" ] && [ -x "$RECON_SCRIPT" ]; then
        success "Enhanced reconnaissance script ready"
    else
        fail "Reconnaissance script missing or not executable"
    fi

    # Check device documentation
    DEVICE_DOCS="/home/user/LAT5150DRVMIL/00-documentation/devices"
    if [ -d "$DEVICE_DOCS" ]; then
        DEVICE_COUNT=$(ls -1 "$DEVICE_DOCS"/device_*.md 2>/dev/null | wc -l)
        success "Device documentation: $DEVICE_COUNT devices documented"
    else
        warn "Device documentation directory missing"
    fi
}

validate_security() {
    section "9. SECURITY VALIDATION"

    # Check localhost binding
    if netstat -tln 2>/dev/null | grep ":5001" | grep -q "127.0.0.1"; then
        success "API bound to localhost only (127.0.0.1:5001)"
    elif netstat -tln 2>/dev/null | grep -q ":5001"; then
        fail "API not bound to localhost (security risk!)"
    else
        warn "API not running - cannot check binding"
    fi

    # Check SSH security
    if [ -f /etc/ssh/sshd_config ]; then
        if grep -q "^PermitRootLogin.*prohibit-password" /etc/ssh/sshd_config || \
           grep -q "^PermitRootLogin.*no" /etc/ssh/sshd_config; then
            success "SSH root login restricted"
        else
            warn "SSH allows root login (consider disabling)"
        fi

        if grep -q "^PasswordAuthentication no" /etc/ssh/sshd_config; then
            success "SSH password authentication disabled"
        else
            warn "SSH password authentication enabled (use keys)"
        fi
    fi

    # Check firewall
    if command -v iptables &> /dev/null && [ "$EUID" -eq 0 ]; then
        if iptables -L -n | grep -q "5001"; then
            success "Firewall rules configured for port 5001"
        else
            warn "No firewall rules for port 5001 (add protection)"
        fi
    fi

    # Check SELinux/AppArmor
    if command -v getenforce &> /dev/null; then
        SELINUX_STATUS=$(getenforce)
        if [ "$SELINUX_STATUS" = "Enforcing" ]; then
            success "SELinux: Enforcing"
        else
            warn "SELinux: $SELINUX_STATUS (consider enabling)"
        fi
    elif command -v aa-status &> /dev/null; then
        if aa-status --enabled 2>/dev/null; then
            success "AppArmor: Enabled"
        else
            warn "AppArmor: Disabled (consider enabling)"
        fi
    else
        warn "No mandatory access control (SELinux/AppArmor) detected"
    fi
}

validate_documentation() {
    section "10. DOCUMENTATION VALIDATION"

    DOCS=(
        "/home/user/LAT5150DRVMIL/README.md"
        "/home/user/LAT5150DRVMIL/DEPLOYMENT_GUIDE.md"
        "/home/user/LAT5150DRVMIL/TACTICAL_INTERFACE_GUIDE.md"
        "/home/user/LAT5150DRVMIL/TEMPEST_COMPLIANCE.md"
        "/home/user/LAT5150DRVMIL/XEN_INTEGRATION_GUIDE.md"
    )

    for doc in "${DOCS[@]}"; do
        if [ -f "$doc" ]; then
            success "Documentation: $(basename $doc)"
        else
            warn "Missing documentation: $(basename $doc)"
        fi
    done
}

###############################################################################
# Main Execution
###############################################################################

main() {
    clear
    log "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    log "${BLUE}║   LAT5150 DRVMIL Tactical AI Sub-Engine - Deployment Validation  ║${NC}"
    log "${BLUE}║   Version: 1.0.0                                                  ║${NC}"
    log "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    log ""
    info "Validation started: $(date)"
    info "Log file: $LOG_FILE"

    # Run all validation tests
    validate_environment
    validate_dependencies
    validate_directory_structure
    validate_permissions
    validate_systemd_service
    validate_api_health
    validate_xen_integration
    validate_dsmil_system
    validate_security
    validate_documentation

    # Summary
    section "VALIDATION SUMMARY"
    log ""
    log "  ${GREEN}✅ Passed:${NC}   $PASSED"
    log "  ${RED}❌ Failed:${NC}   $FAILED"
    log "  ${YELLOW}⚠️  Warnings:${NC} $WARNINGS"
    log ""

    # Overall status
    if [ $FAILED -eq 0 ]; then
        log "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
        log "${GREEN}║  ✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT  ║${NC}"
        log "${GREEN}╚═══════════════════════════════════════════════╝${NC}"

        if [ $WARNINGS -gt 0 ]; then
            log ""
            log "${YELLOW}Note: $WARNINGS warnings detected. Review recommended but not required.${NC}"
        fi

        exit 0
    else
        log "${RED}╔═══════════════════════════════════════════════════════════════╗${NC}"
        log "${RED}║  ❌ SYSTEM NOT READY - $FAILED CRITICAL ISSUES DETECTED  ║${NC}"
        log "${RED}╚═══════════════════════════════════════════════════════════════╝${NC}"
        log ""
        log "${RED}Review failures above and resolve before deployment.${NC}"
        log "Full log: $LOG_FILE"

        exit 1
    fi
}

# Run main function
main "$@"
