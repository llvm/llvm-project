#!/bin/bash
###############################################################################
# LAT5150 DRVMIL Tactical AI Sub-Engine - Master Production Deployment Script
# Version: 1.0.0
# Purpose: Orchestrate complete production deployment
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/lat5150_deployment_$(date +%Y%m%d_%H%M%S).log"

# Deployment options
SKIP_VALIDATION=false
SKIP_BACKUP=false
DEPLOY_VMS=false
VM_IPS=""
RUN_DSMIL_SCAN=false
AUTO_YES=false

###############################################################################
# Helper Functions
###############################################################################

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

info() {
    log "${BLUE}[INFO]${NC} $1"
}

success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    log "${RED}[ERROR]${NC} $1"
}

warn() {
    log "${YELLOW}[WARNING]${NC} $1"
}

section() {
    log ""
    log "${MAGENTA}╔════════════════════════════════════════════════════════════════════╗${NC}"
    log "${MAGENTA}║  $1${NC}"
    log "${MAGENTA}╚════════════════════════════════════════════════════════════════════╝${NC}"
}

prompt_continue() {
    if [ "$AUTO_YES" = true ]; then
        return 0
    fi

    read -p "$(echo -e ${CYAN}"Continue? [y/N]: "${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error "Deployment aborted by user"
        exit 1
    fi
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root"
        error "Please run: sudo $0 $@"
        exit 1
    fi
}

###############################################################################
# Deployment Steps
###############################################################################

step_validate() {
    if [ "$SKIP_VALIDATION" = true ]; then
        warn "Skipping validation (--skip-validation flag)"
        return 0
    fi

    section "STEP 1: SYSTEM VALIDATION"

    VALIDATE_SCRIPT="$SCRIPT_DIR/validate-deployment.sh"

    if [ ! -f "$VALIDATE_SCRIPT" ]; then
        error "Validation script not found: $VALIDATE_SCRIPT"
        return 1
    fi

    info "Running pre-deployment validation..."
    if bash "$VALIDATE_SCRIPT"; then
        success "Validation passed"
        return 0
    else
        error "Validation failed"
        warn "Fix issues above or use --skip-validation to proceed anyway"
        return 1
    fi
}

step_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        warn "Skipping backup (--skip-backup flag)"
        return 0
    fi

    section "STEP 2: BACKUP CURRENT SYSTEM"

    BACKUP_DIR="/backup/LAT5150DRVMIL_$(date +%Y%m%d_%H%M%S)"

    info "Creating backup in: $BACKUP_DIR"

    if mkdir -p "$BACKUP_DIR"; then
        # Backup main directory
        info "Backing up main directory..."
        rsync -av --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
              "$BASE_DIR/" "$BACKUP_DIR/" >> "$LOG_FILE" 2>&1

        # Backup systemd service if exists
        if [ -f /etc/systemd/system/lat5150-tactical.service ]; then
            info "Backing up SystemD service..."
            cp /etc/systemd/system/lat5150-tactical.service "$BACKUP_DIR/"
        fi

        # Create backup manifest
        cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" << EOF
LAT5150 DRVMIL Backup
Created: $(date)
Hostname: $(hostname)
User: $(whoami)

Backup Contents:
- Application files: $BASE_DIR
- SystemD service: /etc/systemd/system/lat5150-tactical.service

Restore Instructions:
1. Stop service: sudo systemctl stop lat5150-tactical.service
2. Restore files: sudo rsync -av $BACKUP_DIR/ $BASE_DIR/
3. Restore service: sudo cp $BACKUP_DIR/lat5150-tactical.service /etc/systemd/system/
4. Reload daemon: sudo systemctl daemon-reload
5. Start service: sudo systemctl start lat5150-tactical.service
EOF

        success "Backup created: $BACKUP_DIR"
        info "Backup manifest: $BACKUP_DIR/BACKUP_MANIFEST.txt"
    else
        error "Failed to create backup directory"
        return 1
    fi
}

step_install_dependencies() {
    section "STEP 3: INSTALL DEPENDENCIES"

    info "Checking Python dependencies..."

    MISSING_DEPS=()

    # Check Flask
    if ! python3 -c "import flask" 2>/dev/null; then
        MISSING_DEPS+=("flask")
    fi

    # Check NumPy
    if ! python3 -c "import numpy" 2>/dev/null; then
        MISSING_DEPS+=("numpy")
    fi

    # Check SciPy
    if ! python3 -c "import scipy" 2>/dev/null; then
        MISSING_DEPS+=("scipy")
    fi

    # Check Jina
    if ! python3 -c "import jina" 2>/dev/null; then
        MISSING_DEPS+=("jina")
    fi

    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        info "Installing missing dependencies: ${MISSING_DEPS[*]}"

        if pip3 install "${MISSING_DEPS[@]}" >> "$LOG_FILE" 2>&1; then
            success "Dependencies installed successfully"
        else
            error "Failed to install dependencies"
            error "Check log: $LOG_FILE"
            return 1
        fi
    else
        success "All dependencies already installed"
    fi
}

step_install_service() {
    section "STEP 4: INSTALL SYSTEMD SERVICE"

    INSTALL_SCRIPT="$SCRIPT_DIR/install-autostart.sh"

    if [ ! -f "$INSTALL_SCRIPT" ]; then
        error "Install script not found: $INSTALL_SCRIPT"
        return 1
    fi

    info "Installing SystemD service..."

    # Check if service already installed
    if systemctl is-enabled lat5150-tactical.service &>/dev/null; then
        warn "Service already installed"
        info "Reinstalling..."

        if bash "$INSTALL_SCRIPT" remove >> "$LOG_FILE" 2>&1; then
            info "Removed existing service"
        fi
    fi

    if bash "$INSTALL_SCRIPT" install >> "$LOG_FILE" 2>&1; then
        success "Service installed successfully"

        # Verify service is running
        sleep 2
        if systemctl is-active lat5150-tactical.service &>/dev/null; then
            success "Service is running"
        else
            error "Service failed to start"
            error "Check status: sudo systemctl status lat5150-tactical.service"
            return 1
        fi
    else
        error "Failed to install service"
        error "Check log: $LOG_FILE"
        return 1
    fi
}

step_verify_api() {
    section "STEP 5: VERIFY API HEALTH"

    info "Waiting for API to be ready..."

    MAX_ATTEMPTS=30
    ATTEMPT=0

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s --max-time 2 http://127.0.0.1:5001/health >/dev/null 2>&1; then
            success "API is responding"

            # Check health status
            HEALTH_DATA=$(curl -s http://127.0.0.1:5001/health)
            if command -v jq &>/dev/null; then
                STATUS=$(echo "$HEALTH_DATA" | jq -r '.status')
                if [ "$STATUS" = "healthy" ]; then
                    success "API health status: $STATUS"

                    # Display feature status
                    RAG=$(echo "$HEALTH_DATA" | jq -r '.rag_enabled')
                    INT8=$(echo "$HEALTH_DATA" | jq -r '.int8_enabled')
                    LEARNING=$(echo "$HEALTH_DATA" | jq -r '.learning_enabled')

                    info "Features:"
                    info "  - RAG:      $RAG"
                    info "  - INT8:     $INT8"
                    info "  - Learning: $LEARNING"

                    return 0
                else
                    warn "API health status: $STATUS"
                fi
            else
                info "API response: $HEALTH_DATA"
                return 0
            fi
        fi

        ((ATTEMPT++))
        sleep 1
        echo -n "."
    done

    echo ""
    error "API failed to respond after $MAX_ATTEMPTS attempts"
    return 1
}

step_deploy_vms() {
    if [ "$DEPLOY_VMS" = false ]; then
        warn "Skipping VM deployment (use --deploy-vms to enable)"
        return 0
    fi

    section "STEP 6: DEPLOY VM SHORTCUTS"

    if [ -z "$VM_IPS" ]; then
        error "No VM IPs specified (use --vm-ips)"
        return 1
    fi

    DEPLOY_SCRIPT="$SCRIPT_DIR/deploy-vm-shortcuts.sh"

    if [ ! -f "$DEPLOY_SCRIPT" ]; then
        error "Deploy script not found: $DEPLOY_SCRIPT"
        return 1
    fi

    info "Deploying to VMs: $VM_IPS"

    IFS=',' read -ra VM_ARRAY <<< "$VM_IPS"

    for VM_IP in "${VM_ARRAY[@]}"; do
        info "Deploying to VM: $VM_IP"

        if bash "$DEPLOY_SCRIPT" "$VM_IP" >> "$LOG_FILE" 2>&1; then
            success "Deployed to $VM_IP"
        else
            error "Failed to deploy to $VM_IP"
            warn "Check log: $LOG_FILE"
        fi
    done
}

step_run_dsmil_scan() {
    if [ "$RUN_DSMIL_SCAN" = false ]; then
        warn "Skipping DSMIL scan (use --scan-dsmil to enable)"
        return 0
    fi

    section "STEP 7: RUN DSMIL HARDWARE SCAN"

    RECON_SCRIPT="$BASE_DIR/01-source/debugging/nsa_device_reconnaissance_enhanced.py"

    if [ ! -f "$RECON_SCRIPT" ]; then
        error "Reconnaissance script not found: $RECON_SCRIPT"
        return 1
    fi

    if [ ! -e /dev/dsmil ]; then
        warn "/dev/dsmil not found - scan may have limited functionality"
    fi

    info "Running enhanced DSMIL reconnaissance..."
    info "This may take several minutes..."

    if python3 "$RECON_SCRIPT" >> "$LOG_FILE" 2>&1; then
        success "DSMIL scan completed"

        # Find latest results file
        RESULTS_FILE=$(ls -t "$BASE_DIR"/nsa_reconnaissance_enhanced_*.json 2>/dev/null | head -1)

        if [ -n "$RESULTS_FILE" ]; then
            info "Results saved: $RESULTS_FILE"

            if command -v jq &>/dev/null; then
                RESPONSIVE=$(jq -r '.responsive_devices' "$RESULTS_FILE")
                NPU_DETECTED=$(jq -r '.npu_hardware_detected | length' "$RESULTS_FILE")
                NPU_RELATED=$(jq -r '.npu_related_devices' "$RESULTS_FILE")

                info "Summary:"
                info "  - Responsive devices: $RESPONSIVE"
                info "  - NPU hardware detected: $NPU_DETECTED"
                info "  - NPU-related devices: $NPU_RELATED"
            fi
        fi
    else
        error "DSMIL scan failed"
        error "Check log: $LOG_FILE"
        return 1
    fi
}

step_final_verification() {
    section "FINAL VERIFICATION"

    info "Running final system checks..."

    CHECKS_PASSED=true

    # Check 1: Service running
    info "Checking service status..."
    if systemctl is-active lat5150-tactical.service &>/dev/null; then
        success "Service is running"
    else
        error "Service is not running"
        CHECKS_PASSED=false
    fi

    # Check 2: API responding
    info "Checking API health..."
    if curl -s --max-time 2 http://127.0.0.1:5001/health >/dev/null 2>&1; then
        success "API is responding"
    else
        error "API is not responding"
        CHECKS_PASSED=false
    fi

    # Check 3: Localhost binding
    info "Checking network binding..."
    if netstat -tln 2>/dev/null | grep -q "127.0.0.1:5001"; then
        success "API bound to localhost (secure)"
    else
        error "API not bound to localhost correctly"
        CHECKS_PASSED=false
    fi

    # Check 4: Service enabled
    info "Checking auto-start configuration..."
    if systemctl is-enabled lat5150-tactical.service &>/dev/null; then
        success "Service will start on boot"
    else
        error "Service not enabled for auto-start"
        CHECKS_PASSED=false
    fi

    if [ "$CHECKS_PASSED" = true ]; then
        return 0
    else
        return 1
    fi
}

###############################################################################
# Main Deployment
###############################################################################

show_usage() {
    cat << EOF
LAT5150 DRVMIL Tactical AI Sub-Engine - Master Production Deployment

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -y, --yes               Auto-answer yes to all prompts
    --skip-validation       Skip pre-deployment validation
    --skip-backup           Skip backup creation
    --deploy-vms            Deploy shortcuts to VMs
    --vm-ips IP1,IP2,...    Comma-separated VM IPs for deployment
    --scan-dsmil            Run DSMIL hardware scan after deployment

Examples:
    # Standard deployment
    sudo $0

    # Deploy with VM integration
    sudo $0 --deploy-vms --vm-ips 192.168.100.10,192.168.100.11

    # Full deployment with all options
    sudo $0 -y --deploy-vms --vm-ips 192.168.100.10 --scan-dsmil

    # Quick deployment (skip validation and backup)
    sudo $0 --skip-validation --skip-backup

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -y|--yes)
                AUTO_YES=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --deploy-vms)
                DEPLOY_VMS=true
                shift
                ;;
            --vm-ips)
                VM_IPS="$2"
                shift 2
                ;;
            --scan-dsmil)
                RUN_DSMIL_SCAN=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

main() {
    clear
    log "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    log "${BLUE}║  LAT5150 DRVMIL Tactical AI Sub-Engine - Production Deployment       ║${NC}"
    log "${BLUE}║  Version: 1.0.0                                                       ║${NC}"
    log "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    log ""
    info "Deployment started: $(date)"
    info "Log file: $LOG_FILE"
    log ""

    # Parse command-line arguments
    parse_arguments "$@"

    # Check root
    check_root "$@"

    # Display deployment plan
    section "DEPLOYMENT PLAN"
    info "The following steps will be executed:"
    info "  1. System validation $([ "$SKIP_VALIDATION" = true ] && echo "(SKIP)" || echo "")"
    info "  2. Backup current system $([ "$SKIP_BACKUP" = true ] && echo "(SKIP)" || echo "")"
    info "  3. Install dependencies"
    info "  4. Install SystemD service"
    info "  5. Verify API health"
    info "  6. Deploy VM shortcuts $([ "$DEPLOY_VMS" = true ] && echo "(VMs: $VM_IPS)" || echo "(SKIP)")"
    info "  7. Run DSMIL scan $([ "$RUN_DSMIL_SCAN" = true ] && echo "" || echo "(SKIP)")"
    info "  8. Final verification"
    log ""

    prompt_continue

    # Execute deployment steps
    if ! step_validate; then
        error "Deployment aborted due to validation failure"
        exit 1
    fi

    if ! step_backup; then
        error "Deployment aborted due to backup failure"
        exit 1
    fi

    if ! step_install_dependencies; then
        error "Deployment aborted due to dependency installation failure"
        exit 1
    fi

    if ! step_install_service; then
        error "Deployment aborted due to service installation failure"
        exit 1
    fi

    if ! step_verify_api; then
        error "Deployment aborted due to API verification failure"
        exit 1
    fi

    step_deploy_vms || warn "VM deployment had issues (non-fatal)"

    step_run_dsmil_scan || warn "DSMIL scan had issues (non-fatal)"

    if step_final_verification; then
        section "DEPLOYMENT SUCCESSFUL"

        success "LAT5150 DRVMIL Tactical AI Sub-Engine is now deployed!"
        log ""
        info "Access the tactical interface at: ${CYAN}http://127.0.0.1:5001${NC}"
        info "Monitor the system with: ${CYAN}sudo $SCRIPT_DIR/monitor-system.sh${NC}"
        info "View logs with: ${CYAN}sudo journalctl -u lat5150-tactical.service -f${NC}"
        log ""
        info "System is ready for operational use."
        log ""

        # Display next steps
        log "${YELLOW}Next Steps:${NC}"
        log "  1. Access tactical interface in browser"
        log "  2. Configure TEMPEST display mode (default: Comfort/Level C)"
        log "  3. Deploy VM shortcuts to guest systems (if not done)"
        log "  4. Run DSMIL device reconnaissance (if not done)"
        log "  5. Review deployment guide: $BASE_DIR/DEPLOYMENT_GUIDE.md"
        log ""

        log "Full deployment log: $LOG_FILE"

        exit 0
    else
        section "DEPLOYMENT COMPLETED WITH WARNINGS"

        warn "Some verification checks failed"
        info "The system may still be functional, but requires attention"
        info "Review the issues above and check logs"
        log ""
        log "Deployment log: $LOG_FILE"

        exit 1
    fi
}

# Run main deployment
main "$@"
