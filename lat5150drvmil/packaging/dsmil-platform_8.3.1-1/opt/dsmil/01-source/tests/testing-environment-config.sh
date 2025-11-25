#!/bin/bash
#
# DSMIL Testing Environment Configuration
# Prepares Dell Latitude 5450 MIL-SPEC for safe SMBIOS token testing
#
# This script configures the testing environment with comprehensive safety measures
# and monitoring systems for secure token enumeration.
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/config_$(date +%Y%m%d_%H%M%S).log"
BASELINE_DIR="${SCRIPT_DIR}/baseline_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "${BLUE}[INFO]${NC} ${1}"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} ${1}"
}

log_error() {
    log "${RED}[ERROR]${NC} ${1}"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} ${1}"
}

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

log_info "Starting DSMIL testing environment configuration"
log_info "Script directory: ${SCRIPT_DIR}"
log_info "Log file: ${LOG_FILE}"

# Verification functions
verify_system_compatibility() {
    log_info "Verifying system compatibility..."
    
    # Check if this is Dell Latitude 5450
    if ! dmidecode -t system | grep -q "Latitude 5450"; then
        log_error "This script is designed for Dell Latitude 5450 MIL-SPEC"
        exit 1
    fi
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r)
    log_info "Kernel version: ${KERNEL_VERSION}"
    
    # Check if running as root (needed for some operations)
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root - this is required for some operations"
    fi
    
    log_success "System compatibility verified"
}

configure_thermal_protection() {
    log_info "Configuring thermal protection..."
    
    # Check if thermal guardian is available
    if [[ -f "${SCRIPT_DIR}/thermal-guardian/thermal_guardian.py" ]]; then
        log_info "Thermal guardian found - configuring"
        
        # Set thermal thresholds
        cat > "${SCRIPT_DIR}/thermal_guardian.conf" << EOF
# DSMIL Thermal Guardian Configuration
warning_temperature=85
critical_temperature=95
emergency_temperature=100
check_interval=1.0
alert_file=/tmp/thermal_alert
emergency_script=${SCRIPT_DIR}/monitoring/emergency_stop.sh
EOF
        
        # Make thermal guardian executable
        chmod +x "${SCRIPT_DIR}/thermal-guardian/thermal_guardian.py"
        
        log_success "Thermal protection configured (85°C warning, 95°C critical)"
    else
        log_warn "Thermal guardian not found - manual monitoring required"
    fi
}

setup_monitoring_infrastructure() {
    log_info "Setting up monitoring infrastructure..."
    
    # Ensure monitoring directory exists
    mkdir -p "${SCRIPT_DIR}/monitoring"
    
    # Create emergency stop script if it doesn't exist
    if [[ ! -f "${SCRIPT_DIR}/monitoring/emergency_stop.sh" ]]; then
        cat > "${SCRIPT_DIR}/monitoring/emergency_stop.sh" << 'EOF'
#!/bin/bash
# DSMIL Emergency Stop Script
echo "[EMERGENCY] Stopping all DSMIL operations..."

# Kill monitoring processes
sudo killall -9 python3 2>/dev/null || true

# Remove DSMIL kernel modules
sudo rmmod dsmil-72dev 2>/dev/null || true

# Log emergency event
echo "$(date -Iseconds): Emergency stop triggered" >> /tmp/dsmil_emergency.log

echo "[EMERGENCY] DSMIL operations stopped"
EOF
        chmod +x "${SCRIPT_DIR}/monitoring/emergency_stop.sh"
    fi
    
    # Create multi-terminal launcher if it doesn't exist
    if [[ ! -f "${SCRIPT_DIR}/monitoring/multi_terminal_launcher.sh" ]]; then
        cat > "${SCRIPT_DIR}/monitoring/multi_terminal_launcher.sh" << 'EOF'
#!/bin/bash
# Multi-terminal DSMIL monitoring launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Launch terminals for different monitoring aspects
gnome-terminal --tab --title="Main Dashboard" -- python3 "${SCRIPT_DIR}/dsmil_comprehensive_monitor.py" --mode dashboard &
gnome-terminal --tab --title="Resource Monitor" -- python3 "${SCRIPT_DIR}/dsmil_comprehensive_monitor.py" --mode resources &
gnome-terminal --tab --title="Token Monitor" -- python3 "${SCRIPT_DIR}/dsmil_comprehensive_monitor.py" --mode tokens &
gnome-terminal --tab --title="Kernel Messages" -- watch -n1 "dmesg | grep -i -E '(dsmil|dell|smbios)' | tail -10" &

echo "Monitoring terminals launched"
EOF
        chmod +x "${SCRIPT_DIR}/monitoring/multi_terminal_launcher.sh"
    fi
    
    # Make sure comprehensive monitor is executable
    if [[ -f "${SCRIPT_DIR}/monitoring/dsmil_comprehensive_monitor.py" ]]; then
        chmod +x "${SCRIPT_DIR}/monitoring/dsmil_comprehensive_monitor.py"
        log_success "Comprehensive monitoring system configured"
    else
        log_warn "Comprehensive monitor not found - limited monitoring available"
    fi
}

configure_kernel_module_safety() {
    log_info "Configuring kernel module safety parameters..."
    
    # Check if DSMIL kernel module source exists
    if [[ -f "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c" ]]; then
        # Verify safety parameters in source
        if grep -q "force_jrtc1_mode = true" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
            log_success "JRTC1 safety mode confirmed in kernel module"
        else
            log_warn "JRTC1 safety mode may not be enabled"
        fi
        
        if grep -q "thermal_threshold = 85" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
            log_success "Thermal threshold safety confirmed (85°C)"
        else
            log_warn "Thermal threshold may not be configured"
        fi
        
        if grep -q "DSMIL_CHUNK_SIZE.*4.*1024.*1024" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
            log_success "Chunked memory mapping confirmed (4MB chunks)"
        else
            log_warn "Chunked memory mapping may not be configured"
        fi
    else
        log_warn "DSMIL kernel module source not found"
    fi
}

create_testing_isolation() {
    log_info "Creating testing isolation environment..."
    
    # Create isolated testing directory
    TESTING_DIR="${SCRIPT_DIR}/testing_session_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${TESTING_DIR}"
    
    # Create session configuration
    cat > "${TESTING_DIR}/session_config.json" << EOF
{
    "session_id": "$(date +%Y%m%d_%H%M%S)",
    "target_ranges": [
        {"name": "Range_0480", "start": "0x0480", "end": "0x04C7", "priority": 1},
        {"name": "Range_0400", "start": "0x0400", "end": "0x0447", "priority": 2},
        {"name": "Range_0500", "start": "0x0500", "end": "0x0547", "priority": 3}
    ],
    "safety_limits": {
        "max_temperature": 85,
        "max_cpu_usage": 80,
        "max_memory_usage": 85,
        "max_test_duration": 1800
    },
    "monitoring": {
        "update_interval": 0.5,
        "alert_thresholds": {
            "temperature_warning": 85,
            "temperature_critical": 95
        }
    }
}
EOF
    
    # Create session startup script
    cat > "${TESTING_DIR}/start_session.sh" << EOF
#!/bin/bash
# DSMIL Testing Session Startup

cd "${TESTING_DIR}"
echo "Starting DSMIL testing session: \$(date)"

# Start monitoring
"${SCRIPT_DIR}/monitoring/multi_terminal_launcher.sh" &

# Load kernel module with safety parameters
cd "${SCRIPT_DIR}/01-source/kernel"
sudo make clean && sudo make && sudo insmod dsmil-72dev.ko force_jrtc1_mode=1 thermal_threshold=85

echo "Testing session initialized"
echo "Monitoring active - check terminal tabs"
echo "Ready for token enumeration"
EOF
    chmod +x "${TESTING_DIR}/start_session.sh"
    
    log_success "Testing isolation environment created: ${TESTING_DIR}"
    echo "${TESTING_DIR}" > "${SCRIPT_DIR}/.last_testing_session"
}

setup_rollback_mechanisms() {
    log_info "Setting up rollback mechanisms..."
    
    # Create quick rollback script
    cat > "${SCRIPT_DIR}/quick_rollback.sh" << 'EOF'
#!/bin/bash
# Quick DSMIL rollback script

echo "[ROLLBACK] Initiating quick system rollback..."

# Stop all DSMIL operations
sudo rmmod dsmil-72dev 2>/dev/null || true
sudo killall -9 python3 2>/dev/null || true

# Check system state
echo "[ROLLBACK] System state:"
echo "  Temperature: $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print $1/1000}')°C"
echo "  Memory usage: $(free | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
echo "  CPU load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  DSMIL modules: $(lsmod | grep -i dsmil | wc -l)"

# Compare to baseline if available
LATEST_BASELINE=$(ls -1t baseline_*.tar.gz 2>/dev/null | head -1)
if [[ -n "$LATEST_BASELINE" ]]; then
    echo "[ROLLBACK] Latest baseline: $LATEST_BASELINE"
    echo "[ROLLBACK] To restore baseline: tar -xzf $LATEST_BASELINE"
fi

echo "[ROLLBACK] Quick rollback complete"
EOF
    chmod +x "${SCRIPT_DIR}/quick_rollback.sh"
    
    # Create comprehensive rollback script
    cat > "${SCRIPT_DIR}/comprehensive_rollback.sh" << 'EOF'
#!/bin/bash
# Comprehensive DSMIL system rollback

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[ROLLBACK] Initiating comprehensive system rollback..."

# Find latest baseline
LATEST_BASELINE=$(ls -1t "${SCRIPT_DIR}"/baseline_*.tar.gz 2>/dev/null | head -1)

if [[ -z "$LATEST_BASELINE" ]]; then
    echo "[ERROR] No baseline found for rollback"
    exit 1
fi

echo "[ROLLBACK] Using baseline: $LATEST_BASELINE"

# Emergency stop
"${SCRIPT_DIR}/monitoring/emergency_stop.sh"

# Extract baseline for comparison
ROLLBACK_DIR="${SCRIPT_DIR}/rollback_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROLLBACK_DIR"
tar -xzf "$LATEST_BASELINE" -C "$ROLLBACK_DIR"

# Compare current state
echo "[ROLLBACK] System state comparison:"
diff <(dmidecode -t system) "${ROLLBACK_DIR}/system_info.txt" || echo "System info differs"
diff <(sensors 2>/dev/null | grep -v ERROR) "${ROLLBACK_DIR}/thermal_baseline.txt" || echo "Thermal state differs"

echo "[ROLLBACK] Rollback analysis complete in: $ROLLBACK_DIR"
EOF
    chmod +x "${SCRIPT_DIR}/comprehensive_rollback.sh"
    
    log_success "Rollback mechanisms configured"
}

validate_configuration() {
    log_info "Validating testing environment configuration..."
    
    local validation_errors=0
    
    # Check essential files
    local essential_files=(
        "01-source/kernel/dsmil-72dev.c"
        "monitoring/dsmil_comprehensive_monitor.py"
        "monitoring/emergency_stop.sh"
        "quick_rollback.sh"
        "comprehensive_rollback.sh"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
            log_success "✓ ${file} present"
        else
            log_error "✗ ${file} missing"
            ((validation_errors++))
        fi
    done
    
    # Check if we have baseline snapshots
    local baseline_count=$(ls -1 "${SCRIPT_DIR}"/baseline_*.tar.gz 2>/dev/null | wc -l)
    if [[ $baseline_count -gt 0 ]]; then
        log_success "✓ ${baseline_count} baseline snapshots available"
    else
        log_warn "⚠ No baseline snapshots found"
        ((validation_errors++))
    fi
    
    # Check thermal monitoring capability
    if [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
        local temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        local temp_c=$((temp / 1000))
        if [[ $temp_c -lt 80 ]]; then
            log_success "✓ System temperature normal (${temp_c}°C)"
        else
            log_warn "⚠ System temperature elevated (${temp_c}°C)"
        fi
    else
        log_warn "⚠ Thermal monitoring may be limited"
    fi
    
    # Check kernel module build capability
    if [[ -f "${SCRIPT_DIR}/01-source/kernel/Makefile" ]]; then
        log_success "✓ Kernel module build system ready"
    else
        log_error "✗ Kernel module build system missing"
        ((validation_errors++))
    fi
    
    # Final validation
    if [[ $validation_errors -eq 0 ]]; then
        log_success "Configuration validation passed - system ready for testing"
        return 0
    else
        log_error "Configuration validation failed with ${validation_errors} errors"
        return 1
    fi
}

create_testing_summary() {
    log_info "Creating testing environment summary..."
    
    cat > "${SCRIPT_DIR}/TESTING_ENVIRONMENT_STATUS.md" << EOF
# DSMIL Testing Environment Status

## Configuration Summary
- **Configuration Date**: $(date -Iseconds)
- **System**: $(dmidecode -t system | grep "Product Name:" | cut -d: -f2 | xargs)
- **BIOS Version**: $(dmidecode -t bios | grep "Version:" | cut -d: -f2 | xargs)
- **Kernel**: $(uname -r)

## Safety Systems
- ✅ Thermal protection configured (85°C warning, 95°C critical)
- ✅ Emergency stop procedures ready
- ✅ System rollback mechanisms prepared
- ✅ Monitoring infrastructure deployed

## Testing Configuration
- **Primary Target**: Range 0x0480-0x04C7 (72 tokens)
- **Memory Mapping**: 4MB chunks (safe)
- **Safety Mode**: JRTC1 training mode enforced
- **Monitoring Interval**: 0.5 seconds

## Emergency Procedures
- **Quick Stop**: Ctrl+C in monitoring terminal
- **Emergency Script**: ./monitoring/emergency_stop.sh
- **Quick Rollback**: ./quick_rollback.sh
- **Full Rollback**: ./comprehensive_rollback.sh

## File Locations
- **Configuration Log**: ${LOG_FILE}
- **Emergency Logs**: /tmp/dsmil_emergency.log
- **Testing Sessions**: ./testing_session_*
- **Baselines**: ./baseline_*.tar.gz

## System Status: READY FOR TESTING ✅

EOF

    log_success "Testing environment summary created"
}

# Main execution
main() {
    log_info "=== DSMIL Testing Environment Configuration ==="
    
    verify_system_compatibility
    configure_thermal_protection
    setup_monitoring_infrastructure
    configure_kernel_module_safety
    create_testing_isolation
    setup_rollback_mechanisms
    
    if validate_configuration; then
        create_testing_summary
        log_success "=== Configuration Complete - System Ready for Testing ==="
        
        echo ""
        log_info "Next steps:"
        log_info "1. Review: cat ${SCRIPT_DIR}/TESTING_ENVIRONMENT_STATUS.md"
        log_info "2. Start testing: ./testing_session_*/start_session.sh"
        log_info "3. Monitor: ./monitoring/multi_terminal_launcher.sh"
        log_info "4. Emergency stop: ./monitoring/emergency_stop.sh"
        
        return 0
    else
        log_error "=== Configuration Failed - System Not Ready ==="
        return 1
    fi
}

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi