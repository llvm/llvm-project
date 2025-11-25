#!/bin/bash
#
# Dell SMBIOS Token Discovery Script
# Safe enumeration and analysis of SMBIOS tokens for DSMIL device discovery
#
# SAFETY FEATURES:
# - Read-only access only
# - Avoids dangerous token ranges (0x8000-0x8014, 0xF600-0xF601)
# - Emergency stop mechanisms
# - Comprehensive logging and monitoring
# - Pattern analysis for DSMIL device control tokens
#
# Usage:
#   ./dell-smbios-token-discovery.sh [OPTIONS]
#
# Options:
#   --safe-mode      Use extra-safe scanning parameters
#   --debug-mode     Enable verbose output and debug logging
#   --monitor-only   Monitor existing enumeration without starting new scan
#   --emergency-stop Stop all enumeration immediately
#   --report         Generate comprehensive analysis report
#   --help           Show this help message

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${SCRIPT_DIR}/../kernel"
LOG_DIR="${SCRIPT_DIR}/../../logs"
REPORT_DIR="${SCRIPT_DIR}/../../reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODULE_NAME="dell-smbios-token-enum"
LOG_FILE="${LOG_DIR}/token-discovery-${TIMESTAMP}.log"
REPORT_FILE="${REPORT_DIR}/token-analysis-${TIMESTAMP}.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "${LOG_FILE}"
    fi
}

# Safety check functions
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for kernel module operations"
        exit 1
    fi
}

check_kernel_version() {
    local kernel_version=$(uname -r)
    local major_version=$(echo "$kernel_version" | cut -d. -f1)
    local minor_version=$(echo "$kernel_version" | cut -d. -f2)
    
    log_info "Kernel version: $kernel_version"
    
    if [[ $major_version -lt 5 ]] || ([[ $major_version -eq 5 ]] && [[ $minor_version -lt 4 ]]); then
        log_warn "Kernel version may not support all features (recommended: 5.4+)"
    fi
}

check_dell_smbios_support() {
    log_info "Checking Dell SMBIOS support..."
    
    if lsmod | grep -q "dell_smbios"; then
        log_info "Dell SMBIOS kernel module is loaded"
    else
        log_warn "Dell SMBIOS kernel module not loaded - attempting to load..."
        modprobe dell_smbios || log_warn "Failed to load dell_smbios module"
    fi
    
    if [[ -d "/sys/firmware/dmi" ]]; then
        log_info "DMI firmware interface available"
    else
        log_warn "DMI firmware interface not available"
    fi
}

setup_directories() {
    mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
    log_info "Created directories: ${LOG_DIR}, ${REPORT_DIR}"
}

# Emergency stop function
emergency_stop() {
    log_error "EMERGENCY STOP ACTIVATED"
    
    # Stop via sysfs if available
    if [[ -f "/sys/devices/platform/${MODULE_NAME}/emergency_stop" ]]; then
        echo 1 > "/sys/devices/platform/${MODULE_NAME}/emergency_stop"
        log_info "Emergency stop signal sent via sysfs"
    fi
    
    # Force unload module
    if lsmod | grep -q "${MODULE_NAME}"; then
        log_info "Force unloading module..."
        rmmod "${MODULE_NAME}" || log_error "Failed to unload module"
    fi
    
    log_info "Emergency stop complete"
    exit 1
}

# Module operations
build_module() {
    log_info "Building token enumeration module..."
    
    cd "${KERNEL_DIR}"
    if make -f Makefile.token-enum clean && make -f Makefile.token-enum modules; then
        log_info "Module built successfully"
    else
        log_error "Module build failed"
        return 1
    fi
}

load_module() {
    local mode=$1
    local params
    
    case "$mode" in
        "safe")
            params="enable_verbose=0 max_tokens=100 scan_delay_ms=500 emergency_stop=0"
            log_info "Loading module in SAFE mode"
            ;;
        "debug")
            params="enable_verbose=1 max_tokens=256 scan_delay_ms=200 emergency_stop=0"
            log_info "Loading module in DEBUG mode"
            ;;
        "normal")
            params="enable_verbose=0 max_tokens=1024 scan_delay_ms=100 emergency_stop=0"
            log_info "Loading module in NORMAL mode"
            ;;
        *)
            log_error "Unknown load mode: $mode"
            return 1
            ;;
    esac
    
    # Unload if already loaded
    if lsmod | grep -q "${MODULE_NAME}"; then
        log_info "Module already loaded, unloading first..."
        rmmod "${MODULE_NAME}" || true
        sleep 2
    fi
    
    cd "${KERNEL_DIR}"
    if insmod "${MODULE_NAME}.ko" $params; then
        log_info "Module loaded successfully with parameters: $params"
        return 0
    else
        log_error "Module load failed"
        return 1
    fi
}

unload_module() {
    log_info "Unloading token enumeration module..."
    
    if lsmod | grep -q "${MODULE_NAME}"; then
        if rmmod "${MODULE_NAME}"; then
            log_info "Module unloaded successfully"
        else
            log_error "Module unload failed"
            return 1
        fi
    else
        log_info "Module not loaded"
    fi
}

# Monitoring functions
monitor_enumeration() {
    local duration=${1:-30}
    
    log_info "Monitoring token enumeration for ${duration} seconds..."
    
    # Start monitoring in background
    timeout "${duration}" dmesg -w | grep -i -E "(token|smbios|dell)" | while read -r line; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') $line" | tee -a "${LOG_FILE}"
    done &
    
    local monitor_pid=$!
    
    # Wait for monitoring to complete
    wait $monitor_pid 2>/dev/null || true
    
    log_info "Monitoring complete"
}

check_module_status() {
    log_info "=== Module Status ==="
    
    if lsmod | grep -q "${MODULE_NAME}"; then
        log_info "Module: LOADED"
        lsmod | grep "${MODULE_NAME}"
    else
        log_warn "Module: NOT LOADED"
        return 1
    fi
    
    # Check parameters
    if [[ -d "/sys/module/${MODULE_NAME}/parameters" ]]; then
        log_info "Module parameters:"
        for param in /sys/module/${MODULE_NAME}/parameters/*; do
            local param_name=$(basename "$param")
            local param_value=$(cat "$param")
            log_info "  ${param_name}: ${param_value}"
        done
    fi
    
    # Check emergency stop status
    if [[ -f "/sys/devices/platform/${MODULE_NAME}/emergency_stop" ]]; then
        local emergency_status=$(cat "/sys/devices/platform/${MODULE_NAME}/emergency_stop")
        if [[ "$emergency_status" == "1" ]]; then
            log_warn "  Emergency stop: ACTIVE"
        else
            log_info "  Emergency stop: INACTIVE"
        fi
    fi
}

# Analysis and reporting functions
generate_report() {
    log_info "Generating comprehensive token analysis report..."
    
    {
        echo "Dell SMBIOS Token Discovery Report"
        echo "=================================="
        echo "Generated: $(date)"
        echo "System: $(uname -a)"
        echo "Dell Model: $(dmidecode -s system-product-name 2>/dev/null || echo 'Unknown')"
        echo "BIOS Version: $(dmidecode -s bios-version 2>/dev/null || echo 'Unknown')"
        echo ""
        
        # Module status
        echo "=== Module Status ==="
        if lsmod | grep -q "${MODULE_NAME}"; then
            echo "Module Status: LOADED"
            lsmod | grep "${MODULE_NAME}"
        else
            echo "Module Status: NOT LOADED"
        fi
        echo ""
        
        # Token enumeration results
        echo "=== Token Enumeration Results ==="
        if [[ -f "/proc/dell-token-enum" ]]; then
            cat "/proc/dell-token-enum"
        else
            echo "Token enumeration data not available"
        fi
        echo ""
        
        # System DMI information
        echo "=== System DMI Information ==="
        dmidecode -t system 2>/dev/null | grep -E "(Manufacturer|Product|Version|Serial)" || echo "DMI data not available"
        echo ""
        
        # SMBIOS modules
        echo "=== Loaded SMBIOS Modules ==="
        lsmod | grep -i -E "(dell|smbios)" || echo "No Dell/SMBIOS modules loaded"
        echo ""
        
        # Kernel messages related to token discovery
        echo "=== Related Kernel Messages ==="
        dmesg | grep -i -E "(token|dell.*smbios)" | tail -50 || echo "No relevant kernel messages"
        
    } > "${REPORT_FILE}"
    
    log_info "Report generated: ${REPORT_FILE}"
}

analyze_dsmil_patterns() {
    log_info "Analyzing discovered tokens for DSMIL patterns..."
    
    if [[ ! -f "/proc/dell-token-enum" ]]; then
        log_warn "Token enumeration data not available for analysis"
        return 1
    fi
    
    # Extract DSMIL-related tokens
    local dsmil_tokens
    dsmil_tokens=$(grep -E "0x8[34]" "/proc/dell-token-enum" 2>/dev/null || true)
    
    if [[ -n "$dsmil_tokens" ]]; then
        log_info "DSMIL-related tokens found:"
        echo "$dsmil_tokens" | while read -r line; do
            log_info "  $line"
        done
        
        # Look for device control patterns
        local device_pattern_count
        device_pattern_count=$(echo "$dsmil_tokens" | grep -c "0x44" || echo "0")
        
        log_info "Potential device control tokens: $device_pattern_count"
        
        if [[ "$device_pattern_count" -gt 0 ]]; then
            log_info "DSMIL device control mechanism potentially discovered!"
        fi
    else
        log_info "No DSMIL-related tokens found in safe ranges"
    fi
}

# Main execution functions
run_safe_discovery() {
    log_info "=== SAFE TOKEN DISCOVERY MODE ==="
    
    check_root
    setup_directories
    check_kernel_version
    check_dell_smbios_support
    
    # Build and load module in safe mode
    build_module || exit 1
    load_module "safe" || exit 1
    
    # Monitor enumeration
    log_info "Starting enumeration monitoring..."
    monitor_enumeration 60
    
    # Check status
    check_module_status || true
    
    # Analyze results
    analyze_dsmil_patterns || true
    
    # Generate report
    generate_report
    
    log_info "Safe discovery complete - check report: ${REPORT_FILE}"
}

run_debug_discovery() {
    log_info "=== DEBUG TOKEN DISCOVERY MODE ==="
    
    DEBUG_MODE=1
    
    check_root
    setup_directories
    check_kernel_version  
    check_dell_smbios_support
    
    # Build and load module in debug mode
    build_module || exit 1
    load_module "debug" || exit 1
    
    # Extended monitoring with verbose output
    log_info "Starting extended enumeration monitoring..."
    monitor_enumeration 120
    
    # Check status
    check_module_status || true
    
    # Analyze results
    analyze_dsmil_patterns || true
    
    # Generate detailed report
    generate_report
    
    log_info "Debug discovery complete - check report: ${REPORT_FILE}"
}

monitor_only() {
    log_info "=== MONITORING EXISTING ENUMERATION ==="
    
    if ! lsmod | grep -q "${MODULE_NAME}"; then
        log_error "Token enumeration module not loaded"
        exit 1
    fi
    
    check_module_status
    monitor_enumeration 30
    analyze_dsmil_patterns || true
    generate_report
}

show_usage() {
    cat << EOF
Dell SMBIOS Token Discovery Script
==================================

SAFETY NOTICE: This script performs READ-ONLY token enumeration and 
avoids dangerous ranges (0x8000-0x8014, 0xF600-0xF601).

Usage: $0 [OPTIONS]

Options:
  --safe-mode      Use extra-safe scanning parameters (slow, careful)
  --debug-mode     Enable verbose output and extended monitoring
  --monitor-only   Monitor existing enumeration without starting new scan
  --emergency-stop Stop all enumeration immediately and unload module
  --report         Generate analysis report from current data
  --status         Show current module status
  --help           Show this help message

Examples:
  $0 --safe-mode      # Safest option, recommended for first run
  $0 --debug-mode     # More verbose output for analysis
  $0 --monitor-only   # Just monitor if module already running
  $0 --report         # Generate report without new enumeration
  $0 --emergency-stop # Emergency stop and cleanup

Safety Features:
- Read-only token access only
- Dangerous range avoidance
- Emergency stop capability
- Comprehensive logging
- Pattern analysis for DSMIL devices

Report Location: ${REPORT_DIR}/
Log Location: ${LOG_DIR}/
EOF
}

# Signal handlers
trap emergency_stop SIGINT SIGTERM

# Main script logic
main() {
    local mode="normal"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --safe-mode)
                mode="safe"
                shift
                ;;
            --debug-mode)
                mode="debug"
                shift
                ;;
            --monitor-only)
                mode="monitor"
                shift
                ;;
            --emergency-stop)
                emergency_stop
                exit 0
                ;;
            --report)
                setup_directories
                generate_report
                exit 0
                ;;
            --status)
                check_module_status
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute based on mode
    case "$mode" in
        "safe")
            run_safe_discovery
            ;;
        "debug")
            run_debug_discovery
            ;;
        "monitor")
            monitor_only
            ;;
        "normal")
            log_info "Running normal discovery mode (use --safe-mode for first run)"
            run_safe_discovery
            ;;
        *)
            log_error "Invalid mode: $mode"
            exit 1
            ;;
    esac
    
    # Cleanup
    unload_module || true
    
    log_info "Token discovery complete"
    log_info "Report: ${REPORT_FILE}"
    log_info "Log: ${LOG_FILE}"
}

# Execute main function
main "$@"