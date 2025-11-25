#!/bin/bash
#
# DSMIL System Safety Validation Script
# Comprehensive safety verification before token testing
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/safety_validation_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

log_pass() {
    log "${GREEN}✅ PASS${NC}: ${1}"
}

log_fail() {
    log "${RED}❌ FAIL${NC}: ${1}"
}

log_warn() {
    log "${YELLOW}⚠️  WARN${NC}: ${1}"
}

log_info() {
    log "${BLUE}ℹ️  INFO${NC}: ${1}"
}

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Validation counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

validation_result() {
    local result=$1
    local message=$2
    
    case $result in
        "PASS")
            log_pass "$message"
            ((PASS_COUNT++))
            ;;
        "FAIL")
            log_fail "$message"
            ((FAIL_COUNT++))
            ;;
        "WARN")
            log_warn "$message"
            ((WARN_COUNT++))
            ;;
    esac
}

log_info "Starting DSMIL system safety validation"
log_info "Validation log: ${LOG_FILE}"

# === HARDWARE VALIDATION ===
log_info ""
log_info "=== HARDWARE VALIDATION ==="

# Check system model
PRODUCT_NAME=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "Unknown")
if [[ "$PRODUCT_NAME" == "Latitude 5450" ]]; then
    validation_result "PASS" "Dell Latitude 5450 confirmed"
else
    validation_result "FAIL" "Not Dell Latitude 5450 (detected: $PRODUCT_NAME) - system compatibility unknown"
fi

# Check BIOS version
BIOS_VERSION=$(cat /sys/devices/virtual/dmi/id/bios_version 2>/dev/null || echo "Unknown")
if [[ "$BIOS_VERSION" != "Unknown" ]]; then
    validation_result "PASS" "BIOS version detected: $BIOS_VERSION"
else
    validation_result "WARN" "BIOS version could not be determined"
fi

# Check thermal sensors
THERMAL_ZONES=$(ls /sys/class/thermal/thermal_zone*/temp 2>/dev/null | wc -l)
if [[ $THERMAL_ZONES -gt 0 ]]; then
    validation_result "PASS" "$THERMAL_ZONES thermal sensors detected"
    
    # Check current temperature
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [[ -r "$zone" ]]; then
            temp_raw=$(cat "$zone")
            temp_c=$((temp_raw / 1000))
            if [[ $temp_c -lt 70 ]]; then
                validation_result "PASS" "Thermal zone temperature normal: ${temp_c}°C"
            elif [[ $temp_c -lt 85 ]]; then
                validation_result "WARN" "Thermal zone temperature elevated: ${temp_c}°C"
            else
                validation_result "FAIL" "Thermal zone temperature critical: ${temp_c}°C"
            fi
            break  # Just check first readable zone
        fi
    done
else
    validation_result "FAIL" "No thermal sensors detected"
fi

# Check CPU capabilities
CPU_CORES=$(nproc)
if [[ $CPU_CORES -ge 4 ]]; then
    validation_result "PASS" "Sufficient CPU cores: $CPU_CORES"
else
    validation_result "WARN" "Limited CPU cores: $CPU_CORES"
fi

# Check memory
MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
if [[ $MEMORY_GB -ge 8 ]]; then
    validation_result "PASS" "Sufficient memory: ${MEMORY_GB}GB"
else
    validation_result "WARN" "Limited memory: ${MEMORY_GB}GB"
fi

# === SOFTWARE ENVIRONMENT ===
log_info ""
log_info "=== SOFTWARE ENVIRONMENT VALIDATION ==="

# Check kernel version
KERNEL_VERSION=$(uname -r)
if [[ "$KERNEL_VERSION" =~ ^6\. ]]; then
    validation_result "PASS" "Modern kernel version: $KERNEL_VERSION"
else
    validation_result "WARN" "Older kernel version: $KERNEL_VERSION"
fi

# Check for required tools
REQUIRED_TOOLS=("dmidecode" "sensors" "make" "gcc" "python3" "dmesg")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        validation_result "PASS" "Required tool available: $tool"
    else
        validation_result "FAIL" "Required tool missing: $tool"
    fi
done

# Check Python modules
PYTHON_MODULES=("psutil" "subprocess" "threading" "json")
for module in "${PYTHON_MODULES[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        validation_result "PASS" "Python module available: $module"
    else
        validation_result "FAIL" "Python module missing: $module"
    fi
done

# === DSMIL INFRASTRUCTURE ===
log_info ""
log_info "=== DSMIL INFRASTRUCTURE VALIDATION ==="

# Check kernel module source
if [[ -f "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c" ]]; then
    validation_result "PASS" "DSMIL kernel module source present"
    
    # Check safety parameters in source
    if grep -q "force_jrtc1_mode = true" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
        validation_result "PASS" "JRTC1 safety mode enforced in source"
    else
        validation_result "FAIL" "JRTC1 safety mode not enforced"
    fi
    
    if grep -q "thermal_threshold = 85" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
        validation_result "PASS" "Thermal threshold configured (85°C)"
    else
        validation_result "FAIL" "Thermal threshold not configured"
    fi
    
    if grep -q "DSMIL_CHUNK_SIZE.*4.*1024.*1024" "${SCRIPT_DIR}/01-source/kernel/dsmil-72dev.c"; then
        validation_result "PASS" "Chunked memory mapping configured (4MB)"
    else
        validation_result "FAIL" "Chunked memory mapping not configured"
    fi
else
    validation_result "FAIL" "DSMIL kernel module source missing"
fi

# Check build system
if [[ -f "${SCRIPT_DIR}/01-source/kernel/Makefile" ]]; then
    validation_result "PASS" "Kernel module build system present"
    
    # Test compilation
    cd "${SCRIPT_DIR}/01-source/kernel"
    if make clean >/dev/null 2>&1 && make >/dev/null 2>&1; then
        validation_result "PASS" "Kernel module compiles successfully"
        # Clean up build artifacts
        make clean >/dev/null 2>&1
    else
        validation_result "FAIL" "Kernel module compilation failed"
    fi
    cd "$SCRIPT_DIR"
else
    validation_result "FAIL" "Kernel module build system missing"
fi

# === MONITORING SYSTEMS ===
log_info ""
log_info "=== MONITORING SYSTEMS VALIDATION ==="

# Check monitoring scripts
if [[ -f "${SCRIPT_DIR}/monitoring/dsmil_comprehensive_monitor.py" ]]; then
    validation_result "PASS" "Comprehensive monitoring system present"
    
    # Test monitoring system
    if python3 "${SCRIPT_DIR}/monitoring/dsmil_comprehensive_monitor.py" --json-output >/dev/null 2>&1; then
        validation_result "PASS" "Monitoring system functional"
    else
        validation_result "WARN" "Monitoring system may have issues"
    fi
else
    validation_result "FAIL" "Comprehensive monitoring system missing"
fi

# Check emergency procedures
if [[ -f "${SCRIPT_DIR}/monitoring/emergency_stop.sh" && -x "${SCRIPT_DIR}/monitoring/emergency_stop.sh" ]]; then
    validation_result "PASS" "Emergency stop script ready"
else
    validation_result "FAIL" "Emergency stop script missing or not executable"
fi

# Check thermal guardian
if [[ -f "${SCRIPT_DIR}/thermal-guardian/thermal_guardian.py" ]]; then
    validation_result "PASS" "Thermal guardian system present"
else
    validation_result "WARN" "Thermal guardian system not found"
fi

# === SAFETY BASELINES ===
log_info ""
log_info "=== SAFETY BASELINES VALIDATION ==="

# Check for baseline snapshots
BASELINE_COUNT=$(ls -1 "${SCRIPT_DIR}"/baseline_*.tar.gz 2>/dev/null | wc -l)
if [[ $BASELINE_COUNT -gt 0 ]]; then
    validation_result "PASS" "$BASELINE_COUNT baseline snapshots available"
    
    # Check latest baseline
    LATEST_BASELINE=$(ls -1t "${SCRIPT_DIR}"/baseline_*.tar.gz 2>/dev/null | head -1)
    if [[ -n "$LATEST_BASELINE" ]]; then
        validation_result "PASS" "Latest baseline: $(basename "$LATEST_BASELINE")"
        
        # Verify baseline contents
        if tar -tzf "$LATEST_BASELINE" | grep -q "MANIFEST.txt"; then
            validation_result "PASS" "Baseline contains required manifest"
        else
            validation_result "WARN" "Baseline may be incomplete"
        fi
    fi
else
    validation_result "FAIL" "No baseline snapshots found - create baseline before testing"
fi

# Check rollback mechanisms
if [[ -f "${SCRIPT_DIR}/quick_rollback.sh" && -x "${SCRIPT_DIR}/quick_rollback.sh" ]]; then
    validation_result "PASS" "Quick rollback mechanism ready"
else
    validation_result "WARN" "Quick rollback mechanism not configured"
fi

# === SYSTEM STATE ===
log_info ""
log_info "=== CURRENT SYSTEM STATE VALIDATION ==="

# Check for running DSMIL modules
if lsmod | grep -q dsmil; then
    validation_result "WARN" "DSMIL modules currently loaded - unload before testing"
else
    validation_result "PASS" "No DSMIL modules currently loaded"
fi

# Check system load
LOAD_1MIN=$(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1 | xargs)
if (( $(echo "$LOAD_1MIN < 2.0" | bc -l 2>/dev/null || echo "1") )); then
    validation_result "PASS" "System load acceptable: $LOAD_1MIN"
else
    validation_result "WARN" "System load elevated: $LOAD_1MIN"
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
if (( $(echo "$MEMORY_USAGE < 70.0" | bc -l 2>/dev/null || echo "1") )); then
    validation_result "PASS" "Memory usage acceptable: ${MEMORY_USAGE}%"
else
    validation_result "WARN" "Memory usage elevated: ${MEMORY_USAGE}%"
fi

# Check disk space
DISK_USAGE=$(df "$SCRIPT_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
if [[ $DISK_USAGE -lt 90 ]]; then
    validation_result "PASS" "Disk space sufficient: ${DISK_USAGE}% used"
else
    validation_result "WARN" "Disk space limited: ${DISK_USAGE}% used"
fi

# === NETWORK ISOLATION ===
log_info ""
log_info "=== NETWORK ISOLATION VALIDATION ==="

# Check for active network connections that might interfere
NETWORK_CONNECTIONS=$(netstat -tn 2>/dev/null | grep ESTABLISHED | wc -l || echo "0")
if [[ $NETWORK_CONNECTIONS -lt 20 ]]; then
    validation_result "PASS" "Network connections reasonable: $NETWORK_CONNECTIONS"
else
    validation_result "WARN" "Many network connections active: $NETWORK_CONNECTIONS"
fi

# === FINAL VALIDATION ===
log_info ""
log_info "=== VALIDATION SUMMARY ==="

TOTAL_CHECKS=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))
log_info "Total checks performed: $TOTAL_CHECKS"
log_info "Passed: $PASS_COUNT"
log_info "Failed: $FAIL_COUNT"
log_info "Warnings: $WARN_COUNT"

# Determine overall safety status
if [[ $FAIL_COUNT -eq 0 ]]; then
    if [[ $WARN_COUNT -eq 0 ]]; then
        log_info ""
        log_pass "SYSTEM SAFETY STATUS: OPTIMAL ✅"
        log_info "System is ready for safe DSMIL token testing"
        SAFETY_STATUS="OPTIMAL"
        EXIT_CODE=0
    else
        log_info ""
        log_warn "SYSTEM SAFETY STATUS: ACCEPTABLE ⚠️"
        log_info "System can proceed with caution - $WARN_COUNT warnings present"
        SAFETY_STATUS="ACCEPTABLE"
        EXIT_CODE=0
    fi
else
    log_info ""
    log_fail "SYSTEM SAFETY STATUS: NOT READY ❌"
    log_info "System has $FAIL_COUNT critical issues that must be resolved"
    SAFETY_STATUS="NOT_READY"
    EXIT_CODE=1
fi

# Create safety status file
cat > "${SCRIPT_DIR}/SAFETY_STATUS.txt" << EOF
DSMIL System Safety Validation Report
=====================================
Date: $(date -Iseconds)
Status: $SAFETY_STATUS

Validation Results:
- Total Checks: $TOTAL_CHECKS
- Passed: $PASS_COUNT
- Failed: $FAIL_COUNT
- Warnings: $WARN_COUNT

System Ready for Testing: $([ $EXIT_CODE -eq 0 ] && echo "YES" || echo "NO")

Detailed Log: $LOG_FILE
EOF

log_info ""
log_info "Safety validation complete - status saved to SAFETY_STATUS.txt"
log_info "Detailed log available at: $LOG_FILE"

exit $EXIT_CODE