#!/bin/bash
# DSMIL Probe Validation and Rollback Script
# Dell Latitude 5450 MIL-SPEC DSMIL Device Probing
# 
# SAFETY: This script implements comprehensive validation and rollback
# for safe probing of 72 DSMIL devices (6 groups × 12 devices)

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="dsmil_probe_validation"
readonly VERSION="1.0.0"
readonly LOG_DIR="/var/log/dsmil"
readonly BACKUP_DIR="/backup/dsmil"
readonly CONFIG_FILE="/etc/dsmil/probe_config.conf"

# Logging
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"
HEALTH_LOG="$LOG_DIR/dsmil_health.log"

# State tracking
PROBE_STATE_FILE="/tmp/dsmil_probe_state"
ROLLBACK_FILE="/tmp/dsmil_rollback_plan"

# Safety thresholds
readonly MAX_TEMP_CELSIUS=85
readonly MIN_FREE_MEMORY_MB=2000  
readonly MAX_SYSTEM_LOAD=8.0
readonly STABILITY_WAIT_SECONDS=30

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'  
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Trap for cleanup
trap cleanup EXIT
trap emergency_abort SIGINT SIGTERM

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    
    # Also log to health log for monitoring
    if [[ "$level" == "HEALTH" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $message" >> "$HEALTH_LOG"
    fi
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }  
log_error() { log "ERROR" "$@"; }
log_health() { log "HEALTH" "$@"; }

print_banner() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  DSMIL Probe Validation & Rollback Framework v$VERSION"
    echo "  Dell Latitude 5450 MIL-SPEC - 72 Device Probing System"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# =============================================================================
# SAFETY AND VALIDATION FUNCTIONS  
# =============================================================================

check_prerequisites() {
    log_info "Checking system prerequisites..."
    
    # Create required directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR" /etc/dsmil /tmp
    
    # Check for required tools
    local required_tools=("sensors" "bc" "lsmod" "dmesg" "free" "uptime")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' not found"
            return 1
        fi
    done
    
    # Check permissions
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        return 1
    fi
    
    log_info "Prerequisites check passed"
    return 0
}

get_cpu_temperature() {
    # Try multiple methods to get CPU temperature
    local temp=""
    
    # Method 1: lm-sensors
    if command -v sensors &> /dev/null; then
        temp=$(sensors 2>/dev/null | grep -i "package id 0" | awk '{print $4}' | tr -d '+°C' | head -1)
    fi
    
    # Method 2: thermal zone
    if [[ -z "$temp" ]] && [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp=$((temp / 1000))  # Convert millicelsius to celsius
    fi
    
    # Method 3: ACPI
    if [[ -z "$temp" ]] && command -v acpi &> /dev/null; then
        temp=$(acpi -t 2>/dev/null | head -1 | awk '{print $4}' | tr -d '°C')
    fi
    
    echo "${temp:-0}"
}

get_system_load() {
    uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ','
}

get_free_memory_mb() {
    free -m | awk 'NR==2{printf "%.0f", $7}'
}

validate_system_health() {
    local validation_level=${1:-"basic"}
    
    log_info "Validating system health (level: $validation_level)..."
    
    # CPU Temperature check
    local temp=$(get_cpu_temperature)
    if (( $(echo "$temp > $MAX_TEMP_CELSIUS" | bc -l) )); then
        log_error "CPU temperature too high: ${temp}°C (max: ${MAX_TEMP_CELSIUS}°C)"
        return 1
    fi
    log_health "CPU temperature: ${temp}°C"
    
    # Memory check  
    local free_mem=$(get_free_memory_mb)
    if (( free_mem < MIN_FREE_MEMORY_MB )); then
        log_error "Insufficient free memory: ${free_mem}MB (min: ${MIN_FREE_MEMORY_MB}MB)"
        return 1
    fi
    log_health "Free memory: ${free_mem}MB"
    
    # System load check
    local load=$(get_system_load)
    if (( $(echo "$load > $MAX_SYSTEM_LOAD" | bc -l) )); then
        log_error "System load too high: $load (max: $MAX_SYSTEM_LOAD)"
        return 1
    fi
    log_health "System load: $load"
    
    # Disk space check
    local disk_free=$(df / | awk 'NR==2{print $4}')
    if (( disk_free < 5000000 )); then  # 5GB in KB
        log_error "Insufficient disk space: $((disk_free/1024))MB"
        return 1
    fi
    log_health "Disk space: $((disk_free/1024))MB free"
    
    if [[ "$validation_level" == "comprehensive" ]]; then
        # Additional comprehensive checks
        
        # Check for existing DSMIL driver
        if lsmod | grep -q dell.milspec; then
            log_warn "dell-milspec driver already loaded"
        fi
        
        # Check dmesg for hardware errors
        if dmesg | tail -100 | grep -qi "error\|panic\|oops\|bug"; then
            log_warn "Recent kernel errors detected in dmesg"
        fi
        
        # Check system stability (uptime)
        local uptime_seconds=$(awk '{print int($1)}' /proc/uptime)
        if (( uptime_seconds < 300 )); then  # Less than 5 minutes uptime
            log_warn "System recently booted (uptime: ${uptime_seconds}s)"
        fi
        log_health "System uptime: ${uptime_seconds}s"
    fi
    
    log_info "System health validation passed"
    return 0
}

create_system_backup() {
    log_info "Creating system state backup..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/system_backup_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup critical system state
    cat /proc/meminfo > "$backup_path/meminfo.txt"
    cat /proc/interrupts > "$backup_path/interrupts.txt"
    dmesg > "$backup_path/dmesg.txt"
    lsmod > "$backup_path/lsmod.txt"
    lspci -vvv > "$backup_path/lspci.txt"
    
    # Backup DSMIL-related state
    if [[ -d "/sys/devices/platform/dell-milspec" ]]; then
        find /sys/devices/platform/dell-milspec -type f -readable -exec cp {} "$backup_path/" \; 2>/dev/null || true
    fi
    
    # Create backup metadata
    cat > "$backup_path/metadata.txt" << EOF
Backup Created: $(date)
System: $(uname -a)
Hostname: $(hostname)
Uptime: $(uptime)
Purpose: DSMIL probe validation backup
EOF
    
    # Create backup archive
    tar -czf "${backup_path}.tar.gz" -C "$BACKUP_DIR" "$(basename "$backup_path")" 2>/dev/null
    rm -rf "$backup_path"
    
    echo "${backup_path}.tar.gz" > "/tmp/dsmil_last_backup"
    log_info "System backup created: ${backup_path}.tar.gz"
    
    return 0
}

# =============================================================================
# DSMIL DEVICE VALIDATION
# =============================================================================

validate_dsmil_devices() {
    log_info "Validating DSMIL device presence..."
    
    local devices_found=0
    local devices_expected=72
    
    echo "# DSMIL Device Validation Report" > "/tmp/dsmil_device_report"
    echo "# Generated: $(date)" >> "/tmp/dsmil_device_report"
    echo "" >> "/tmp/dsmil_device_report"
    
    # Check device nodes for each group
    for group in {0..5}; do
        echo "Group $group:" >> "/tmp/dsmil_device_report"
        local group_devices=0
        
        for device in {0..11}; do
            local device_hex=$(printf "%X" $device)
            local device_path="/dev/DSMIL${group}D${device_hex}"
            
            if [[ -c "$device_path" ]]; then
                local device_info=$(stat -c "%F %t %T" "$device_path" 2>/dev/null)
                echo "  ✓ DSMIL${group}D${device_hex}: $device_info" >> "/tmp/dsmil_device_report"
                ((devices_found++))
                ((group_devices++))
            else
                echo "  ✗ DSMIL${group}D${device_hex}: Not found" >> "/tmp/dsmil_device_report"
            fi
        done
        
        echo "  Group $group total: $group_devices/12 devices" >> "/tmp/dsmil_device_report"
        echo "" >> "/tmp/dsmil_device_report"
    done
    
    echo "Overall: $devices_found/$devices_expected devices found" >> "/tmp/dsmil_device_report"
    
    log_info "DSMIL device validation: $devices_found/$devices_expected devices found"
    
    # Copy report to log
    cat "/tmp/dsmil_device_report" >> "$LOG_FILE"
    
    if (( devices_found == 0 )); then
        log_error "No DSMIL devices found - hardware may not be initialized"
        return 1
    elif (( devices_found < devices_expected )); then
        log_warn "Partial DSMIL device availability: $devices_found/$devices_expected"
        return 2  # Warning - partial availability
    else
        log_info "All DSMIL devices found and accessible"
        return 0
    fi
}

probe_acpi_methods() {
    log_info "Probing ACPI methods for DSMIL devices..."
    
    local acpi_report="/tmp/dsmil_acpi_report"
    echo "# DSMIL ACPI Method Report" > "$acpi_report"
    echo "# Generated: $(date)" >> "$acpi_report"
    echo "" >> "$acpi_report"
    
    # Check if DSDT contains DSMIL references
    if sudo cat /sys/firmware/acpi/tables/DSDT | strings | grep -E "DSMIL[0-5]" > "/tmp/dsmil_acpi_refs" 2>/dev/null; then
        local ref_count=$(wc -l < "/tmp/dsmil_acpi_refs")
        echo "DSDT DSMIL references found: $ref_count" >> "$acpi_report"
        echo "" >> "$acpi_report"
        
        # Group references by DSMIL group
        for group in {0..5}; do
            local group_refs=$(grep -c "DSMIL$group" "/tmp/dsmil_acpi_refs" || echo "0")
            echo "Group $group: $group_refs references" >> "$acpi_report"
        done
        
        log_info "Found $ref_count ACPI DSMIL references"
    else
        echo "No DSDT DSMIL references found or access denied" >> "$acpi_report"
        log_warn "Could not access ACPI DSDT or no DSMIL references found"
    fi
    
    # Check for ACPI device files
    if [[ -d "/sys/firmware/acpi" ]]; then
        find /sys/firmware/acpi -name "*DSMIL*" 2>/dev/null >> "$acpi_report" || true
    fi
    
    # Copy report to log
    cat "$acpi_report" >> "$LOG_FILE"
    
    return 0
}

# =============================================================================
# PROBE STATE MANAGEMENT
# =============================================================================

save_probe_state() {
    local phase="$1"
    local group="$2"
    local device="$3"  
    local action="$4"
    
    local timestamp=$(date +%s)
    local state_entry="$timestamp:$phase:$group:$device:$action"
    
    echo "$state_entry" >> "$PROBE_STATE_FILE"
    log_info "Probe state saved: $state_entry"
}

create_rollback_plan() {
    local phase="$1"
    
    log_info "Creating rollback plan for phase: $phase"
    
    echo "# DSMIL Rollback Plan - Phase $phase" > "$ROLLBACK_FILE"
    echo "# Generated: $(date)" >> "$ROLLBACK_FILE"
    echo "# Purpose: Emergency rollback instructions" >> "$ROLLBACK_FILE"
    echo "" >> "$ROLLBACK_FILE"
    
    case "$phase" in
        "passive")
            echo "# Phase: Passive Enumeration" >> "$ROLLBACK_FILE"
            echo "# Actions: None required - read-only operations" >> "$ROLLBACK_FILE"
            echo "ROLLBACK_REQUIRED=false" >> "$ROLLBACK_FILE"
            ;;
        "readonly")  
            echo "# Phase: Read-Only Device Queries" >> "$ROLLBACK_FILE"
            echo "# Actions: None required - no device state changes" >> "$ROLLBACK_FILE"
            echo "ROLLBACK_REQUIRED=false" >> "$ROLLBACK_FILE"
            ;;
        "single_device")
            echo "# Phase: Single Device Activation" >> "$ROLLBACK_FILE"
            echo "# Actions: Deactivate specific device" >> "$ROLLBACK_FILE"
            echo "ROLLBACK_REQUIRED=true" >> "$ROLLBACK_FILE"
            echo "DEVICE_TO_DEACTIVATE=\$(tail -1 $PROBE_STATE_FILE | cut -d: -f3-4)" >> "$ROLLBACK_FILE"
            echo "# Command: sudo milspec-control --deactivate-device \$DEVICE_TO_DEACTIVATE" >> "$ROLLBACK_FILE"
            ;;
        "group_coordination")
            echo "# Phase: Group Coordination" >> "$ROLLBACK_FILE"  
            echo "# Actions: Deactivate entire group" >> "$ROLLBACK_FILE"
            echo "ROLLBACK_REQUIRED=true" >> "$ROLLBACK_FILE"
            echo "GROUP_TO_DEACTIVATE=\$(tail -1 $PROBE_STATE_FILE | cut -d: -f3)" >> "$ROLLBACK_FILE"
            echo "# Command: sudo milspec-control --deactivate-group \$GROUP_TO_DEACTIVATE" >> "$ROLLBACK_FILE"
            ;;
        "multi_group")
            echo "# Phase: Multi-Group Operations" >> "$ROLLBACK_FILE"
            echo "# Actions: Emergency shutdown all groups" >> "$ROLLBACK_FILE"  
            echo "ROLLBACK_REQUIRED=true" >> "$ROLLBACK_FILE"
            echo "# Command: sudo milspec-control --emergency-shutdown" >> "$ROLLBACK_FILE"
            ;;
    esac
    
    echo "" >> "$ROLLBACK_FILE"
    echo "BACKUP_FILE=\$(cat /tmp/dsmil_last_backup 2>/dev/null || echo 'none')" >> "$ROLLBACK_FILE"
    
    log_info "Rollback plan created: $ROLLBACK_FILE"
}

execute_rollback() {
    local reason="$1"
    
    log_error "EXECUTING EMERGENCY ROLLBACK: $reason"
    
    if [[ ! -f "$ROLLBACK_FILE" ]]; then
        log_error "No rollback plan found - manual intervention required"
        return 1
    fi
    
    source "$ROLLBACK_FILE"
    
    if [[ "${ROLLBACK_REQUIRED:-false}" == "true" ]]; then
        # Attempt automated rollback based on phase
        if command -v milspec-control &> /dev/null; then
            log_info "Attempting automated DSMIL rollback..."
            
            # Try emergency shutdown first
            sudo milspec-control --emergency-disable-all 2>/dev/null || log_warn "Emergency disable failed"
            
            # Unload driver module
            sudo rmmod dell-milspec 2>/dev/null || log_warn "Module unload failed"
            
            log_info "Automated rollback attempted"
        else
            log_warn "milspec-control not available - manual rollback required"
        fi
    fi
    
    # Wait for system stabilization
    log_info "Waiting $STABILITY_WAIT_SECONDS seconds for system stabilization..."
    sleep "$STABILITY_WAIT_SECONDS"
    
    # Validate system health post-rollback
    if validate_system_health "basic"; then
        log_info "System health validated after rollback"
    else
        log_error "System health issues persist after rollback"
        return 1
    fi
    
    log_info "Emergency rollback completed"
    return 0
}

# =============================================================================
# MAIN PROBE PHASES
# =============================================================================

phase_passive_enumeration() {
    log_info "Starting Phase 1: Passive Enumeration"
    save_probe_state "passive" "all" "all" "start"
    create_rollback_plan "passive"
    
    # Device validation
    local device_status
    validate_dsmil_devices
    device_status=$?
    
    if (( device_status == 1 )); then
        log_error "No DSMIL devices found - cannot proceed"
        return 1
    elif (( device_status == 2 )); then
        log_warn "Partial device availability - proceeding with caution"
    fi
    
    # ACPI method probing
    probe_acpi_methods
    
    save_probe_state "passive" "all" "all" "complete"
    log_info "Phase 1 completed successfully"
    return 0
}

phase_readonly_queries() {
    log_info "Starting Phase 2: Read-Only Device Queries"
    save_probe_state "readonly" "all" "all" "start" 
    create_rollback_plan "readonly"
    
    # Check if driver is loaded
    if lsmod | grep -q dell.milspec; then
        log_info "dell-milspec driver is loaded"
        
        # Try basic status query if milspec-control is available
        if command -v milspec-control &> /dev/null; then
            log_info "Attempting basic status query..."
            if sudo milspec-control --status > "/tmp/dsmil_status_query" 2>&1; then
                log_info "Status query successful"
                cat "/tmp/dsmil_status_query" >> "$LOG_FILE"
            else
                log_warn "Status query failed"
            fi
        fi
        
        # Check sysfs interface
        if [[ -d "/sys/devices/platform/dell-milspec" ]]; then
            log_info "Found dell-milspec sysfs interface"
            find /sys/devices/platform/dell-milspec -type f -readable 2>/dev/null | while read -r file; do
                if [[ -r "$file" ]]; then
                    local content=$(cat "$file" 2>/dev/null || echo "N/A")
                    log_info "sysfs $file: $content"
                fi
            done
        fi
    else
        log_info "dell-milspec driver not loaded - safe read-only state"
    fi
    
    save_probe_state "readonly" "all" "all" "complete"
    log_info "Phase 2 completed successfully"
    return 0
}

phase_single_device_activation() {
    log_info "Starting Phase 3: Single Device Activation (MEDIUM RISK)"
    
    # Enhanced safety checks for activation phase
    if ! validate_system_health "comprehensive"; then
        log_error "System health validation failed - aborting activation phase"
        return 1
    fi
    
    save_probe_state "single_device" "0" "4" "start"  # DSMIL0D4 - Audit Logger
    create_rollback_plan "single_device"
    
    log_warn "This phase involves actual device activation - increased risk"
    log_info "Target: DSMIL0D4 (Audit Logger) - lowest risk device"
    
    # Start intensive monitoring
    ./dsmil_monitor.sh > "monitor_phase3_$(date +%H%M%S).log" &
    local monitor_pid=$!
    
    # Simulated single device activation (replace with actual implementation)
    log_info "Simulating single device activation..."
    sleep 10  # Simulate activation time
    
    # Check system stability
    log_info "Checking system stability after activation..."
    sleep "$STABILITY_WAIT_SECONDS"
    
    if validate_system_health "comprehensive"; then
        log_info "System remains stable after device activation"
        save_probe_state "single_device" "0" "4" "success"
    else
        log_error "System instability detected after activation"  
        save_probe_state "single_device" "0" "4" "failed"
        kill $monitor_pid 2>/dev/null || true
        execute_rollback "system instability"
        return 1
    fi
    
    kill $monitor_pid 2>/dev/null || true
    log_info "Phase 3 completed successfully"
    return 0
}

# =============================================================================
# MAIN EXECUTION FRAMEWORK
# =============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS] PHASE"
    echo ""
    echo "PHASES:"  
    echo "  passive         Phase 1: Passive enumeration (SAFE)"
    echo "  readonly        Phase 2: Read-only queries (LOW RISK)"  
    echo "  single-device   Phase 3: Single device activation (MEDIUM RISK)"
    echo "  group           Phase 4: Group coordination (HIGH RISK)"
    echo "  multi-group     Phase 5: Multi-group operations (CRITICAL)"
    echo ""
    echo "OPTIONS:"
    echo "  --force         Skip interactive confirmations"
    echo "  --dry-run       Simulate operations without actual changes"
    echo "  --health-only   Run health checks only"
    echo "  --rollback      Execute emergency rollback"
    echo "  --status        Show current probe status"
    echo "  --help          Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 passive                # Safe passive enumeration"
    echo "  $0 readonly               # Read-only device queries"
    echo "  $0 single-device --force  # Single device activation"
    echo "  $0 --health-only          # Health check only"
    echo "  $0 --rollback             # Emergency rollback"
}

show_status() {
    echo -e "${BLUE}═══ DSMIL Probe Status ═══${NC}"
    
    if [[ -f "$PROBE_STATE_FILE" ]]; then
        echo "Recent probe activities:"
        tail -10 "$PROBE_STATE_FILE" | while IFS=':' read -r timestamp phase group device action; do
            local date_str=$(date -d "@$timestamp" 2>/dev/null || echo "$timestamp")
            printf "  %s: Phase %s, Group %s, Device %s - %s\n" \
                   "$date_str" "$phase" "$group" "$device" "$action"
        done
    else
        echo "No probe activity recorded"
    fi
    
    echo ""
    echo "System Health:"
    printf "  Temperature: %s°C\n" "$(get_cpu_temperature)"
    printf "  Free Memory: %sMB\n" "$(get_free_memory_mb)"  
    printf "  System Load: %s\n" "$(get_system_load)"
    
    if [[ -f "/tmp/dsmil_device_report" ]]; then
        echo ""
        echo "Device Status:"
        grep "total:" "/tmp/dsmil_device_report" | head -6
    fi
}

confirm_risk_phase() {
    local phase="$1"
    local risk_level="$2"
    
    if [[ "${FORCE_MODE:-false}" == "true" ]]; then
        return 0
    fi
    
    echo -e "${YELLOW}⚠️  WARNING: You are about to enter $risk_level risk phase: $phase${NC}"
    echo ""
    echo "This phase may:"
    echo "• Activate military subsystem hardware"  
    echo "• Potentially cause system instability"
    echo "• Require emergency rollback procedures"
    echo ""
    echo "Prerequisites:"
    echo "• Full system backup created"
    echo "• Emergency rollback plan ready"
    echo "• System health validated"
    echo ""
    
    read -p "Are you sure you want to proceed? (type 'YES' to continue): " -r
    if [[ $REPLY == "YES" ]]; then
        return 0
    else
        echo "Operation cancelled by user"
        return 1
    fi
}

cleanup() {
    # Clean up temporary files
    rm -f /tmp/dsmil_acpi_refs /tmp/dsmil_device_report /tmp/dsmil_acpi_report
    rm -f /tmp/dsmil_status_query
    
    # Kill any background monitors
    pkill -f "dsmil_monitor.sh" 2>/dev/null || true
}

emergency_abort() {
    log_error "Emergency abort signal received"
    echo -e "${RED}🚨 EMERGENCY ABORT - Initiating rollback${NC}"
    
    execute_rollback "user abort signal"
    exit 1
}

main() {
    local phase=""
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_MODE=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --health-only)
                print_banner
                validate_system_health "comprehensive"
                exit $?
                ;;
            --rollback)
                print_banner  
                execute_rollback "user requested"
                exit $?
                ;;
            --status)
                show_status
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            passive|readonly|single-device|group|multi-group)
                phase="$1"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$phase" ]]; then
        echo "Error: No phase specified"
        show_usage
        exit 1
    fi
    
    print_banner
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Prerequisites check
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # System health validation
    if ! validate_system_health "comprehensive"; then
        log_error "System health validation failed - unsafe to proceed"
        exit 1
    fi
    
    # Create backup for activation phases
    if [[ "$phase" =~ ^(single-device|group|multi-group)$ ]]; then
        create_system_backup
    fi
    
    # Execute requested phase
    case "$phase" in
        passive)
            phase_passive_enumeration
            ;;
        readonly)
            phase_readonly_queries
            ;;
        single-device)
            if confirm_risk_phase "single device activation" "MEDIUM"; then
                phase_single_device_activation
            else
                exit 1
            fi
            ;;
        group)
            log_error "Group coordination phase not yet implemented"
            exit 1
            ;;
        multi-group) 
            log_error "Multi-group operations phase not yet implemented"
            exit 1
            ;;
        *)
            log_error "Unknown phase: $phase"
            exit 1
            ;;
    esac
    
    log_info "Phase '$phase' completed successfully"
    echo -e "${GREEN}✅ DSMIL probe phase '$phase' completed safely${NC}"
}

# Execute main function with all arguments
main "$@"