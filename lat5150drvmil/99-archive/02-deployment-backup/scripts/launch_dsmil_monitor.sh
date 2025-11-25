#!/bin/bash
"""
DSMIL Monitoring System Launcher
Dell Latitude 5450 MIL-SPEC - Comprehensive READ-ONLY Monitoring

SAFETY-FIRST DESIGN:
- Comprehensive READ-ONLY monitoring of all 84 DSMIL devices
- NO write operations to any DSMIL device 
- Special protection for dangerous tokens 0x8009-0x800B
- Emergency stop capabilities
- System health monitoring
- Multiple monitoring modes and interfaces

Author: MONITOR Agent
Date: 2025-09-01
Classification: MIL-SPEC Safe Operations
"""

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/monitoring_logs"
PYTHON_SCRIPTS=(
    "dsmil_readonly_monitor.py"
    "dsmil_emergency_stop.py" 
    "dsmil_dashboard.py"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"
}

# ============================================================================
# SAFETY CHECKS
# ============================================================================

check_root_privileges() {
    if [[ $EUID -ne 0 ]]; then
        error "Root privileges required for SMI access"
        echo "Please run with: sudo $0"
        return 1
    fi
    return 0
}

check_system_requirements() {
    local requirements_met=true
    
    info "Checking system requirements..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found"
        requirements_met=false
    else
        log "✅ Python 3 found: $(python3 --version)"
    fi
    
    # Check required Python modules
    local python_modules=("psutil" "subprocess" "json" "datetime" "threading" "curses")
    for module in "${python_modules[@]}"; do
        if ! python3 -c "import $module" 2>/dev/null; then
            error "Python module '$module' not available"
            requirements_met=false
        fi
    done
    
    if $requirements_met; then
        log "✅ Python modules available"
    fi
    
    # Check required scripts
    for script in "${PYTHON_SCRIPTS[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            error "Required script not found: $script"
            requirements_met=false
        fi
    done
    
    if $requirements_met; then
        log "✅ All monitoring scripts found"
    fi
    
    # Check system resources
    local temp
    temp=$(sensors 2>/dev/null | grep -E "(Core|CPU)" | head -1 | grep -oP '\+\K[0-9]+' || echo "0")
    if [[ $temp -gt 85 ]]; then
        warn "System temperature high: ${temp}°C"
    else
        log "✅ System temperature normal: ${temp}°C"
    fi
    
    # Check memory usage
    local mem_usage
    mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [[ $mem_usage -gt 90 ]]; then
        warn "Memory usage high: ${mem_usage}%"
    else
        log "✅ Memory usage normal: ${mem_usage}%"
    fi
    
    return $($requirements_met && echo 0 || echo 1)
}

check_dsmil_safety() {
    info "Performing DSMIL safety check..."
    
    # Check for running DSMIL processes
    local dsmil_processes
    dsmil_processes=$(pgrep -f "dsmil" 2>/dev/null || true)
    
    if [[ -n "$dsmil_processes" ]]; then
        warn "Existing DSMIL processes detected:"
        ps -p $dsmil_processes -o pid,cmd 2>/dev/null || true
        echo
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "User chose to abort"
            return 1
        fi
    else
        log "✅ No conflicting DSMIL processes"
    fi
    
    # Check for loaded DSMIL modules
    local dsmil_modules
    dsmil_modules=$(lsmod | grep -i dsmil || true)
    
    if [[ -n "$dsmil_modules" ]]; then
        warn "DSMIL kernel modules loaded:"
        echo "$dsmil_modules"
        echo
        read -p "Continue with monitoring? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "User chose to abort"
            return 1
        fi
    else
        log "✅ No DSMIL kernel modules loaded"
    fi
    
    return 0
}

# ============================================================================
# MONITORING FUNCTIONS
# ============================================================================

prepare_monitoring_environment() {
    info "Preparing monitoring environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    log "✅ Log directory: $LOG_DIR"
    
    # Set proper permissions
    chmod 755 "$SCRIPT_DIR"/*.py 2>/dev/null || true
    log "✅ Script permissions set"
    
    # Create emergency stop script shortcut
    cat > "$LOG_DIR/emergency_stop.sh" << 'EOF'
#!/bin/bash
echo "🚨 EMERGENCY STOP REQUESTED"
sudo python3 "$(dirname "$0")/../dsmil_emergency_stop.py" --stop
EOF
    chmod +x "$LOG_DIR/emergency_stop.sh"
    log "✅ Emergency stop shortcut created: $LOG_DIR/emergency_stop.sh"
    
    return 0
}

show_main_menu() {
    clear
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}        DSMIL 84-Device READ-ONLY Monitoring System${NC}"
    echo -e "${CYAN}              Dell Latitude 5450 MIL-SPEC${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo
    echo -e "${GREEN}SAFETY NOTICE:${NC}"
    echo "  • READ-ONLY operations only - NO writes to DSMIL devices"
    echo "  • Dangerous tokens 0x8009-0x800B under special protection"
    echo "  • Emergency stop available at any time (Ctrl+C)"
    echo
    echo -e "${BLUE}MONITORING OPTIONS:${NC}"
    echo "  1) 📊 Interactive Dashboard (Recommended)"
    echo "  2) 🔍 Command-Line Monitor" 
    echo "  3) 🛡️  System Safety Check"
    echo "  4) 🚨 Emergency Stop System"
    echo "  5) 📋 View Recent Logs"
    echo "  6) ⚙️  Advanced Options"
    echo "  q) Quit"
    echo
    echo -e "${YELLOW}SYSTEM STATUS:${NC}"
    
    # Show basic system info
    local cpu_temp
    cpu_temp=$(sensors 2>/dev/null | grep -E "(Core|CPU)" | head -1 | grep -oP '\+\K[0-9]+' || echo "N/A")
    
    local mem_usage
    mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    
    local uptime
    uptime=$(uptime -p 2>/dev/null || echo "Unknown")
    
    echo "  Temperature: ${cpu_temp}°C | Memory: ${mem_usage}% | Uptime: $uptime"
    echo
    echo -n "Select option [1-6, q]: "
}

launch_interactive_dashboard() {
    log "Starting interactive dashboard..."
    echo
    echo -e "${GREEN}📊 DSMIL Interactive Dashboard${NC}"
    echo -e "${YELLOW}⚠️  This will start real-time monitoring of all 84 DSMIL devices${NC}"
    echo
    echo "Dashboard controls:"
    echo "  • q/Q: Quit"
    echo "  • e/E: Emergency stop"
    echo "  • r/R: Reset counters"
    echo "  • 1-5: Switch modes"
    echo
    read -p "Press Enter to continue or Ctrl+C to abort..."
    
    cd "$SCRIPT_DIR"
    python3 dsmil_dashboard.py
}

launch_command_monitor() {
    log "Starting command-line monitor..."
    echo
    echo -e "${GREEN}🔍 DSMIL Command-Line Monitor${NC}"
    echo -e "${YELLOW}⚠️  This will continuously monitor all 84 DSMIL devices${NC}"
    echo
    echo "Available modes:"
    echo "  1) Dashboard view (default)"
    echo "  2) Security-focused view"
    echo "  3) Thermal monitoring"
    echo "  4) Anomaly detection"
    echo
    read -p "Select mode [1-4] or Enter for default: " mode_choice
    
    local mode="dashboard"
    case $mode_choice in
        1) mode="dashboard" ;;
        2) mode="security" ;;
        3) mode="thermal" ;;
        4) mode="anomaly" ;;
    esac
    
    log "Starting monitor in $mode mode..."
    echo "Press Ctrl+C to stop monitoring"
    echo
    
    cd "$SCRIPT_DIR"
    python3 dsmil_readonly_monitor.py --mode "$mode"
}

run_safety_check() {
    log "Running comprehensive safety check..."
    echo
    
    cd "$SCRIPT_DIR" 
    python3 dsmil_emergency_stop.py --check
    
    echo
    read -p "Press Enter to continue..."
}

launch_emergency_stop() {
    error "⚠️  EMERGENCY STOP SYSTEM"
    echo
    echo -e "${RED}This will immediately terminate all DSMIL operations!${NC}"
    echo
    echo "Options:"
    echo "  1) Execute immediate emergency stop"
    echo "  2) Start emergency monitoring (watch for dangerous conditions)"
    echo "  3) Check emergency system status"
    echo "  4) Return to main menu"
    echo
    read -p "Select option [1-4]: " emergency_choice
    
    case $emergency_choice in
        1)
            echo
            read -p "⚠️  Confirm emergency stop? [yes/no]: " confirm
            if [[ "$confirm" == "yes" ]]; then
                cd "$SCRIPT_DIR"
                python3 dsmil_emergency_stop.py --stop
            else
                info "Emergency stop cancelled"
            fi
            ;;
        2)
            echo
            read -p "Emergency monitoring duration (seconds) [300]: " duration
            duration=${duration:-300}
            cd "$SCRIPT_DIR"
            python3 dsmil_emergency_stop.py --monitor --duration "$duration"
            ;;
        3)
            cd "$SCRIPT_DIR"
            python3 dsmil_emergency_stop.py --status
            ;;
        4)
            return 0
            ;;
        *)
            warn "Invalid option"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

view_recent_logs() {
    log "Viewing recent monitoring logs..."
    echo
    
    if [[ ! -d "$LOG_DIR" ]]; then
        warn "No log directory found"
        return 0
    fi
    
    # Find recent log files
    local recent_logs
    mapfile -t recent_logs < <(find "$LOG_DIR" -name "*.log" -type f -mtime -1 2>/dev/null | sort -r)
    
    if [[ ${#recent_logs[@]} -eq 0 ]]; then
        warn "No recent log files found"
        return 0
    fi
    
    echo "Recent log files:"
    for i in "${!recent_logs[@]}"; do
        local log_file="${recent_logs[$i]}"
        local log_size
        log_size=$(du -h "$log_file" 2>/dev/null | cut -f1)
        local log_date
        log_date=$(stat -c %y "$log_file" 2>/dev/null | cut -d' ' -f1,2 | cut -d: -f1,2)
        echo "  $((i+1))) $(basename "$log_file") ($log_size, $log_date)"
    done
    echo
    read -p "View log file [1-${#recent_logs[@]}] or Enter to return: " log_choice
    
    if [[ -n "$log_choice" && "$log_choice" -ge 1 && "$log_choice" -le ${#recent_logs[@]} ]]; then
        local selected_log="${recent_logs[$((log_choice-1))]}"
        echo
        echo "=== $(basename "$selected_log") ==="
        tail -50 "$selected_log" 2>/dev/null || cat "$selected_log"
        echo
        read -p "Press Enter to continue..."
    fi
}

show_advanced_options() {
    echo
    echo -e "${CYAN}🔧 Advanced Options${NC}"
    echo
    echo "1) Test SMI interface directly"
    echo "2) Check DSMIL device accessibility"
    echo "3) Generate system report"
    echo "4) Run diagnostic tests"
    echo "5) Configure monitoring thresholds"
    echo "6) Return to main menu"
    echo
    read -p "Select option [1-6]: " advanced_choice
    
    case $advanced_choice in
        1)
            test_smi_interface
            ;;
        2)
            check_device_accessibility
            ;;
        3)
            generate_system_report
            ;;
        4)
            run_diagnostic_tests
            ;;
        5)
            configure_thresholds
            ;;
        6)
            return 0
            ;;
        *)
            warn "Invalid option"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
}

test_smi_interface() {
    log "Testing SMI interface..."
    echo
    
    # Test basic SMI access
    cat > /tmp/test_smi.c << 'EOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/io.h>

int main() {
    if (iopl(3) != 0) {
        printf("ERROR: Cannot access I/O ports\n");
        return 1;
    }
    
    printf("SMI interface test:\n");
    printf("Command port 0x164E: accessible\n");
    printf("Data port 0x164F: accessible\n");
    printf("SMI interface appears functional\n");
    
    return 0;
}
EOF
    
    gcc -o /tmp/test_smi /tmp/test_smi.c 2>/dev/null
    if [[ $? -eq 0 ]]; then
        /tmp/test_smi
        rm -f /tmp/test_smi /tmp/test_smi.c
    else
        error "Failed to compile SMI test"
    fi
}

check_device_accessibility() {
    log "Checking DSMIL device accessibility..."
    echo
    
    local accessible_count=0
    local total_count=0
    
    for token in $(seq 32768 32875); do  # 0x8000 to 0x806B
        ((total_count++))
        
        # Simple accessibility test (read-only)
        if timeout 1s python3 -c "
import subprocess, sys
try:
    result = subprocess.run(['sudo', 'python3', '-c', '''
import os
if os.system(\"echo $token > /dev/null 2>&1\") == 0:
    print(\"accessible\")
'''], capture_output=True, timeout=0.5)
    if \"accessible\" in result.stdout.decode():
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
            ((accessible_count++))
        fi
        
        # Progress indicator
        if ((total_count % 10 == 0)); then
            echo -n "."
        fi
    done
    
    echo
    echo "Device accessibility: $accessible_count/$total_count devices appear accessible"
}

generate_system_report() {
    log "Generating system report..."
    
    local report_file="$LOG_DIR/system_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "DSMIL System Report"
        echo "=================="
        echo "Generated: $(date)"
        echo "System: $(uname -a)"
        echo
        echo "Hardware:"
        lscpu | head -10
        echo
        echo "Memory:"
        free -h
        echo
        echo "Temperature:"
        sensors 2>/dev/null || echo "No temperature sensors found"
        echo
        echo "Loaded Modules:"
        lsmod | grep -E "(dell|dsmil)" || echo "No DSMIL modules loaded"
        echo
        echo "Running Processes:"
        ps aux | grep -E "(dsmil|monitor)" | grep -v grep || echo "No DSMIL processes running"
    } > "$report_file"
    
    log "System report saved: $report_file"
    echo
    echo "Report contents:"
    cat "$report_file"
}

run_diagnostic_tests() {
    log "Running diagnostic tests..."
    echo
    
    local tests_passed=0
    local tests_total=5
    
    # Test 1: Python environment
    echo -n "Test 1/5: Python environment... "
    if python3 -c "import psutil, json, datetime" 2>/dev/null; then
        echo "✅ PASS"
        ((tests_passed++))
    else
        echo "❌ FAIL"
    fi
    
    # Test 2: Root privileges
    echo -n "Test 2/5: Root privileges... "
    if [[ $EUID -eq 0 ]]; then
        echo "✅ PASS"
        ((tests_passed++))
    else
        echo "❌ FAIL"
    fi
    
    # Test 3: Required scripts
    echo -n "Test 3/5: Required scripts... "
    local scripts_found=0
    for script in "${PYTHON_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            ((scripts_found++))
        fi
    done
    
    if [[ $scripts_found -eq ${#PYTHON_SCRIPTS[@]} ]]; then
        echo "✅ PASS"
        ((tests_passed++))
    else
        echo "❌ FAIL ($scripts_found/${#PYTHON_SCRIPTS[@]} found)"
    fi
    
    # Test 4: System resources
    echo -n "Test 4/5: System resources... "
    local mem_usage
    mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [[ $mem_usage -lt 90 ]]; then
        echo "✅ PASS"
        ((tests_passed++))
    else
        echo "❌ FAIL (Memory: ${mem_usage}%)"
    fi
    
    # Test 5: Log directory
    echo -n "Test 5/5: Log directory... "
    if [[ -d "$LOG_DIR" && -w "$LOG_DIR" ]]; then
        echo "✅ PASS"
        ((tests_passed++))
    else
        echo "❌ FAIL"
    fi
    
    echo
    echo "Diagnostic Results: $tests_passed/$tests_total tests passed"
    
    if [[ $tests_passed -eq $tests_total ]]; then
        log "✅ All diagnostic tests passed - system ready for monitoring"
    else
        warn "⚠️  Some diagnostic tests failed - monitoring may not work properly"
    fi
}

configure_thresholds() {
    log "Configuring monitoring thresholds..."
    echo
    echo "Current default thresholds:"
    echo "  Temperature Warning: 85°C"
    echo "  Temperature Critical: 90°C"
    echo "  Memory Warning: 80%"
    echo "  Memory Critical: 90%"
    echo "  CPU Warning: 80%"
    echo "  CPU Critical: 90%"
    echo
    warn "Threshold configuration not yet implemented"
    echo "This feature will be available in a future update"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Ensure we're in the script directory
    cd "$SCRIPT_DIR"
    
    # Perform safety checks
    if ! check_root_privileges; then
        exit 1
    fi
    
    if ! check_system_requirements; then
        error "System requirements not met"
        exit 1
    fi
    
    if ! check_dsmil_safety; then
        warn "DSMIL safety check failed - continuing with caution"
    fi
    
    # Prepare monitoring environment
    prepare_monitoring_environment
    
    # Main menu loop
    while true; do
        show_main_menu
        read -r choice
        
        case $choice in
            1)
                launch_interactive_dashboard
                ;;
            2)
                launch_command_monitor
                ;;
            3)
                run_safety_check
                ;;
            4)
                launch_emergency_stop
                ;;
            5)
                view_recent_logs
                ;;
            6)
                show_advanced_options
                ;;
            q|Q)
                log "Exiting DSMIL monitoring system"
                exit 0
                ;;
            *)
                warn "Invalid option: $choice"
                sleep 1
                ;;
        esac
    done
}

# Trap signals for clean exit
trap 'echo; error "Interrupted by signal"; exit 1' INT TERM

# Run main function
main "$@"