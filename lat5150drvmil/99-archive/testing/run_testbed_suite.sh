#!/bin/bash
"""
SMBIOS Token Testing Suite - Master Control Script
==================================================

Master control script for the complete SMBIOS token testing framework
on Dell Latitude 5450 MIL-SPEC systems. Provides unified interface
for all testing operations with comprehensive safety controls.

Features:
- Automated system preparation and validation
- Complete testing suite execution
- Real-time monitoring and safety controls
- Comprehensive reporting and analysis
- Ubuntu 24.04 and Debian Trixie compatibility
- Emergency stop capabilities

Author: TESTBED Agent
Version: 1.0.0
Date: 2025-09-01
"""

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
TESTING_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${CYAN}[${timestamp}] ℹ️  ${message}${NC}" ;;
        "SUCCESS") echo -e "${GREEN}[${timestamp}] ✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}[${timestamp}] ⚠️  ${message}${NC}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] ❌ ${message}${NC}" ;;
        "CRITICAL") echo -e "${RED}[${timestamp}] 🚨 ${message}${NC}" ;;
        *) echo -e "${WHITE}[${timestamp}] ${message}${NC}" ;;
    esac
}

# Banner function
show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    SMBIOS Token Testing Suite v1.0.0                            ║
║                   Dell Latitude 5450 MIL-SPEC - TESTBED Agent                   ║
║                                                                                  ║
║  🧪 Systematic SMBIOS token testing with comprehensive safety controls          ║
║  🔒 Real-time thermal monitoring and emergency stop capabilities                 ║
║  📊 Advanced correlation analysis and reporting                                  ║
║  🛡️ Ubuntu 24.04 and Debian Trixie compatibility                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# System information
show_system_info() {
    log "INFO" "System Information:"
    echo "  Hardware: $(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'Unknown')"
    echo "  Kernel: $(uname -r)"
    echo "  Distribution: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    echo "  Python: $(python3 --version)"
    echo "  Working Directory: $WORK_DIR"
    echo "  Testing Directory: $TESTING_DIR"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking system prerequisites..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check required Python modules
    if ! python3 -c "import psutil" 2>/dev/null; then
        log "WARNING" "psutil module missing - will be installed automatically"
    fi
    
    # Check system tools
    local required_tools=("gcc" "make" "sudo" "modinfo" "lsmod")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    # Check SMBIOS tools (optional but recommended)
    if ! command -v smbios-token-ctl &> /dev/null; then
        log "WARNING" "smbios-token-ctl not found - install libsmbios-bin package"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "ERROR" "Missing required dependencies: ${missing_deps[*]}"
        echo "Install with: sudo apt update && sudo apt install -y ${missing_deps[*]}"
        return 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
    return 0
}

# Run system safety validation
run_safety_validation() {
    log "INFO" "Running comprehensive system safety validation..."
    
    if ! python3 "$TESTING_DIR/safety_validator.py"; then
        log "CRITICAL" "Safety validation failed!"
        echo ""
        echo -e "${RED}SAFETY VALIDATION FAILED${NC}"
        echo "The system is not safe for SMBIOS token testing."
        echo "Please resolve all safety issues before proceeding."
        echo ""
        return 1
    fi
    
    log "SUCCESS" "Safety validation passed"
    return 0
}

# Check distribution compatibility
check_compatibility() {
    log "INFO" "Checking distribution compatibility..."
    
    if ! python3 "$TESTING_DIR/debian_compatibility.py"; then
        log "WARNING" "Compatibility issues detected"
        echo ""
        echo -e "${YELLOW}COMPATIBILITY ISSUES DETECTED${NC}"
        echo "Some components may not work optimally on this distribution."
        echo ""
        
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "INFO" "Testing aborted by user"
            return 1
        fi
    fi
    
    log "SUCCESS" "Distribution compatibility verified"
    return 0
}

# Start monitoring systems
start_monitoring() {
    log "INFO" "Starting monitoring systems..."
    
    # Check if monitoring directory exists
    local monitor_dir="$WORK_DIR/monitoring"
    if [ ! -d "$monitor_dir" ]; then
        log "WARNING" "Monitoring directory not found at $monitor_dir"
        return 0
    fi
    
    # Start comprehensive monitor if available
    local monitor_script="$monitor_dir/dsmil_comprehensive_monitor.py"
    if [ -f "$monitor_script" ]; then
        log "INFO" "Starting DSMIL comprehensive monitor..."
        python3 "$monitor_script" &
        local monitor_pid=$!
        echo "$monitor_pid" > "$TESTING_DIR/.monitor.pid"
        log "SUCCESS" "Monitor started (PID: $monitor_pid)"
    fi
    
    # Note about multi-terminal launcher
    local launcher_script="$monitor_dir/multi_terminal_launcher.sh"
    if [ -f "$launcher_script" ]; then
        log "INFO" "Multi-terminal monitoring available at: $launcher_script"
        log "INFO" "Run manually in separate terminal for visual monitoring"
    fi
    
    return 0
}

# Stop monitoring systems
stop_monitoring() {
    log "INFO" "Stopping monitoring systems..."
    
    # Stop monitor if running
    local monitor_pid_file="$TESTING_DIR/.monitor.pid"
    if [ -f "$monitor_pid_file" ]; then
        local monitor_pid=$(cat "$monitor_pid_file")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            log "SUCCESS" "Monitor stopped (PID: $monitor_pid)"
        fi
        rm -f "$monitor_pid_file"
    fi
    
    return 0
}

# Emergency stop function
emergency_stop() {
    log "CRITICAL" "EMERGENCY STOP ACTIVATED!"
    
    # Stop monitoring
    stop_monitoring
    
    # Unload DSMIL module if loaded
    if lsmod | grep -q dsmil; then
        log "INFO" "Unloading DSMIL module..."
        sudo rmmod dsmil-72dev 2>/dev/null || true
    fi
    
    # Run emergency stop script if available
    local emergency_script="$WORK_DIR/monitoring/emergency_stop.sh"
    if [ -f "$emergency_script" ] && [ -x "$emergency_script" ]; then
        log "INFO" "Executing emergency stop script..."
        "$emergency_script"
    fi
    
    log "SUCCESS" "Emergency stop complete"
}

# Trap function for cleanup
cleanup_on_exit() {
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        log "WARNING" "Script exiting with error code $exit_code"
        emergency_stop
    else
        stop_monitoring
    fi
    
    exit $exit_code
}

# Set up signal handling
trap cleanup_on_exit EXIT
trap emergency_stop INT TERM

# Main menu
show_menu() {
    echo -e "${WHITE}Available Testing Options:${NC}"
    echo "1. 🔍 System Validation Only"
    echo "2. 🧪 Single Token Test (0x0480)"
    echo "3. 🎯 Single Group Test (Group 0: 12 tokens)"
    echo "4. 📊 Range Test (Range 0x0480: 72 tokens)"
    echo "5. 🚀 Full Comprehensive Test (All ranges)"
    echo "6. 📋 Generate Report from Existing Data"
    echo "7. 🔧 System Compatibility Check"
    echo "8. ❌ Exit"
    echo ""
}

# Run single token test
run_single_token_test() {
    log "INFO" "Starting single token test..."
    
    if ! python3 -c "
import sys; sys.path.append('$TESTING_DIR')
from smbios_testbed_framework import SMBIOSTokenTester
tester = SMBIOSTokenTester('$WORK_DIR')
session = tester.create_test_session('single_token')
result = tester.test_single_token(0x0480)
tester.save_test_results()
print('Single token test completed')
"; then
        log "ERROR" "Single token test failed"
        return 1
    fi
    
    log "SUCCESS" "Single token test completed"
    return 0
}

# Run orchestrated test
run_orchestrated_test() {
    local scenario="$1"
    log "INFO" "Starting orchestrated test: $scenario"
    
    if ! python3 "$TESTING_DIR/orchestrate_token_testing.py" <<< "$scenario"; then
        log "ERROR" "Orchestrated test failed"
        return 1
    fi
    
    log "SUCCESS" "Orchestrated test completed"
    return 0
}

# Generate comprehensive report
generate_report() {
    log "INFO" "Generating comprehensive test report..."
    
    if ! python3 "$TESTING_DIR/comprehensive_test_reporter.py"; then
        log "ERROR" "Report generation failed"
        return 1
    fi
    
    log "SUCCESS" "Comprehensive report generated"
    
    # Show report location
    local report_dir="$TESTING_DIR/reports"
    if [ -d "$report_dir" ]; then
        log "INFO" "Reports available in: $report_dir"
        ls -la "$report_dir"/*.html 2>/dev/null || true
    fi
    
    return 0
}

# Main function
main() {
    show_banner
    show_system_info
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Main loop
    while true; do
        show_menu
        read -p "Select option (1-8): " choice
        echo ""
        
        case $choice in
            1)
                log "INFO" "Running system validation..."
                check_compatibility
                run_safety_validation
                ;;
            2)
                log "INFO" "Preparing for single token test..."
                if check_compatibility && run_safety_validation; then
                    start_monitoring
                    run_single_token_test
                    stop_monitoring
                    generate_report
                fi
                ;;
            3)
                log "INFO" "Preparing for group test..."
                if check_compatibility && run_safety_validation; then
                    start_monitoring
                    echo "2" | python3 "$TESTING_DIR/orchestrate_token_testing.py"
                    stop_monitoring
                    generate_report
                fi
                ;;
            4)
                log "INFO" "Preparing for range test..."
                if check_compatibility && run_safety_validation; then
                    echo -e "${YELLOW}This will test 72 tokens (approximately 45 minutes)${NC}"
                    read -p "Continue? (y/N): " -r
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        start_monitoring
                        echo "3" | python3 "$TESTING_DIR/orchestrate_token_testing.py"
                        stop_monitoring
                        generate_report
                    fi
                fi
                ;;
            5)
                log "INFO" "Preparing for comprehensive test..."
                if check_compatibility && run_safety_validation; then
                    echo -e "${RED}This will test ALL token ranges (approximately 2+ hours)${NC}"
                    read -p "Are you sure? (y/N): " -r
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        start_monitoring
                        echo "4" | python3 "$TESTING_DIR/orchestrate_token_testing.py"
                        stop_monitoring
                        generate_report
                    fi
                fi
                ;;
            6)
                generate_report
                ;;
            7)
                check_compatibility
                ;;
            8)
                log "INFO" "Exiting SMBIOS testing suite"
                exit 0
                ;;
            *)
                log "ERROR" "Invalid option. Please select 1-8."
                ;;
        esac
        
        echo ""
        echo -e "${WHITE}Press Enter to continue...${NC}"
        read -r
        echo ""
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi