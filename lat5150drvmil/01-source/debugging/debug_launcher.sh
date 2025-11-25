#!/bin/bash
# DSMIL Debug Infrastructure Launcher
# Provides convenient access to all debugging tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_BASE_DIR="/tmp/dsmil_unified_debug"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking system prerequisites..."
    
    # Check if running on Dell Latitude 5450
    local model
    model=$(sudo dmidecode -s system-product-name 2>/dev/null || echo "Unknown")
    if [[ "$model" != *"Latitude 5450"* ]]; then
        warn "System model: $model (Expected: Dell Latitude 5450)"
    else
        info "✓ Dell Latitude 5450 detected"
    fi
    
    # Check kernel version
    local kernel_version
    kernel_version=$(uname -r)
    if [[ "$kernel_version" < "6.14" ]]; then
        error "Kernel version $kernel_version < 6.14.0 (required)"
        return 1
    else
        info "✓ Kernel version: $kernel_version"
    fi
    
    # Check if DSMIL module is loaded
    if lsmod | grep -q "dsmil_72dev"; then
        info "✓ DSMIL kernel module loaded"
    else
        warn "DSMIL kernel module not loaded - attempting to load..."
        if sudo modprobe dsmil-72dev 2>/dev/null; then
            info "✓ DSMIL kernel module loaded successfully"
        else
            error "Failed to load DSMIL kernel module"
            return 1
        fi
    fi
    
    # Check Python dependencies
    local deps=("numpy" "psutil")
    for dep in "${deps[@]}"; do
        if python3 -c "import $dep" 2>/dev/null; then
            info "✓ Python dependency: $dep"
        else
            warn "Missing Python dependency: $dep"
            echo "Install with: pip3 install $dep"
        fi
    done
    
    # Check permissions
    if [[ $EUID -eq 0 ]]; then
        info "✓ Running as root (full monitoring capabilities)"
    else
        warn "Running as user (limited monitoring capabilities)"
        info "For full system call tracing, run with sudo"
    fi
    
    # Check disk space
    local available_space
    available_space=$(df /tmp | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 1048576 ]]; then  # 1GB in KB
        warn "Low disk space in /tmp: ${available_space}KB"
    else
        info "✓ Sufficient disk space: ${available_space}KB available"
    fi
    
    log "Prerequisite check completed"
}

# Display system status
show_system_status() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    DSMIL System Status                        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    
    # System information
    echo -e "${BLUE}System Information:${NC}"
    echo "  Model: $(sudo dmidecode -s system-product-name 2>/dev/null || echo 'Unknown')"
    echo "  Kernel: $(uname -r)"
    echo "  Uptime: $(uptime -p)"
    echo "  Temperature: $(sensors | grep 'Core 0' | awk '{print $3}' | head -1 2>/dev/null || echo 'N/A')"
    echo
    
    # DSMIL module status
    echo -e "${BLUE}DSMIL Module Status:${NC}"
    if lsmod | grep -q "dsmil_72dev"; then
        local module_info
        module_info=$(lsmod | grep "dsmil_72dev")
        echo "  Status: ${GREEN}LOADED${NC}"
        echo "  Info: $module_info"
        
        # Module parameters
        if [[ -d "/sys/module/dsmil_72dev/parameters" ]]; then
            echo "  Parameters:"
            for param in /sys/module/dsmil_72dev/parameters/*; do
                if [[ -r "$param" ]]; then
                    local param_name param_value
                    param_name=$(basename "$param")
                    param_value=$(cat "$param")
                    echo "    $param_name: $param_value"
                fi
            done
        fi
    else
        echo "  Status: ${RED}NOT LOADED${NC}"
    fi
    echo
    
    # Memory information
    echo -e "${BLUE}Memory Status:${NC}"
    echo "  Total: $(free -h | awk '/^Mem:/{print $2}')"
    echo "  Available: $(free -h | awk '/^Mem:/{print $7}')"
    echo "  DSMIL Region: 0x52000000-0x68800000 (360MB)"
    echo
    
    # Debug output status
    echo -e "${BLUE}Debug Output:${NC}"
    echo "  Base directory: $DEBUG_BASE_DIR"
    if [[ -d "$DEBUG_BASE_DIR" ]]; then
        local file_count size
        file_count=$(find "$DEBUG_BASE_DIR" -type f | wc -l)
        size=$(du -sh "$DEBUG_BASE_DIR" 2>/dev/null | cut -f1)
        echo "  Existing files: $file_count ($size)"
    else
        echo "  Status: Directory does not exist (will be created)"
    fi
}

# Interactive menu
show_menu() {
    echo
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              DSMIL Debug Infrastructure Menu                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo "Debug Session Options:"
    echo "  ${GREEN}1${NC}) Quick Debug Session (5 minutes, all components)"
    echo "  ${GREEN}2${NC}) Interactive Debug Session (manual control)"
    echo "  ${GREEN}3${NC}) Token Range Testing (specify range)"
    echo "  ${GREEN}4${NC}) Extended Debug Session (30 minutes)"
    echo
    echo "Individual Component Testing:"
    echo "  ${GREEN}5${NC}) Infrastructure Debugger Only"
    echo "  ${GREEN}6${NC}) Kernel Trace Analyzer Only"
    echo "  ${GREEN}7${NC}) Memory Pattern Analyzer Only"
    echo "  ${GREEN}8${NC}) Correlation Engine Only"
    echo
    echo "System Operations:"
    echo "  ${GREEN}9${NC}) System Status Check"
    echo "  ${GREEN}10${NC}) Clean Debug Output Directory"
    echo "  ${GREEN}11${NC}) View Recent Debug Reports"
    echo
    echo "  ${GREEN}s${NC}) Show system status"
    echo "  ${GREEN}h${NC}) Show help information"
    echo "  ${GREEN}q${NC}) Quit"
    echo
}

# Execute unified debugging session
run_unified_debug() {
    local duration=$1
    local extra_args=$2
    
    log "Starting unified debug session (${duration}s)..."
    
    cd "$SCRIPT_DIR"
    python3 unified_debug_orchestrator.py \
        --duration "$duration" \
        --output-dir "$DEBUG_BASE_DIR" \
        $extra_args
}

# Execute token testing
run_token_testing() {
    local token_range=$1
    local duration=${2:-120}
    
    log "Starting token testing session..."
    log "Token range: $token_range"
    log "Duration: ${duration}s"
    
    cd "$SCRIPT_DIR"
    python3 unified_debug_orchestrator.py \
        --test-tokens "$token_range" \
        --duration "$duration" \
        --output-dir "$DEBUG_BASE_DIR"
}

# Execute individual component
run_component() {
    local component=$1
    local duration=${2:-300}
    
    case $component in
        "infrastructure")
            log "Starting Infrastructure Debugger..."
            cd "$SCRIPT_DIR"
            python3 dsmil_debug_infrastructure.py --interactive
            ;;
        "kernel")
            log "Starting Kernel Trace Analyzer..."
            cd "$SCRIPT_DIR"
            python3 kernel_trace_analyzer.py --trace "$duration" --output-dir "$DEBUG_BASE_DIR/kernel_trace"
            ;;
        "memory")
            log "Starting Memory Pattern Analyzer..."
            cd "$SCRIPT_DIR"
            python3 memory_pattern_analyzer.py --monitor "$duration" --output-dir "$DEBUG_BASE_DIR/memory_analysis"
            ;;
        "correlation")
            log "Starting Correlation Engine..."
            cd "$SCRIPT_DIR"
            python3 smbios_correlation_engine.py --monitor "$duration" --database "$DEBUG_BASE_DIR/correlation.db"
            ;;
        *)
            error "Unknown component: $component"
            return 1
            ;;
    esac
}

# Clean debug output directory
clean_debug_output() {
    if [[ -d "$DEBUG_BASE_DIR" ]]; then
        echo -n "Delete all debug output in $DEBUG_BASE_DIR? [y/N]: "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            log "Cleaning debug output directory..."
            rm -rf "$DEBUG_BASE_DIR"/*
            info "✓ Debug output cleaned"
        else
            info "Operation cancelled"
        fi
    else
        info "Debug output directory does not exist"
    fi
}

# View recent debug reports
view_recent_reports() {
    if [[ ! -d "$DEBUG_BASE_DIR" ]]; then
        warn "No debug output directory found"
        return
    fi
    
    echo -e "${BLUE}Recent Debug Reports:${NC}"
    
    # Find recent unified reports
    local unified_reports
    mapfile -t unified_reports < <(find "$DEBUG_BASE_DIR" -name "unified_debug_report_*.json" -type f -printf '%T@ %p\n' | sort -rn | head -5 | cut -d' ' -f2-)
    
    if [[ ${#unified_reports[@]} -gt 0 ]]; then
        echo "  ${GREEN}Unified Reports:${NC}"
        for report in "${unified_reports[@]}"; do
            local timestamp size
            timestamp=$(stat -c %y "$report" | cut -d. -f1)
            size=$(du -h "$report" | cut -f1)
            echo "    $(basename "$report") - $timestamp ($size)"
        done
    fi
    
    # Find recent component reports
    local component_reports
    mapfile -t component_reports < <(find "$DEBUG_BASE_DIR" -name "*_report_*.json" -not -name "unified_*" -type f -printf '%T@ %p\n' | sort -rn | head -10 | cut -d' ' -f2-)
    
    if [[ ${#component_reports[@]} -gt 0 ]]; then
        echo "  ${GREEN}Component Reports:${NC}"
        for report in "${component_reports[@]}"; do
            local timestamp size
            timestamp=$(stat -c %y "$report" | cut -d. -f1)
            size=$(du -h "$report" | cut -f1)
            echo "    $(basename "$report") - $timestamp ($size)"
        done
    fi
    
    if [[ ${#unified_reports[@]} -eq 0 && ${#component_reports[@]} -eq 0 ]]; then
        warn "No debug reports found"
    fi
    
    echo
    echo "View a report? Enter filename or press Enter to continue:"
    read -r filename
    if [[ -n "$filename" && -f "$DEBUG_BASE_DIR/$filename" ]]; then
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$DEBUG_BASE_DIR/$filename" | less
        else
            less "$DEBUG_BASE_DIR/$filename"
        fi
    fi
}

# Show help information
show_help() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     Help Information                          ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}DSMIL Debug Infrastructure${NC}"
    echo "Comprehensive debugging tools for Dell Latitude 5450 MIL-SPEC DSMIL system"
    echo
    echo -e "${BLUE}System Requirements:${NC}"
    echo "  • Dell Latitude 5450 MIL-SPEC"
    echo "  • Debian Trixie with Linux kernel 6.14+"
    echo "  • dsmil-72dev kernel module loaded"
    echo "  • Python 3.9+ with numpy, psutil packages"
    echo
    echo -e "${BLUE}Token Information:${NC}"
    echo "  • Total tokens: 72 (0x0480-0x04C7)"
    echo "  • Organization: 6 groups × 12 devices"
    echo "  • Memory region: 0x52000000-0x68800000 (360MB)"
    echo
    echo -e "${BLUE}Debug Components:${NC}"
    echo "  • Infrastructure Debugger: Core debugging with event tracking"
    echo "  • Kernel Trace Analyzer: Kernel message analysis and pattern detection"
    echo "  • Memory Pattern Analyzer: Memory access pattern recognition"
    echo "  • Correlation Engine: Cross-component event correlation"
    echo "  • Unified Orchestrator: Coordinated multi-component debugging"
    echo
    echo -e "${BLUE}Output Locations:${NC}"
    echo "  • Base directory: $DEBUG_BASE_DIR"
    echo "  • Reports: unified_debug_report_*.json"
    echo "  • Component data: infrastructure/, kernel_trace/, etc."
    echo
    echo -e "${BLUE}Safety Features:${NC}"
    echo "  • Thermal monitoring with emergency stops"
    echo "  • JRTC1 training mode enforcement"
    echo "  • Graceful shutdown on system signals"
    echo "  • Read-only operations by default"
    echo
    echo -e "${BLUE}Command Line Usage:${NC}"
    echo "  $0                    - Interactive menu"
    echo "  $0 quick              - Quick 5-minute session"
    echo "  $0 interactive        - Interactive debugging"
    echo "  $0 test 0x0480:0x048F - Test token range"
    echo "  $0 status             - Show system status"
    echo
    echo "For detailed documentation, see README.md"
}

# Main interactive loop
interactive_mode() {
    while true; do
        show_menu
        echo -n "Select option: "
        read -r choice
        
        case $choice in
            1)
                run_unified_debug 300 ""
                ;;
            2)
                log "Starting interactive debug session..."
                cd "$SCRIPT_DIR"
                python3 unified_debug_orchestrator.py --interactive --output-dir "$DEBUG_BASE_DIR"
                ;;
            3)
                echo -n "Enter token range (e.g., 0x0480:0x048F): "
                read -r token_range
                if [[ -n "$token_range" ]]; then
                    echo -n "Enter duration in seconds [120]: "
                    read -r duration
                    duration=${duration:-120}
                    run_token_testing "$token_range" "$duration"
                else
                    warn "No token range specified"
                fi
                ;;
            4)
                run_unified_debug 1800 ""
                ;;
            5)
                run_component "infrastructure"
                ;;
            6)
                echo -n "Enter duration in seconds [300]: "
                read -r duration
                duration=${duration:-300}
                run_component "kernel" "$duration"
                ;;
            7)
                echo -n "Enter duration in seconds [300]: "
                read -r duration
                duration=${duration:-300}
                run_component "memory" "$duration"
                ;;
            8)
                echo -n "Enter duration in seconds [300]: "
                read -r duration
                duration=${duration:-300}
                run_component "correlation" "$duration"
                ;;
            9)
                check_prerequisites
                ;;
            10)
                clean_debug_output
                ;;
            11)
                view_recent_reports
                ;;
            s)
                show_system_status
                ;;
            h)
                show_help
                ;;
            q)
                log "Goodbye!"
                exit 0
                ;;
            *)
                warn "Invalid option: $choice"
                ;;
        esac
        
        echo
        echo "Press Enter to continue..."
        read -r
    done
}

# Command line interface
main() {
    # Create debug base directory
    mkdir -p "$DEBUG_BASE_DIR"
    
    # Handle command line arguments
    case ${1:-""} in
        "quick")
            check_prerequisites
            run_unified_debug 300 ""
            ;;
        "interactive")
            check_prerequisites
            cd "$SCRIPT_DIR"
            python3 unified_debug_orchestrator.py --interactive --output-dir "$DEBUG_BASE_DIR"
            ;;
        "test")
            if [[ -n ${2:-""} ]]; then
                check_prerequisites
                run_token_testing "$2" "${3:-120}"
            else
                error "Token range required for test mode"
                echo "Usage: $0 test 0x0480:0x048F [duration]"
                exit 1
            fi
            ;;
        "status")
            check_prerequisites
            show_system_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        "")
            # Interactive mode
            log "DSMIL Debug Infrastructure Launcher"
            check_prerequisites
            show_system_status
            interactive_mode
            ;;
        *)
            error "Unknown command: $1"
            echo "Usage: $0 [quick|interactive|test|status|help]"
            echo "Run without arguments for interactive mode"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
cleanup() {
    log "Cleaning up..."
    # Kill any background processes if needed
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"