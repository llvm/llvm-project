#!/bin/bash

# DSMIL IOCTL Handler Discovery Script
# Safely probes kernel module IOCTL handlers using assembly-level techniques
# Runs multiple probing methods and combines results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
DEVICE_PATH="/dev/dsmil-72dev"
RESULTS_DIR="ioctl_discovery_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/discovery_$TIMESTAMP.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}DSMIL IOCTL Handler Discovery System${NC}"
echo "======================================"
echo "Timestamp: $(date)"
echo "Results will be saved to: $RESULTS_FILE"
echo

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$RESULTS_FILE"
}

# Function to check if we have required privileges
check_privileges() {
    log "Checking system privileges..."
    
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Warning: Not running as root. Some features may be limited.${NC}"
        echo "For full functionality, run: sudo $0"
        return 1
    else
        echo -e "${GREEN}✓ Running with root privileges${NC}"
        return 0
    fi
}

# Function to check if DSMIL module is loaded
check_module() {
    log "Checking DSMIL kernel module status..."
    
    if lsmod | grep -q "dsmil"; then
        local module_info=$(lsmod | grep dsmil)
        echo -e "${GREEN}✓ DSMIL module is loaded${NC}"
        log "Module info: $module_info"
        return 0
    else
        echo -e "${RED}✗ DSMIL module not loaded${NC}"
        log "Module not found. Attempting to load..."
        
        # Try to load the module
        if [[ -f "01-source/kernel/dsmil-72dev.ko" ]]; then
            if insmod "01-source/kernel/dsmil-72dev.ko" 2>/dev/null; then
                echo -e "${GREEN}✓ Successfully loaded DSMIL module${NC}"
                log "Module loaded successfully"
                return 0
            else
                echo -e "${RED}✗ Failed to load DSMIL module${NC}"
                log "Module loading failed"
                return 1
            fi
        else
            echo -e "${RED}✗ DSMIL module file not found${NC}"
            log "Module file not found at 01-source/kernel/dsmil-72dev.ko"
            return 1
        fi
    fi
}

# Function to check device file
check_device() {
    log "Checking device file access..."
    
    if [[ -e "$DEVICE_PATH" ]]; then
        echo -e "${GREEN}✓ Device file $DEVICE_PATH exists${NC}"
        local perms=$(ls -l "$DEVICE_PATH")
        log "Device permissions: $perms"
        return 0
    else
        echo -e "${RED}✗ Device file $DEVICE_PATH not found${NC}"
        log "Device file not found"
        
        # Check for alternative device files
        echo "Checking for alternative device paths..."
        local alt_paths=("/dev/dsmil1" "/dev/dsmil2" "/dev/dsmil_enhanced" "/dev/mildev")
        
        for path in "${alt_paths[@]}"; do
            if [[ -e "$path" ]]; then
                echo -e "${YELLOW}Found alternative: $path${NC}"
                DEVICE_PATH="$path"
                log "Using alternative device path: $DEVICE_PATH"
                return 0
            fi
        done
        
        echo -e "${RED}No accessible device files found${NC}"
        return 1
    fi
}

# Function to build probe tools
build_tools() {
    log "Building IOCTL probe tools..."
    
    # Check if we have build tools
    if ! which gcc >/dev/null 2>&1; then
        echo -e "${RED}✗ GCC not found. Installing build tools...${NC}"
        apt-get update && apt-get install -y build-essential
    fi
    
    # Build the tools
    echo "Building basic probe tool..."
    if make -f Makefile.probe clean && make -f Makefile.probe; then
        echo -e "${GREEN}✓ Basic probe tool built successfully${NC}"
        log "Basic probe tool compilation successful"
    else
        echo -e "${RED}✗ Failed to build basic probe tool${NC}"
        log "Basic probe tool compilation failed"
        return 1
    fi
    
    echo "Building advanced probe tool..."
    if gcc -Wall -Wextra -O2 -std=gnu11 -D_GNU_SOURCE -o ioctl_advanced_probe ioctl_advanced_probe.c; then
        echo -e "${GREEN}✓ Advanced probe tool built successfully${NC}"
        log "Advanced probe tool compilation successful"
    else
        echo -e "${RED}✗ Failed to build advanced probe tool${NC}"
        log "Advanced probe tool compilation failed"
        return 1
    fi
    
    return 0
}

# Function to run basic probe
run_basic_probe() {
    log "Running basic IOCTL probe..."
    echo -e "${BLUE}=== BASIC IOCTL PROBE ===${NC}"
    
    if [[ -x "./ioctl_probe_safe" ]]; then
        echo "Executing basic probe on $DEVICE_PATH..."
        ./ioctl_probe_safe "$DEVICE_PATH" 2>&1 | tee -a "$RESULTS_FILE"
        echo
    else
        echo -e "${RED}✗ Basic probe tool not found or not executable${NC}"
        log "Basic probe tool execution failed"
        return 1
    fi
}

# Function to run advanced probe
run_advanced_probe() {
    log "Running advanced IOCTL probe..."
    echo -e "${BLUE}=== ADVANCED IOCTL PROBE ===${NC}"
    
    if [[ -x "./ioctl_advanced_probe" ]]; then
        echo "Executing advanced probe on $DEVICE_PATH..."
        ./ioctl_advanced_probe "$DEVICE_PATH" 2>&1 | tee -a "$RESULTS_FILE"
        echo
    else
        echo -e "${RED}✗ Advanced probe tool not found or not executable${NC}"
        log "Advanced probe tool execution failed"
        return 1
    fi
}

# Function to analyze kernel messages
analyze_kernel_messages() {
    log "Analyzing kernel messages..."
    echo -e "${BLUE}=== KERNEL MESSAGE ANALYSIS ===${NC}"
    
    echo "Recent DSMIL kernel messages:"
    dmesg | tail -50 | grep -i "dsmil\|ioctl\|smi" || echo "No relevant messages found"
    echo
    
    log "Kernel message analysis complete"
}

# Function to test specific commands
test_specific_commands() {
    log "Testing specific IOCTL commands..."
    echo -e "${BLUE}=== SPECIFIC COMMAND TESTS ===${NC}"
    
    # Known working command
    local test_cmd="0x80044D01"  # MILDEV_IOC_GET_VERSION
    echo "Testing known working command: $test_cmd"
    
    if [[ -x "./ioctl_probe_safe" ]]; then
        ./ioctl_probe_safe "$DEVICE_PATH" "$test_cmd" 2>&1 | tee -a "$RESULTS_FILE"
    fi
    
    echo
    log "Specific command testing complete"
}

# Function to generate summary report
generate_summary() {
    log "Generating discovery summary..."
    echo -e "${BLUE}=== DISCOVERY SUMMARY ===${NC}"
    
    local summary_file="$RESULTS_DIR/summary_$TIMESTAMP.txt"
    
    cat > "$summary_file" << EOF
DSMIL IOCTL Handler Discovery Summary
====================================
Timestamp: $(date)
Device: $DEVICE_PATH
Module Status: $(lsmod | grep dsmil || echo "Not loaded")

Results Location: $RESULTS_FILE

Key Findings:
EOF
    
    # Extract key findings from results
    if [[ -f "$RESULTS_FILE" ]]; then
        echo "Extracting key findings..."
        
        # Count handlers found
        local handlers_found=$(grep -c "HANDLER EXISTS\|FOUND!" "$RESULTS_FILE" || echo "0")
        echo "IOCTL Handlers Found: $handlers_found" >> "$summary_file"
        
        # Check for SMI activity
        if grep -q "SMI" "$RESULTS_FILE"; then
            echo "SMI Activity: Detected" >> "$summary_file"
        else
            echo "SMI Activity: None detected" >> "$summary_file"
        fi
        
        # Check for crashes
        local crashes=$(grep -c "CRASH" "$RESULTS_FILE" || echo "0")
        echo "Crashes During Probing: $crashes" >> "$summary_file"
        
        # Memory signatures found
        if grep -q "signature" "$RESULTS_FILE"; then
            echo "Memory Signatures: Found" >> "$summary_file"
            grep "signature" "$RESULTS_FILE" | head -5 >> "$summary_file"
        else
            echo "Memory Signatures: None found" >> "$summary_file"
        fi
    fi
    
    echo >> "$summary_file"
    echo "For detailed results, see: $RESULTS_FILE" >> "$summary_file"
    
    cat "$summary_file"
    log "Summary generated at: $summary_file"
}

# Main execution flow
main() {
    log "Starting DSMIL IOCTL discovery process"
    
    # System checks
    check_privileges
    has_root=$?
    
    check_module
    module_ok=$?
    
    check_device  
    device_ok=$?
    
    if [[ $module_ok -ne 0 || $device_ok -ne 0 ]]; then
        echo -e "${RED}Critical dependencies not met. Cannot proceed.${NC}"
        log "Dependency check failed"
        exit 1
    fi
    
    # Build tools
    if ! build_tools; then
        echo -e "${RED}Failed to build probe tools. Cannot proceed.${NC}"
        log "Tool building failed"
        exit 1
    fi
    
    # Run probes
    echo -e "${GREEN}Starting IOCTL handler discovery...${NC}"
    log "Beginning probe execution"
    
    run_basic_probe
    
    if [[ $has_root -eq 0 ]]; then
        run_advanced_probe
    else
        echo -e "${YELLOW}Skipping advanced probe (requires root)${NC}"
        log "Advanced probe skipped due to insufficient privileges"
    fi
    
    test_specific_commands
    analyze_kernel_messages
    
    # Generate final report
    generate_summary
    
    echo
    echo -e "${GREEN}IOCTL discovery complete!${NC}"
    echo -e "Results saved to: ${BLUE}$RESULTS_FILE${NC}"
    echo -e "Summary available at: ${BLUE}$RESULTS_DIR/summary_$TIMESTAMP.txt${NC}"
    
    log "IOCTL discovery process completed successfully"
}

# Handle script interruption
cleanup() {
    log "Discovery script interrupted"
    echo -e "${YELLOW}Discovery interrupted. Partial results may be available.${NC}"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"