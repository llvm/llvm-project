#!/bin/bash
#
# DSMIL Phase 1 Production Activation Script
# Safely activates monitoring for 29 NSA-identified devices
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
BACKEND_DIR="/home/john/LAT5150DRVMIL/web-interface/backend"
SCRIPTS_DIR="/home/john/LAT5150DRVMIL"
LOG_FILE="phase1_activation_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log_message "${BLUE}${BOLD}Checking prerequisites...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_message "${RED}ERROR: Python 3 not found${NC}"
        exit 1
    fi
    
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        log_message "${YELLOW}Sudo password required${NC}"
        sudo -v
    fi
    
    # Check required files
    local required_files=(
        "$SCRIPTS_DIR/test_phase1_safe_devices.py"
        "$SCRIPTS_DIR/phase1_monitoring_dashboard.py"
        "$BACKEND_DIR/expanded_safe_devices.py"
        "$BACKEND_DIR/config.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_message "${RED}ERROR: Required file missing: $file${NC}"
            exit 1
        fi
    done
    
    log_message "${GREEN}✅ All prerequisites met${NC}"
}

# Function to display phase 1 summary
display_summary() {
    log_message "\n${BLUE}${BOLD}========================================${NC}"
    log_message "${BLUE}${BOLD}   DSMIL PHASE 1 PRODUCTION ACTIVATION${NC}"
    log_message "${BLUE}${BOLD}========================================${NC}"
    log_message ""
    log_message "${BOLD}Deployment Summary:${NC}"
    log_message "  • Safe Devices: 29 (34.5% coverage)"
    log_message "  • Quarantined: 5 (permanent isolation)"
    log_message "  • Unknown: 50 (future phases)"
    log_message ""
    log_message "${BOLD}Device Categories:${NC}"
    log_message "  • Core Monitoring: 6 devices (100% confidence)"
    log_message "  • Security: 5 devices (65-90% confidence)"
    log_message "  • Network: 6 devices (65-90% confidence)"
    log_message "  • Training: 12 devices (50-60% confidence)"
    log_message ""
    log_message "${RED}${BOLD}⚠️  CRITICAL SAFETY REMINDER:${NC}"
    log_message "${RED}The following devices are QUARANTINED:${NC}"
    log_message "${RED}  • 0x8009: Emergency Wipe Controller${NC}"
    log_message "${RED}  • 0x800A: Secondary Wipe Trigger${NC}"
    log_message "${RED}  • 0x800B: Final Sanitization${NC}"
    log_message "${RED}  • 0x8019: Network Isolation/Wipe${NC}"
    log_message "${RED}  • 0x8029: Communications Blackout${NC}"
    log_message ""
}

# Function to test devices
test_devices() {
    log_message "\n${BLUE}${BOLD}Step 1: Testing Phase 1 Devices${NC}"
    log_message "${YELLOW}Running comprehensive device tests...${NC}"
    
    cd "$SCRIPTS_DIR"
    if python3 test_phase1_safe_devices.py 2>&1 | tee -a "$LOG_FILE"; then
        log_message "${GREEN}✅ Device testing completed successfully${NC}"
        return 0
    else
        log_message "${RED}❌ Device testing failed${NC}"
        return 1
    fi
}

# Function to update configuration
update_configuration() {
    log_message "\n${BLUE}${BOLD}Step 2: Updating Production Configuration${NC}"
    
    # Backup current configuration
    cp "$BACKEND_DIR/config.py" "$BACKEND_DIR/config.py.backup.$(date +%Y%m%d_%H%M%S)"
    log_message "${GREEN}✅ Configuration backed up${NC}"
    
    # Verify configuration updates
    python3 -c "
import sys
sys.path.insert(0, '$BACKEND_DIR')
from config import safe_monitoring_ids, quarantined_device_ids
print(f'Safe devices configured: {len(safe_monitoring_ids)}')
print(f'Quarantined devices: {len(quarantined_device_ids)}')
assert len(safe_monitoring_ids) == 29, 'Expected 29 safe devices'
assert len(quarantined_device_ids) == 5, 'Expected 5 quarantined devices'
" 2>&1 | tee -a "$LOG_FILE"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_message "${GREEN}✅ Configuration validated${NC}"
        return 0
    else
        log_message "${RED}❌ Configuration validation failed${NC}"
        return 1
    fi
}

# Function to start monitoring dashboard
start_monitoring() {
    log_message "\n${BLUE}${BOLD}Step 3: Starting Monitoring Dashboard${NC}"
    log_message "${YELLOW}Starting Phase 1 monitoring dashboard...${NC}"
    log_message ""
    log_message "${BOLD}Dashboard will monitor:${NC}"
    log_message "  • 29 safe devices in real-time"
    log_message "  • Thermal status and system health"
    log_message "  • Device response times and success rates"
    log_message ""
    log_message "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    log_message ""
    
    # Start dashboard
    cd "$SCRIPTS_DIR"
    python3 phase1_monitoring_dashboard.py
}

# Function to generate report
generate_report() {
    log_message "\n${BLUE}${BOLD}Generating Activation Report${NC}"
    
    local report_file="phase1_activation_report_$(date +%Y%m%d_%H%M%S).json"
    
    python3 -c "
import json
from datetime import datetime

report = {
    'activation_timestamp': datetime.now().isoformat(),
    'phase': 'Phase 1 - Safe Device Expansion',
    'status': 'ACTIVATED',
    'coverage': {
        'safe_devices': 29,
        'quarantined_devices': 5,
        'unknown_devices': 50,
        'total_devices': 84,
        'coverage_percentage': 34.5
    },
    'configuration': {
        'backend_updated': True,
        'monitoring_active': True,
        'safety_protocols': 'ENFORCED'
    },
    'next_steps': [
        'Monitor device performance for 30 days',
        'Collect operational metrics',
        'Plan Phase 2 expansion (Days 31-60)',
        'Maintain absolute quarantine on 5 devices'
    ]
}

with open('$report_file', 'w') as f:
    json.dump(report, f, indent=2)

print(f'Report saved to: $report_file')
" 2>&1 | tee -a "$LOG_FILE"
    
    log_message "${GREEN}✅ Activation report generated${NC}"
}

# Main execution
main() {
    log_message "${BOLD}Starting Phase 1 Production Activation${NC}"
    log_message "Timestamp: $(date)"
    log_message ""
    
    # Check prerequisites
    check_prerequisites
    
    # Display summary
    display_summary
    
    # Confirmation prompt
    echo ""
    read -p "$(echo -e ${YELLOW}Proceed with Phase 1 activation? [y/N]: ${NC})" -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_message "${YELLOW}Activation cancelled by user${NC}"
        exit 0
    fi
    
    # Execute activation steps
    if test_devices; then
        log_message "${GREEN}Device testing passed${NC}"
    else
        log_message "${RED}Device testing failed - aborting activation${NC}"
        exit 1
    fi
    
    if update_configuration; then
        log_message "${GREEN}Configuration updated${NC}"
    else
        log_message "${RED}Configuration update failed - aborting activation${NC}"
        exit 1
    fi
    
    # Generate report before starting monitoring
    generate_report
    
    # Start monitoring (this runs continuously)
    start_monitoring
    
    # Final message (shown after monitoring stops)
    log_message "\n${GREEN}${BOLD}========================================${NC}"
    log_message "${GREEN}${BOLD}   PHASE 1 ACTIVATION COMPLETE${NC}"
    log_message "${GREEN}${BOLD}========================================${NC}"
    log_message ""
    log_message "${BOLD}Summary:${NC}"
    log_message "  • 29 devices now under active monitoring"
    log_message "  • 5 devices remain quarantined"
    log_message "  • System operating at 34.5% coverage"
    log_message ""
    log_message "${BOLD}Next Steps:${NC}"
    log_message "  1. Monitor system for 30 days"
    log_message "  2. Analyze performance metrics"
    log_message "  3. Plan Phase 2 expansion"
    log_message ""
    log_message "Log file: $LOG_FILE"
}

# Run main function
main