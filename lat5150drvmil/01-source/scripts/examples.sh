#!/bin/bash
#
# Example usage scripts for Dell MIL-SPEC utilities
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Dell MIL-SPEC Utilities Examples${NC}"
echo "================================"
echo

# Function to pause between examples
pause() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

# Example 1: Basic Status Check
example_status() {
    echo -e "${GREEN}Example 1: Basic Status Check${NC}"
    echo "Check the current MIL-SPEC configuration:"
    echo
    echo "  $ milspec-control status"
    echo
    echo "This shows Mode 5 level, DSMIL status, and security state."
    pause
}

# Example 2: Progressive Security Enhancement
example_security() {
    echo -e "${GREEN}Example 2: Progressive Security Enhancement${NC}"
    echo "Gradually increase security level:"
    echo
    echo "  # Start with basic protection"
    echo "  $ sudo milspec-control mode5 1"
    echo
    echo "  # Enhance to lock VMs"
    echo "  $ sudo milspec-control mode5 2"
    echo
    echo "  # Enable intrusion response (requires planning!)"
    echo "  $ sudo milspec-control mode5 3"
    echo
    echo "WARNING: Level 4 is irreversible!"
    pause
}

# Example 3: DSMIL Activation
example_dsmil() {
    echo -e "${GREEN}Example 3: DSMIL Subsystem Activation${NC}"
    echo "Enable military subsystems:"
    echo
    echo "  # Basic military features"
    echo "  $ sudo milspec-control dsmil 1"
    echo
    echo "  # Full tactical capabilities"
    echo "  $ sudo milspec-control dsmil 2"
    echo
    echo "  # Check which devices are active"
    echo "  $ sudo milspec-control status | grep 'Active Devices'"
    pause
}

# Example 4: Event Monitoring
example_monitoring() {
    echo -e "${GREEN}Example 4: Event Monitoring${NC}"
    echo "Monitor security events in real-time:"
    echo
    echo "  # Basic monitoring with color"
    echo "  $ sudo milspec-monitor -c"
    echo
    echo "  # Detailed monitoring with timestamps and logging"
    echo "  $ sudo milspec-monitor -tcl /var/log/milspec-events.log"
    echo
    echo "  # Monitor only critical security events"
    echo "  $ sudo milspec-monitor -tc | grep -E 'SECURITY|INTRUSION|ERROR'"
    pause
}

# Example 5: Automated Security Response
example_automation() {
    echo -e "${GREEN}Example 5: Automated Security Response${NC}"
    echo "Script to monitor and respond to intrusions:"
    echo
    cat << 'SCRIPT'
#!/bin/bash
# intrusion-response.sh

milspec-monitor -m chardev | while read line; do
    if echo "$line" | grep -q "INTRUSION"; then
        echo "INTRUSION DETECTED! Taking action..."
        
        # Send alert
        logger -p security.crit "MIL-SPEC: Physical intrusion detected"
        
        # Increase security level if not already paranoid
        current=$(milspec-control status | grep "Level:" | awk '{print $3}')
        if [ "$current" -lt 3 ]; then
            milspec-control mode5 3
            echo "Security level increased to PARANOID"
        fi
        
        # Take snapshot of system state
        milspec-control status > /var/log/intrusion-$(date +%s).log
    fi
done
SCRIPT
    pause
}

# Example 6: Pre-deployment Checklist
example_deployment() {
    echo -e "${GREEN}Example 6: Pre-deployment Security Checklist${NC}"
    echo "Script to verify system before deployment:"
    echo
    cat << 'SCRIPT'
#!/bin/bash
# deployment-check.sh

echo "Dell MIL-SPEC Deployment Checklist"
echo "=================================="

# Check driver loaded
if ! lsmod | grep -q dell_milspec; then
    echo "[FAIL] Driver not loaded"
    exit 1
fi
echo "[OK] Driver loaded"

# Verify minimum security level
level=$(milspec-control status | grep -A1 "Mode 5" | grep Level | awk '{print $3}')
if [ "$level" -lt 2 ]; then
    echo "[WARN] Mode 5 level $level is below recommended (2+)"
else
    echo "[OK] Mode 5 level $level"
fi

# Check DSMIL activation
if milspec-control status | grep -q "Active Devices: None"; then
    echo "[WARN] No DSMIL devices active"
else
    echo "[OK] DSMIL devices active"
fi

# Verify TPM measurement
milspec-control measure
echo "[OK] TPM measurement completed"

# Check for intrusions
if milspec-control status | grep -q "Intrusion:.*DETECTED"; then
    echo "[FAIL] Intrusion detected - resolve before deployment"
    exit 1
fi
echo "[OK] No intrusions detected"

echo
echo "System ready for deployment"
SCRIPT
    pause
}

# Example 7: Monitoring as a Service
example_service() {
    echo -e "${GREEN}Example 7: Run Monitor as System Service${NC}"
    echo "Set up continuous monitoring:"
    echo
    echo "  # Install the service file"
    echo "  $ sudo cp milspec-monitor.service /etc/systemd/system/"
    echo
    echo "  # Enable and start the service"
    echo "  $ sudo systemctl enable milspec-monitor.service"
    echo "  $ sudo systemctl start milspec-monitor.service"
    echo
    echo "  # Check service status"
    echo "  $ sudo systemctl status milspec-monitor.service"
    echo
    echo "  # View logs"
    echo "  $ sudo journalctl -u milspec-monitor.service -f"
    pause
}

# Example 8: Integration with Monitoring Systems
example_integration() {
    echo -e "${GREEN}Example 8: Integration with Monitoring Systems${NC}"
    echo "Export events to monitoring systems:"
    echo
    cat << 'SCRIPT'
#!/bin/bash
# export-to-syslog.sh

# Forward MIL-SPEC events to syslog
milspec-monitor -t | while IFS= read -r line; do
    # Parse event type
    if echo "$line" | grep -q "ERROR"; then
        logger -p security.err "milspec: $line"
    elif echo "$line" | grep -q "SECURITY\|INTRUSION"; then
        logger -p security.warning "milspec: $line"
    else
        logger -p security.info "milspec: $line"
    fi
done

# For SIEM integration, export as JSON:
milspec-monitor | while IFS='|' read -r timestamp type data message; do
    cat << JSON | curl -X POST http://siem.local/api/events -H "Content-Type: application/json" -d @-
{
    "source": "dell-milspec",
    "timestamp": "$(date -Iseconds)",
    "type": "$(echo $type | xargs)",
    "data": "$(echo $data | xargs)",
    "message": "$(echo $message | xargs)"
}
JSON
done
SCRIPT
    pause
}

# Main menu
main() {
    while true; do
        echo -e "\n${BLUE}Select an example:${NC}"
        echo "1. Basic Status Check"
        echo "2. Progressive Security Enhancement"
        echo "3. DSMIL Subsystem Activation"
        echo "4. Event Monitoring"
        echo "5. Automated Security Response"
        echo "6. Pre-deployment Checklist"
        echo "7. Run Monitor as System Service"
        echo "8. Integration with Monitoring Systems"
        echo "9. Exit"
        echo
        read -p "Enter choice (1-9): " choice
        
        case $choice in
            1) example_status ;;
            2) example_security ;;
            3) example_dsmil ;;
            4) example_monitoring ;;
            5) example_automation ;;
            6) example_deployment ;;
            7) example_service ;;
            8) example_integration ;;
            9) echo "Exiting..."; exit 0 ;;
            *) echo "Invalid choice" ;;
        esac
    done
}

# Run main menu
main
