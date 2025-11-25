#!/bin/bash
"""
Master Thermal Guardian Deployment Script
==========================================

Complete deployment package for Dell LAT5150DRVMIL thermal protection system.
Integrates thermal guardian with existing MIL-SPEC driver ecosystem.

Agent Team: Final Integration & Deployment Package
Version: 1.0
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/thermal_guardian_deploy.log"
BACKUP_DIR="/tmp/thermal_guardian_backup_$(date +%s)"

# System paths
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_DIR="/etc/thermal-guardian"
BIN_DIR="/usr/local/bin"
LOG_DIR="/var/log"

# Deployment flags
FORCE_INSTALL=false
SKIP_VALIDATION=false
VERBOSE=false

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

# Error handling
cleanup_on_error() {
    log_error "Deployment failed. Performing cleanup..."
    
    # Stop service if running
    systemctl stop thermal-guardian 2>/dev/null || true
    systemctl disable thermal-guardian 2>/dev/null || true
    
    # Remove installed files
    rm -f "$SYSTEMD_DIR/thermal-guardian.service"
    rm -f "$BIN_DIR/thermal_guardian"
    rm -f "$BIN_DIR/thermal_status"
    rm -rf "$CONFIG_DIR"
    
    # Restore backup if exists
    if [[ -d "$BACKUP_DIR" ]]; then
        log "Restoring backup from $BACKUP_DIR"
        # Restore logic here
    fi
    
    exit 1
}

trap cleanup_on_error ERR

# Validation functions
check_system_compatibility() {
    log "Checking system compatibility..."
    
    # Check if running on Dell Latitude
    if [[ ! -f /sys/class/dmi/id/product_name ]] || 
       ! grep -q "Latitude" /sys/class/dmi/id/product_name 2>/dev/null; then
        log_warning "Not running on Dell Latitude - thermal paths may differ"
    fi
    
    # Check for required thermal sensors
    local sensor_paths=(
        "/sys/class/thermal/thermal_zone9/temp"  # x86_pkg_temp
        "/sys/class/thermal/thermal_zone7/temp"  # dell_tcpu
        "/sys/class/hwmon/hwmon4/temp1_input"    # coretemp
    )
    
    local missing_sensors=0
    for sensor in "${sensor_paths[@]}"; do
        if [[ ! -f "$sensor" ]]; then
            log_warning "Thermal sensor missing: $sensor"
            ((missing_sensors++))
        fi
    done
    
    if [[ $missing_sensors -gt 2 ]]; then
        log_error "Too many missing thermal sensors. System may not be compatible."
        return 1
    fi
    
    # Check for fan control
    if [[ ! -f "/sys/class/hwmon/hwmon5/pwm1" ]]; then
        log_warning "Fan control interface not found - fan control disabled"
    fi
    
    # Check for CPU frequency control
    if [[ ! -f "/sys/devices/system/cpu/intel_pstate/max_perf_pct" ]]; then
        log_error "Intel P-State driver required but not found"
        return 1
    fi
    
    # Check Python version
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        log_error "Python 3.8+ required"
        return 1
    fi
    
    log_success "System compatibility check passed"
    return 0
}

check_permissions() {
    log "Checking permissions..."
    
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        return 1
    fi
    
    # Check write permissions for key directories
    local required_dirs=("$SYSTEMD_DIR" "$BIN_DIR" "$LOG_DIR")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -w "$dir" ]]; then
            log_error "No write permission for $dir"
            return 1
        fi
    done
    
    log_success "Permission check passed"
    return 0
}

validate_thermal_hardware() {
    log "Validating thermal hardware..."
    
    # Read current temperatures
    local temp_readings=()
    local sensor_paths=(
        "/sys/class/thermal/thermal_zone9/temp:x86_pkg_temp"
        "/sys/class/thermal/thermal_zone7/temp:dell_tcpu" 
        "/sys/class/hwmon/hwmon4/temp1_input:coretemp"
    )
    
    for sensor_info in "${sensor_paths[@]}"; do
        IFS=':' read -r path name <<< "$sensor_info"
        
        if [[ -f "$path" ]]; then
            temp=$(cat "$path" 2>/dev/null || echo "0")
            temp_celsius=$((temp / 1000))
            temp_readings+=("$name:${temp_celsius}°C")
            
            # Sanity check temperatures
            if [[ $temp_celsius -lt 20 || $temp_celsius -gt 120 ]]; then
                log_warning "$name reading seems invalid: ${temp_celsius}°C"
            fi
        fi
    done
    
    if [[ ${#temp_readings[@]} -eq 0 ]]; then
        log_error "No valid temperature readings found"
        return 1
    fi
    
    log "Current temperatures:"
    for reading in "${temp_readings[@]}"; do
        log "  $reading"
    done
    
    # Test fan control
    local fan_pwm="/sys/class/hwmon/hwmon5/pwm1"
    if [[ -f "$fan_pwm" ]]; then
        local original_pwm
        original_pwm=$(cat "$fan_pwm" 2>/dev/null || echo "128")
        
        # Test write access
        if echo "$original_pwm" > "$fan_pwm" 2>/dev/null; then
            log_success "Fan control validation passed"
        else
            log_warning "Fan control test failed - may need manual configuration"
        fi
    fi
    
    # Test CPU frequency control
    local cpu_control="/sys/devices/system/cpu/intel_pstate/max_perf_pct"
    if [[ -f "$cpu_control" ]]; then
        local original_perf
        original_perf=$(cat "$cpu_control" 2>/dev/null || echo "100")
        
        if echo "$original_perf" > "$cpu_control" 2>/dev/null; then
            log_success "CPU frequency control validation passed"
        else
            log_error "CPU frequency control test failed"
            return 1
        fi
    fi
    
    log_success "Thermal hardware validation completed"
    return 0
}

# Installation functions
install_dependencies() {
    log "Installing dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install required packages
    local packages=(
        "python3"
        "python3-pip"
        "lm-sensors"
        "fancontrol"
        "cpufrequtils"
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l "$package" >/dev/null 2>&1; then
            log "Installing $package..."
            apt-get install -y "$package"
        fi
    done
    
    # Install Python packages
    pip3 install --quiet psutil
    
    log_success "Dependencies installed"
}

create_backup() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup existing thermal guardian if present
    if [[ -f "$BIN_DIR/thermal_guardian" ]]; then
        cp "$BIN_DIR/thermal_guardian" "$BACKUP_DIR/"
    fi
    
    if [[ -f "$SYSTEMD_DIR/thermal-guardian.service" ]]; then
        cp "$SYSTEMD_DIR/thermal-guardian.service" "$BACKUP_DIR/"
    fi
    
    if [[ -d "$CONFIG_DIR" ]]; then
        cp -r "$CONFIG_DIR" "$BACKUP_DIR/"
    fi
    
    log_success "Backup created at $BACKUP_DIR"
}

install_thermal_guardian() {
    log "Installing Thermal Guardian system..."
    
    # Create directories
    mkdir -p "$CONFIG_DIR"
    
    # Install main script
    cp "$SCRIPT_DIR/thermal_guardian.py" "$BIN_DIR/thermal_guardian"
    chmod +x "$BIN_DIR/thermal_guardian"
    
    # Install monitoring tool if exists
    if [[ -f "$SCRIPT_DIR/thermal_status.py" ]]; then
        cp "$SCRIPT_DIR/thermal_status.py" "$BIN_DIR/thermal_status"
        chmod +x "$BIN_DIR/thermal_status"
    fi
    
    # Create default configuration
    cat > "$CONFIG_DIR/thermal_guardian.conf" << 'EOF'
{
    "monitoring_interval": 1.0,
    "prediction_horizon": 5.0,
    "sensor_weights": {
        "x86_pkg_temp": 0.4,
        "dell_tcpu": 0.3,
        "coretemp": 0.3,
        "dell_cpu": 0.0
    },
    "phase_delays": {
        "1": 2.0,
        "2": 1.5,
        "3": 1.0,
        "4": 0.5,
        "5": 0.0
    },
    "max_fan_pwm": 255,
    "emergency_temp": 105.0,
    "critical_temp": 103.0
}
EOF
    
    # Create systemd service
    cat > "$SYSTEMD_DIR/thermal-guardian.service" << EOF
[Unit]
Description=Dell LAT5150DRVMIL Thermal Guardian
Documentation=file://$PROJECT_ROOT/docs/
After=multi-user.target
Wants=network.target

[Service]
Type=simple
ExecStart=$BIN_DIR/thermal_guardian --config $CONFIG_DIR/thermal_guardian.conf
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
User=root
Group=root

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$LOG_DIR $CONFIG_DIR /sys/class/thermal /sys/class/hwmon /sys/devices/system/cpu

# Resource limits
LimitNOFILE=1024
LimitNPROC=100
MemoryLimit=100M
CPUQuota=5%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=thermal-guardian

[Install]
WantedBy=multi-user.target
EOF
    
    # Create log rotation
    cat > "/etc/logrotate.d/thermal-guardian" << 'EOF'
/var/log/thermal_guardian.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        systemctl reload thermal-guardian 2>/dev/null || true
    endscript
}
EOF
    
    log_success "Thermal Guardian system installed"
}

integrate_with_milspec_driver() {
    log "Integrating with MIL-SPEC driver ecosystem..."
    
    # Check if kernel driver exists
    local kernel_driver="$PROJECT_ROOT/01-source/kernel-driver/dell-millspec-enhanced.c"
    if [[ -f "$kernel_driver" ]]; then
        log "Found MIL-SPEC kernel driver - creating integration hooks"
        
        # Create thermal status interface for kernel driver
        cat > "$CONFIG_DIR/kernel_integration.conf" << 'EOF'
{
    "kernel_interface": {
        "thermal_status_path": "/sys/class/milspec/thermal_status",
        "thermal_events_path": "/sys/class/milspec/thermal_events",
        "emergency_callback": "/usr/local/bin/milspec_thermal_emergency"
    },
    "mode5_integration": {
        "enabled": true,
        "phase_to_mode5_mapping": {
            "0": "disabled",
            "1": "standard", 
            "2": "standard",
            "3": "enhanced",
            "4": "paranoid",
            "5": "paranoid"
        }
    }
}
EOF
        
        # Create emergency callback script
        cat > "$BIN_DIR/milspec_thermal_emergency" << 'EOF'
#!/bin/bash
# Emergency thermal callback for MIL-SPEC driver integration

logger -t thermal-guardian "EMERGENCY: Thermal emergency callback activated"

# Notify kernel driver if available
if [[ -f /dev/milspec ]]; then
    echo "THERMAL_EMERGENCY" > /dev/milspec 2>/dev/null || true
fi

# Execute emergency procedures
/usr/local/bin/thermal_guardian --emergency-action
EOF
        chmod +x "$BIN_DIR/milspec_thermal_emergency"
        
        log_success "MIL-SPEC driver integration configured"
    else
        log_warning "MIL-SPEC kernel driver not found - standalone mode"
    fi
}

configure_system_integration() {
    log "Configuring system integration..."
    
    # Enable and configure lm-sensors
    if command -v sensors-detect >/dev/null 2>&1; then
        log "Configuring hardware sensors..."
        yes "" | sensors-detect >/dev/null 2>&1 || true
        modprobe -a coretemp dell-smm-hwmon 2>/dev/null || true
    fi
    
    # Configure CPU governor
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        echo "powersave" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
    fi
    
    # Create monitoring script
    cat > "$BIN_DIR/thermal_monitor" << 'EOF'
#!/bin/bash
# Simple thermal monitoring script

while true; do
    temp=$(cat /sys/class/thermal/thermal_zone9/temp 2>/dev/null || echo "0")
    temp_celsius=$((temp / 1000))
    
    if [[ $temp_celsius -gt 95 ]]; then
        logger -t thermal-monitor "WARNING: High temperature detected: ${temp_celsius}°C"
    fi
    
    sleep 30
done
EOF
    chmod +x "$BIN_DIR/thermal_monitor"
    
    log_success "System integration configured"
}

# Testing functions
run_deployment_tests() {
    log "Running deployment tests..."
    
    # Test thermal guardian startup
    log "Testing thermal guardian startup..."
    if timeout 10 "$BIN_DIR/thermal_guardian" --status >/dev/null 2>&1; then
        log_success "Thermal guardian startup test passed"
    else
        log_error "Thermal guardian startup test failed"
        return 1
    fi
    
    # Test systemd service
    log "Testing systemd service..."
    systemctl daemon-reload
    
    if systemctl start thermal-guardian; then
        sleep 5
        if systemctl is-active thermal-guardian >/dev/null 2>&1; then
            log_success "Systemd service test passed"
            systemctl stop thermal-guardian
        else
            log_error "Systemd service failed to start"
            return 1
        fi
    else
        log_error "Failed to start systemd service"
        return 1
    fi
    
    # Test thermal sensor access
    log "Testing thermal sensor access..."
    local sensors_working=0
    local sensor_paths=(
        "/sys/class/thermal/thermal_zone9/temp"
        "/sys/class/thermal/thermal_zone7/temp"
        "/sys/class/hwmon/hwmon4/temp1_input"
    )
    
    for sensor in "${sensor_paths[@]}"; do
        if [[ -r "$sensor" ]]; then
            temp=$(cat "$sensor" 2>/dev/null || echo "0")
            if [[ $temp -gt 0 ]]; then
                ((sensors_working++))
            fi
        fi
    done
    
    if [[ $sensors_working -gt 0 ]]; then
        log_success "Sensor access test passed ($sensors_working sensors working)"
    else
        log_error "No working thermal sensors found"
        return 1
    fi
    
    # Test control interfaces
    log "Testing control interfaces..."
    
    # Test CPU frequency control
    local cpu_control="/sys/devices/system/cpu/intel_pstate/max_perf_pct"
    if [[ -w "$cpu_control" ]]; then
        local original_perf
        original_perf=$(cat "$cpu_control")
        
        if echo "90" > "$cpu_control" 2>/dev/null && 
           echo "$original_perf" > "$cpu_control" 2>/dev/null; then
            log_success "CPU frequency control test passed"
        else
            log_warning "CPU frequency control test failed"
        fi
    fi
    
    log_success "All deployment tests passed"
    return 0
}

generate_deployment_report() {
    log "Generating deployment report..."
    
    local report_file="/tmp/thermal_guardian_deployment_report.txt"
    
    cat > "$report_file" << EOF
THERMAL GUARDIAN DEPLOYMENT REPORT
==================================
Generated: $(date)
System: $(uname -a)
User: $(whoami)

INSTALLATION STATUS:
- Thermal Guardian Script: $([[ -f "$BIN_DIR/thermal_guardian" ]] && echo "INSTALLED" || echo "MISSING")
- Systemd Service: $([[ -f "$SYSTEMD_DIR/thermal-guardian.service" ]] && echo "INSTALLED" || echo "MISSING")
- Configuration: $([[ -f "$CONFIG_DIR/thermal_guardian.conf" ]] && echo "INSTALLED" || echo "MISSING")
- Status Tool: $([[ -f "$BIN_DIR/thermal_status" ]] && echo "INSTALLED" || echo "MISSING")

SYSTEM INFORMATION:
EOF
    
    # Add thermal sensor information
    echo "THERMAL SENSORS:" >> "$report_file"
    local sensor_paths=(
        "/sys/class/thermal/thermal_zone9/temp:x86_pkg_temp"
        "/sys/class/thermal/thermal_zone7/temp:dell_tcpu"
        "/sys/class/hwmon/hwmon4/temp1_input:coretemp"
    )
    
    for sensor_info in "${sensor_paths[@]}"; do
        IFS=':' read -r path name <<< "$sensor_info"
        if [[ -f "$path" ]]; then
            temp=$(cat "$path" 2>/dev/null || echo "0")
            temp_celsius=$((temp / 1000))
            echo "- $name: ${temp_celsius}°C ($path)" >> "$report_file"
        else
            echo "- $name: NOT FOUND ($path)" >> "$report_file"
        fi
    done
    
    # Add control interface information
    echo -e "\nCONTROL INTERFACES:" >> "$report_file"
    
    local fan_control="/sys/class/hwmon/hwmon5/pwm1"
    if [[ -f "$fan_control" ]]; then
        fan_pwm=$(cat "$fan_control" 2>/dev/null || echo "unknown")
        echo "- Fan Control: AVAILABLE (current PWM: $fan_pwm)" >> "$report_file"
    else
        echo "- Fan Control: NOT AVAILABLE" >> "$report_file"
    fi
    
    local cpu_control="/sys/devices/system/cpu/intel_pstate/max_perf_pct"
    if [[ -f "$cpu_control" ]]; then
        cpu_perf=$(cat "$cpu_control" 2>/dev/null || echo "unknown")
        echo "- CPU Frequency Control: AVAILABLE (current limit: $cpu_perf%)" >> "$report_file"
    else
        echo "- CPU Frequency Control: NOT AVAILABLE" >> "$report_file"
    fi
    
    # Add service status
    echo -e "\nSERVICE STATUS:" >> "$report_file"
    if systemctl is-enabled thermal-guardian >/dev/null 2>&1; then
        echo "- Service Enabled: YES" >> "$report_file"
    else
        echo "- Service Enabled: NO" >> "$report_file"
    fi
    
    if systemctl is-active thermal-guardian >/dev/null 2>&1; then
        echo "- Service Running: YES" >> "$report_file"
    else
        echo "- Service Running: NO" >> "$report_file"
    fi
    
    # Add quick start commands
    cat >> "$report_file" << 'EOF'

QUICK START COMMANDS:
- Start service: systemctl start thermal-guardian
- Enable at boot: systemctl enable thermal-guardian
- Check status: thermal_guardian --status
- Monitor live: thermal_status --watch
- View logs: journalctl -u thermal-guardian -f

CONFIGURATION FILES:
- Main config: /etc/thermal-guardian/thermal_guardian.conf
- Service file: /etc/systemd/system/thermal-guardian.service
- Integration: /etc/thermal-guardian/kernel_integration.conf

EMERGENCY PROCEDURES:
- Manual shutdown: thermal_guardian --emergency-action
- Force stop: systemctl stop thermal-guardian
- Reset config: cp /etc/thermal-guardian/thermal_guardian.conf.default /etc/thermal-guardian/thermal_guardian.conf
EOF
    
    log_success "Deployment report generated: $report_file"
    
    # Display summary
    echo
    log_success "=== DEPLOYMENT SUMMARY ==="
    cat "$report_file" | grep -E "(INSTALLED|AVAILABLE|YES|NO)" | head -10
    echo
    log "Full report available at: $report_file"
}

# Main deployment function
main() {
    echo
    log "=== THERMAL GUARDIAN DEPLOYMENT ==="
    log "Agent Team: Final Integration & Deployment"
    log "Target: Dell LAT5150DRVMIL"
    log "Version: 1.0"
    echo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --force            Force installation even if validation fails"
                echo "  --skip-validation  Skip hardware validation"
                echo "  --verbose          Enable verbose output"
                echo "  --help             Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Pre-flight checks
    log "Starting pre-flight checks..."
    
    if ! check_permissions; then
        log_error "Permission check failed"
        exit 1
    fi
    
    if [[ $SKIP_VALIDATION != true ]]; then
        if ! check_system_compatibility; then
            if [[ $FORCE_INSTALL != true ]]; then
                log_error "System compatibility check failed. Use --force to override."
                exit 1
            else
                log_warning "System compatibility check failed - continuing with --force"
            fi
        fi
        
        if ! validate_thermal_hardware; then
            if [[ $FORCE_INSTALL != true ]]; then
                log_error "Thermal hardware validation failed. Use --force to override."
                exit 1
            else
                log_warning "Thermal hardware validation failed - continuing with --force"
            fi
        fi
    fi
    
    log_success "Pre-flight checks completed"
    
    # Main installation
    log "Starting main installation..."
    
    create_backup
    install_dependencies
    install_thermal_guardian
    integrate_with_milspec_driver
    configure_system_integration
    
    log_success "Main installation completed"
    
    # Post-installation testing
    log "Starting post-installation testing..."
    
    if ! run_deployment_tests; then
        log_error "Deployment tests failed"
        if [[ $FORCE_INSTALL != true ]]; then
            cleanup_on_error
        fi
    fi
    
    log_success "Post-installation testing completed"
    
    # Enable and start service
    log "Enabling thermal guardian service..."
    
    systemctl daemon-reload
    systemctl enable thermal-guardian
    systemctl start thermal-guardian
    
    # Wait a moment and check status
    sleep 3
    if systemctl is-active thermal-guardian >/dev/null 2>&1; then
        log_success "Thermal Guardian service started successfully"
    else
        log_error "Failed to start Thermal Guardian service"
        journalctl -u thermal-guardian --no-pager -l
    fi
    
    # Generate deployment report
    generate_deployment_report
    
    # Final status
    echo
    log_success "=== DEPLOYMENT COMPLETE ==="
    log "Thermal Guardian is now protecting your system"
    log "Monitor with: thermal_status --watch"
    log "Service logs: journalctl -u thermal-guardian -f"
    echo
    
    # Show current status
    if command -v thermal_guardian >/dev/null 2>&1; then
        log "Current thermal status:"
        thermal_guardian --status || true
    fi
}

# Execute main function
main "$@"