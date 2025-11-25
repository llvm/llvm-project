#!/bin/bash
# Quick Thermal Guardian Test Script
# Agent 3 Implementation - Dell LAT5150DRVMIL

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================="
echo "THERMAL GUARDIAN - QUICK SYSTEM TEST"
echo "========================================="
echo

# Test 1: Check if we're root
log_info "Test 1: Checking root permissions..."
if [[ $EUID -eq 0 ]]; then
    log_success "Running as root ✓"
else
    log_error "Must run as root (use sudo)"
    exit 1
fi

# Test 2: Check thermal zones
log_info "Test 2: Checking thermal zones..."
thermal_count=$(find /sys/class/thermal -name "thermal_zone*" -type d 2>/dev/null | wc -l)
if [[ $thermal_count -gt 0 ]]; then
    log_success "Found $thermal_count thermal zones ✓"
    
    # Show thermal zones
    for zone in /sys/class/thermal/thermal_zone*; do
        if [[ -r "$zone/type" && -r "$zone/temp" ]]; then
            zone_type=$(cat "$zone/type" 2>/dev/null || echo "unknown")
            zone_temp=$(cat "$zone/temp" 2>/dev/null || echo "0")
            zone_temp_c=$((zone_temp / 1000))
            echo "    $(basename "$zone"): $zone_type (${zone_temp_c}°C)"
        fi
    done
else
    log_error "No thermal zones found"
    exit 1
fi
echo

# Test 3: Check hardware monitors
log_info "Test 3: Checking hardware monitors..."
hwmon_count=$(find /sys/class/hwmon -name "hwmon*" -type d 2>/dev/null | wc -l)
if [[ $hwmon_count -gt 0 ]]; then
    log_success "Found $hwmon_count hardware monitors ✓"
    
    # Check for Dell SMM
    dell_smm_found=false
    for hwmon in /sys/class/hwmon/hwmon*; do
        if [[ -r "$hwmon/name" ]]; then
            hwmon_name=$(cat "$hwmon/name" 2>/dev/null || echo "unknown")
            echo "    $(basename "$hwmon"): $hwmon_name"
            
            if [[ "$hwmon_name" == "dell_smm" || "$hwmon_name" == "i8k" ]]; then
                dell_smm_found=true
                log_success "Dell SMM found - fan control available ✓"
                
                # Check for PWM controls
                pwm_files=$(find "$hwmon" -name "pwm*" -type f 2>/dev/null | wc -l)
                if [[ $pwm_files -gt 0 ]]; then
                    log_success "Found $pwm_files PWM fan controls ✓"
                fi
            fi
        fi
    done
    
    if [[ $dell_smm_found == false ]]; then
        log_warning "Dell SMM not found - limited fan control"
    fi
else
    log_warning "No hardware monitors found"
fi
echo

# Test 4: Check Intel P-State
log_info "Test 4: Checking Intel P-State..."
if [[ -d /sys/devices/system/cpu/intel_pstate ]]; then
    log_success "Intel P-State available ✓"
    
    # Check P-State files
    for file in max_perf_pct min_perf_pct no_turbo; do
        pstate_file="/sys/devices/system/cpu/intel_pstate/$file"
        if [[ -w "$pstate_file" ]]; then
            current_val=$(cat "$pstate_file" 2>/dev/null || echo "unknown")
            echo "    $file: $current_val"
        else
            log_warning "$file not writable"
        fi
    done
else
    log_warning "Intel P-State not available - limited CPU control"
fi
echo

# Test 5: Test Python script
log_info "Test 5: Testing thermal guardian script..."
if [[ -f "./thermal_guardian.py" ]]; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 6) else 1)" 2>/dev/null; then
        log_success "Python 3.6+ available ✓"
        
        # Test script import
        if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from thermal_guardian import ThermalManager, ThermalConfig
    print('Script imports successfully')
except Exception as e:
    print(f'Import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
            log_success "Script imports successfully ✓"
            
            # Run test mode
            log_info "Running thermal guardian test..."
            if python3 ./thermal_guardian.py --test 2>/dev/null; then
                log_success "Thermal guardian test passed ✓"
            else
                log_warning "Thermal guardian test had issues"
            fi
        else
            log_error "Script import failed"
        fi
    else
        log_error "Python 3.6+ required"
    fi
else
    log_error "thermal_guardian.py not found in current directory"
fi
echo

# Test 6: Check current temperatures
log_info "Test 6: Current system temperatures..."
max_temp=0
temp_count=0

# Check thermal zones
for zone in /sys/class/thermal/thermal_zone*; do
    if [[ -r "$zone/temp" ]]; then
        temp_raw=$(cat "$zone/temp" 2>/dev/null || echo "0")
        temp_c=$((temp_raw / 1000))
        
        if [[ $temp_c -gt 0 && $temp_c -lt 150 ]]; then  # Reasonable range
            temp_count=$((temp_count + 1))
            if [[ $temp_c -gt $max_temp ]]; then
                max_temp=$temp_c
            fi
            
            # Color code temperature
            if [[ $temp_c -ge 90 ]]; then
                echo -e "    $(basename "$zone"): \033[91m${temp_c}°C\033[0m (HIGH)"
            elif [[ $temp_c -ge 80 ]]; then
                echo -e "    $(basename "$zone"): \033[93m${temp_c}°C\033[0m (WARM)"
            else
                echo -e "    $(basename "$zone"): \033[92m${temp_c}°C\033[0m (NORMAL)"
            fi
        fi
    fi
done

if [[ $temp_count -gt 0 ]]; then
    log_success "Read $temp_count temperature sensors ✓"
    echo "    Maximum temperature: ${max_temp}°C"
    
    if [[ $max_temp -ge 95 ]]; then
        log_warning "System is running hot! Consider immediate thermal management."
    elif [[ $max_temp -ge 85 ]]; then
        log_warning "System is warm. Thermal Guardian recommended."
    else
        log_success "System temperatures are normal ✓"
    fi
else
    log_error "No temperature readings available"
fi
echo

# Test 7: Service test (if installed)
log_info "Test 7: Checking service installation..."
if [[ -f "/etc/systemd/system/thermal-guardian.service" ]]; then
    log_success "Service file installed ✓"
    
    if systemctl is-enabled thermal-guardian >/dev/null 2>&1; then
        log_success "Service is enabled ✓"
    else
        log_warning "Service is not enabled"
    fi
    
    if systemctl is-active thermal-guardian >/dev/null 2>&1; then
        log_success "Service is running ✓"
    else
        log_info "Service is not currently running"
    fi
else
    log_info "Service not installed (run install_thermal_guardian.sh)"
fi
echo

# Summary
echo "========================================="
echo "TEST SUMMARY"
echo "========================================="

if [[ $thermal_count -gt 0 && $temp_count -gt 0 ]]; then
    log_success "✅ System is compatible with Thermal Guardian"
    echo
    echo "Next steps:"
    echo "1. Run: sudo ./install_thermal_guardian.sh"
    echo "2. Start: sudo systemctl start thermal-guardian"
    echo "3. Monitor: ./thermal_status.py --watch"
    
    if [[ $max_temp -ge 85 ]]; then
        echo
        log_warning "⚠ Immediate setup recommended due to high temperatures!"
    fi
else
    log_error "❌ System may not be fully compatible"
    echo
    echo "Issues found:"
    if [[ $thermal_count -eq 0 ]]; then
        echo "- No thermal zones detected"
    fi
    if [[ $temp_count -eq 0 ]]; then
        echo "- No temperature readings available"
    fi
fi

echo