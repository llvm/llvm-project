#!/bin/bash
# Quick Thermal System Test for Dell LAT5150DRVMIL
# Rapid compatibility test for thermal guardian deployment
# Tests thermal sensors, fan control, and CPU frequency scaling

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
    ((TESTS_FAILED++))
}

run_test() {
    local test_name="$1"
    ((TESTS_TOTAL++))
    log_info "Test $TESTS_TOTAL: $test_name..."
}

# Test functions
test_root_permissions() {
    run_test "Checking root permissions"
    
    if [[ $EUID -eq 0 ]]; then
        log_success "Running as root ✓"
        return 0
    else
        log_error "Must run as root for hardware access"
        return 1
    fi
}

test_thermal_zones() {
    run_test "Checking thermal zones"
    
    local zones_found=0
    local important_zones=()
    
    # Check for specific thermal zones
    local target_zones=(
        "thermal_zone9:x86_pkg_temp"
        "thermal_zone7:TCPU" 
        "thermal_zone0:AMBF"
        "thermal_zone4:CPUV"
    )
    
    for zone_info in "${target_zones[@]}"; do
        IFS=':' read -r zone_name expected_type <<< "$zone_info"
        zone_path="/sys/class/thermal/$zone_name"
        
        if [[ -f "$zone_path/type" && -f "$zone_path/temp" ]]; then
            type=$(cat "$zone_path/type" 2>/dev/null || echo "unknown")
            temp=$(cat "$zone_path/temp" 2>/dev/null || echo "0")
            temp_celsius=$((temp / 1000))
            
            important_zones+=("$zone_name: $type = ${temp_celsius}°C")
            ((zones_found++))
            
            # Warn about high temperatures
            if [[ $temp_celsius -gt 85 ]]; then
                log_warning "$zone_name temperature high: ${temp_celsius}°C"
            fi
        fi
    done
    
    if [[ $zones_found -ge 3 ]]; then
        log_success "Found $zones_found thermal zones ✓"
        for zone in "${important_zones[@]}"; do
            echo "  - $zone"
        done
        return 0
    else
        log_error "Only found $zones_found thermal zones (need at least 3)"
        return 1
    fi
}

test_hwmon_sensors() {
    run_test "Checking hwmon temperature sensors"
    
    local sensors_found=0
    local important_sensors=()
    
    # Check specific hwmon sensors
    local target_sensors=(
        "/sys/class/hwmon/hwmon7/temp1_input:coretemp"
        "/sys/class/hwmon/hwmon6/temp1_input:dell_smm"
        "/sys/class/hwmon/hwmon5/temp1_input:dell_ddv"
    )
    
    for sensor_info in "${target_sensors[@]}"; do
        IFS=':' read -r sensor_path sensor_name <<< "$sensor_info"
        
        if [[ -f "$sensor_path" ]]; then
            temp=$(cat "$sensor_path" 2>/dev/null || echo "0")
            temp_celsius=$((temp / 1000))
            
            # Get hwmon name
            hwmon_dir=$(dirname "$sensor_path")
            hwmon_name=$(cat "$hwmon_dir/name" 2>/dev/null || echo "unknown")
            
            important_sensors+=("$sensor_name ($hwmon_name): ${temp_celsius}°C")
            ((sensors_found++))
            
            # Warn about high temperatures
            if [[ $temp_celsius -gt 90 ]]; then
                log_warning "$sensor_name temperature critical: ${temp_celsius}°C"
            elif [[ $temp_celsius -gt 80 ]]; then
                log_warning "$sensor_name temperature high: ${temp_celsius}°C"
            fi
        fi
    done
    
    if [[ $sensors_found -ge 2 ]]; then
        log_success "Found $sensors_found hwmon sensors ✓"
        for sensor in "${important_sensors[@]}"; do
            echo "  - $sensor"
        done
        return 0
    else
        log_error "Only found $sensors_found hwmon sensors (need at least 2)"
        return 1
    fi
}

test_fan_control() {
    run_test "Checking fan control capability"
    
    local fan_controls=(
        "/sys/class/hwmon/hwmon6/pwm1:dell_smm"
        "/sys/class/hwmon/hwmon5/pwm1:dell_ddv"
    )
    
    local fan_found=false
    
    for fan_info in "${fan_controls[@]}"; do
        IFS=':' read -r fan_path fan_name <<< "$fan_info"
        
        if [[ -f "$fan_path" ]]; then
            # Test read access
            current_pwm=$(cat "$fan_path" 2>/dev/null || echo "0")
            
            # Test write access
            if echo "$current_pwm" > "$fan_path" 2>/dev/null; then
                log_success "Fan control available: $fan_name (PWM: $current_pwm) ✓"
                fan_found=true
                
                # Check for fan speed sensor
                fan_dir=$(dirname "$fan_path")
                if [[ -f "$fan_dir/fan1_input" ]]; then
                    fan_rpm=$(cat "$fan_dir/fan1_input" 2>/dev/null || echo "0")
                    echo "  - Current fan speed: $fan_rpm RPM"
                fi
                break
            fi
        fi
    done
    
    if [[ $fan_found == true ]]; then
        return 0
    else
        log_error "No working fan control found"
        return 1
    fi
}

test_cpu_frequency_control() {
    run_test "Checking CPU frequency control"
    
    local intel_pstate="/sys/devices/system/cpu/intel_pstate"
    
    if [[ ! -d "$intel_pstate" ]]; then
        log_error "Intel P-State driver not found"
        return 1
    fi
    
    # Test max_perf_pct control
    local max_perf_file="$intel_pstate/max_perf_pct"
    if [[ -f "$max_perf_file" ]]; then
        current_max=$(cat "$max_perf_file" 2>/dev/null || echo "0")
        
        # Test write access
        if echo "$current_max" > "$max_perf_file" 2>/dev/null; then
            log_success "CPU frequency control available (max: $current_max%) ✓"
        else
            log_error "CPU frequency control not writable"
            return 1
        fi
    else
        log_error "CPU max_perf_pct control not found"
        return 1
    fi
    
    # Test turbo control
    local turbo_file="$intel_pstate/no_turbo"
    if [[ -f "$turbo_file" ]]; then
        turbo_status=$(cat "$turbo_file" 2>/dev/null || echo "0")
        if [[ $turbo_status == "0" ]]; then
            echo "  - Turbo boost: Enabled"
        else
            echo "  - Turbo boost: Disabled"
        fi
    fi
    
    return 0
}

test_sensors_command() {
    run_test "Checking sensors command availability"
    
    if command -v sensors >/dev/null 2>&1; then
        log_success "sensors command available ✓"
        
        # Test sensors output
        if sensors >/dev/null 2>&1; then
            echo "  - sensors command working"
            
            # Count temperature sensors
            temp_count=$(sensors 2>/dev/null | grep -c "°C" || echo "0")
            echo "  - Found $temp_count temperature readings"
            
            return 0
        else
            log_warning "sensors command exists but not working"
            return 1
        fi
    else
        log_warning "sensors command not found (install lm-sensors)"
        return 1
    fi
}

test_python_environment() {
    run_test "Checking Python environment"
    
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_success "Python 3 available: $python_version ✓"
        
        # Test version compatibility
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            echo "  - Version compatible (3.8+ required)"
        else
            log_warning "Python version may be too old (3.8+ recommended)"
        fi
        
        return 0
    else
        log_error "Python 3 not found"
        return 1
    fi
}

test_current_temperatures() {
    run_test "Checking current system temperatures"
    
    local max_temp=0
    local temp_warnings=0
    
    # Check thermal zones
    for zone in /sys/class/thermal/thermal_zone*; do
        if [[ -f "$zone/temp" && -f "$zone/type" ]]; then
            temp=$(cat "$zone/temp" 2>/dev/null || echo "0")
            temp_celsius=$((temp / 1000))
            type=$(cat "$zone/type" 2>/dev/null || echo "unknown")
            
            if [[ $temp_celsius -gt $max_temp ]]; then
                max_temp=$temp_celsius
            fi
            
            if [[ $temp_celsius -gt 90 ]]; then
                log_warning "High temperature: $type = ${temp_celsius}°C"
                ((temp_warnings++))
            fi
        fi
    done
    
    if [[ $max_temp -gt 0 ]]; then
        if [[ $max_temp -gt 95 ]]; then
            log_warning "System running very hot: ${max_temp}°C (thermal guardian needed urgently)"
        elif [[ $max_temp -gt 85 ]]; then
            log_warning "System running hot: ${max_temp}°C (thermal guardian recommended)"
        else
            log_success "System temperatures acceptable: max ${max_temp}°C ✓"
        fi
        
        if [[ $temp_warnings -gt 0 ]]; then
            echo "  - $temp_warnings sensors above 90°C"
        fi
        
        return 0
    else
        log_error "Could not read any temperatures"
        return 1
    fi
}

test_systemd_support() {
    run_test "Checking systemd support"
    
    if systemctl --version >/dev/null 2>&1; then
        systemd_version=$(systemctl --version | head -1 | awk '{print $2}')
        log_success "systemd available: version $systemd_version ✓"
        
        # Check if we can create service files
        if [[ -w "/etc/systemd/system" ]]; then
            echo "  - Can create service files"
        else
            log_warning "Cannot write to /etc/systemd/system"
        fi
        
        return 0
    else
        log_error "systemd not available"
        return 1
    fi
}

# Hardware identification
identify_hardware() {
    log_info "=== HARDWARE IDENTIFICATION ==="
    
    # System information
    if [[ -f /sys/class/dmi/id/product_name ]]; then
        product_name=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "unknown")
        echo "Product: $product_name"
    fi
    
    if [[ -f /sys/class/dmi/id/sys_vendor ]]; then
        vendor=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null || echo "unknown")
        echo "Vendor: $vendor"
    fi
    
    # CPU information
    if [[ -f /proc/cpuinfo ]]; then
        cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
        echo "CPU: $cpu_model"
    fi
    
    echo
}

# Summary report
generate_summary() {
    log_info "=== TEST SUMMARY ==="
    
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    local pass_rate=0
    
    if [[ $total_tests -gt 0 ]]; then
        pass_rate=$(( (TESTS_PASSED * 100) / total_tests ))
    fi
    
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    echo "Pass rate: ${pass_rate}%"
    echo
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "🎯 ALL TESTS PASSED - System ready for thermal guardian deployment!"
        echo
        echo "Next step: Run './deploy_thermal_guardian.sh' to install"
        return 0
    elif [[ $pass_rate -ge 70 ]]; then
        log_warning "⚠️  MOSTLY COMPATIBLE - Some features may not work optimally"
        echo
        echo "You can try deployment with: './deploy_thermal_guardian.sh --force'"
        return 1
    else
        log_error "❌ COMPATIBILITY ISSUES - Fix errors before deployment"
        echo
        echo "Review the failed tests above and resolve issues before proceeding"
        return 2
    fi
}

# Main test execution
main() {
    echo
    echo "========================================"
    echo "🌡️  THERMAL GUARDIAN COMPATIBILITY TEST"
    echo "========================================"
    echo "System: Dell LAT5150DRVMIL"
    echo "Date: $(date)"
    echo
    
    identify_hardware
    
    # Run all tests
    log_info "=== RUNNING COMPATIBILITY TESTS ==="
    echo
    
    test_root_permissions
    test_thermal_zones
    test_hwmon_sensors
    test_fan_control
    test_cpu_frequency_control
    test_sensors_command
    test_python_environment
    test_current_temperatures
    test_systemd_support
    
    echo
    generate_summary
}

# Execute main function
main "$@"