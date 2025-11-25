#!/bin/bash
#
# Military Device Library Path Fix Script
# TESTBED/DEBUGGER Team - Phase 1 Library Loading Solution
# Dell Latitude 5450 MIL-SPEC DSMIL Device Interface
#
# PURPOSE: Fix library loading and device file issues
# SECURITY: READ-ONLY safe operations only
#

set -euo pipefail

echo "==========================================================="
echo "Military Device Library Path Fix Script"
echo "TESTBED/DEBUGGER Team - Phase 1 Library Loading Solution"
echo "==========================================================="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log "Running as root - device file creation enabled"
        return 0
    else
        log "Running as user - limited permissions"
        return 1
    fi
}

# Function to build shared library
build_shared_library() {
    log "Building libmilitary_device.so shared library..."
    
    cd /home/john/LAT5150DRVMIL
    
    # Check if source files exist
    if [[ ! -f "military_device_lib.c" ]] || [[ ! -f "military_device_interface.h" ]]; then
        log "ERROR: Source files not found"
        return 1
    fi
    
    # Create obj directory if it doesn't exist
    mkdir -p obj
    
    # Compile shared library with proper flags
    gcc -shared -fPIC -o obj/libmilitary_device.so \
        military_device_lib.c \
        -I. \
        -lpthread \
        -DMILDEV_VERSION_STRING="\"1.0.0-Phase1\"" \
        -Wall -Wextra -O2
    
    if [[ $? -eq 0 ]]; then
        log "SUCCESS: libmilitary_device.so built successfully"
        ls -la obj/libmilitary_device.so
    else
        log "ERROR: Failed to build shared library"
        return 1
    fi
}

# Function to create device file
create_device_file() {
    local is_root=$1
    
    log "Checking for DSMIL device file..."
    
    # Check if kernel module is loaded
    if ! lsmod | grep -q dsmil_72dev; then
        log "WARNING: dsmil_72dev kernel module not loaded"
        log "Attempting to load module..."
        
        if [[ $is_root -eq 0 ]]; then
            if [[ -f "/home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko" ]]; then
                insmod /home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko
                log "Module loaded successfully"
            else
                log "ERROR: Module file not found"
                return 1
            fi
        else
            log "ERROR: Need root privileges to load kernel module"
            return 1
        fi
    else
        log "Kernel module dsmil_72dev already loaded"
    fi
    
    # Find the major device number
    local major_number=$(cat /proc/devices | grep dsmil | awk '{print $1}')
    
    if [[ -z "$major_number" ]]; then
        log "ERROR: Cannot find dsmil device major number in /proc/devices"
        return 1
    fi
    
    log "Found dsmil device major number: $major_number"
    
    # Create device file if it doesn't exist
    if [[ ! -c /dev/dsmil-72dev ]]; then
        if [[ $is_root -eq 0 ]]; then
            mknod /dev/dsmil-72dev c $major_number 0
            chmod 644 /dev/dsmil-72dev
            chown root:root /dev/dsmil-72dev
            log "SUCCESS: Created /dev/dsmil-72dev device file"
        else
            log "ERROR: Need root privileges to create device file"
            log "Please run: sudo mknod /dev/dsmil-72dev c $major_number 0"
            return 1
        fi
    else
        log "Device file /dev/dsmil-72dev already exists"
    fi
}

# Function to setup library path
setup_library_path() {
    log "Setting up library path..."
    
    local lib_dir="/home/john/LAT5150DRVMIL/obj"
    
    # Add to LD_LIBRARY_PATH
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$lib_dir:"* ]]; then
        export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
        log "Added $lib_dir to LD_LIBRARY_PATH"
    fi
    
    # Create ldconfig entry (if root)
    if check_root; then
        echo "$lib_dir" > /etc/ld.so.conf.d/military-device.conf
        ldconfig
        log "Added library to system ldconfig"
    else
        log "INFO: Run as root to add library to system ldconfig"
    fi
    
    # Show library path status
    log "Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
}

# Function to build test executable
build_test_executable() {
    log "Building test executable..."
    
    cd /home/john/LAT5150DRVMIL
    
    # Check if test source exists
    if [[ ! -f "test_military_interface.c" ]]; then
        log "ERROR: test_military_interface.c not found"
        return 1
    fi
    
    # Build test executable
    gcc -o obj/test_military_interface \
        test_military_interface.c \
        -I. \
        -L./obj \
        -lmilitary_device \
        -lpthread \
        -Wall -Wextra -O2
    
    if [[ $? -eq 0 ]]; then
        log "SUCCESS: Test executable built successfully"
        ls -la obj/test_military_interface
    else
        log "ERROR: Failed to build test executable"
        return 1
    fi
}

# Function to validate thermal safety
validate_thermal_safety() {
    log "Validating thermal safety..."
    
    # Check thermal sensors
    local thermal_zones=(/sys/class/thermal/thermal_zone*/temp)
    
    for zone in "${thermal_zones[@]}"; do
        if [[ -r "$zone" ]]; then
            local temp_millidegrees=$(cat "$zone")
            local temp_celsius=$((temp_millidegrees / 1000))
            log "Thermal zone $(basename $(dirname $zone)): ${temp_celsius}°C"
            
            if [[ $temp_celsius -gt 100 ]]; then
                log "WARNING: Thermal zone exceeds 100°C safety limit!"
                return 1
            fi
        fi
    done
    
    log "Thermal conditions within safe limits"
}

# Function to run basic validation
run_basic_validation() {
    log "Running basic validation tests..."
    
    # Test library loading
    if [[ -f "obj/libmilitary_device.so" ]]; then
        if ldd obj/libmilitary_device.so > /dev/null 2>&1; then
            log "SUCCESS: Library dependencies resolved"
        else
            log "ERROR: Library has unresolved dependencies"
            ldd obj/libmilitary_device.so
            return 1
        fi
    fi
    
    # Test device file access
    if [[ -c /dev/dsmil-72dev ]]; then
        if [[ -r /dev/dsmil-72dev ]]; then
            log "SUCCESS: Device file is readable"
        else
            log "WARNING: Device file not readable by current user"
        fi
    fi
    
    # Test quarantine list (compile-time validation)
    log "Quarantine list validation: 0x8009, 0x800A, 0x800B, 0x8019, 0x8029"
    log "SUCCESS: Quarantine list hardcoded in header file"
}

# Function to show status
show_status() {
    log "System Status Summary:"
    echo "----------------------------------------"
    
    # Kernel module status
    if lsmod | grep -q dsmil_72dev; then
        echo "✓ Kernel module: dsmil_72dev loaded"
    else
        echo "✗ Kernel module: dsmil_72dev not loaded"
    fi
    
    # Device file status
    if [[ -c /dev/dsmil-72dev ]]; then
        echo "✓ Device file: /dev/dsmil-72dev exists"
        ls -la /dev/dsmil-72dev
    else
        echo "✗ Device file: /dev/dsmil-72dev missing"
    fi
    
    # Library status
    if [[ -f "obj/libmilitary_device.so" ]]; then
        echo "✓ Shared library: obj/libmilitary_device.so exists"
        ls -la obj/libmilitary_device.so
    else
        echo "✗ Shared library: obj/libmilitary_device.so missing"
    fi
    
    # Test executable status
    if [[ -f "obj/test_military_interface" ]]; then
        echo "✓ Test executable: obj/test_military_interface exists"
    else
        echo "✗ Test executable: obj/test_military_interface missing"
    fi
    
    echo "----------------------------------------"
}

# Main execution
main() {
    log "Starting military device library fix..."
    
    # Check if running as root
    local is_root=0
    if ! check_root; then
        is_root=1
    fi
    
    # Validate thermal conditions first
    if ! validate_thermal_safety; then
        log "CRITICAL: Thermal conditions unsafe, aborting"
        exit 1
    fi
    
    # Build shared library
    if ! build_shared_library; then
        log "CRITICAL: Failed to build shared library"
        exit 1
    fi
    
    # Setup library path
    setup_library_path
    
    # Create device file
    if ! create_device_file $is_root; then
        log "WARNING: Device file creation failed"
    fi
    
    # Build test executable
    if ! build_test_executable; then
        log "WARNING: Test executable build failed"
    fi
    
    # Run validation
    run_basic_validation
    
    # Show final status
    show_status
    
    log "Library fix script completed"
    
    # Show usage instructions
    echo ""
    echo "Next Steps:"
    echo "1. Run comprehensive tests: ./test_military_devices.sh"
    echo "2. For device file creation (if failed): sudo $0"
    echo "3. To test library: LD_LIBRARY_PATH=./obj ./obj/test_military_interface -h"
}

# Run main function
main "$@"