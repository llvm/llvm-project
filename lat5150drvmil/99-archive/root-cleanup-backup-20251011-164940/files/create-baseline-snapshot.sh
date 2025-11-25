#!/bin/bash
# Create baseline system snapshot before SMBIOS token operations
# CRITICAL: Document system state before any token enumeration

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR="/home/john/LAT5150DRVMIL/baseline_${TIMESTAMP}"

echo "=== Creating System Baseline Snapshot ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Directory: ${BASELINE_DIR}"

# Create baseline directory
mkdir -p "${BASELINE_DIR}"

# 1. System Information
echo "Gathering system information..."
{
    echo "=== System Information ==="
    uname -a
    echo ""
    echo "=== Dell System Info ==="
    sudo dmidecode -t system
    echo ""
    echo "=== BIOS Information ==="
    sudo dmidecode -t bios
} > "${BASELINE_DIR}/system_info.txt" 2>&1

# 2. SMBIOS Current State
echo "Capturing SMBIOS state..."
{
    echo "=== SMBIOS Tables ==="
    sudo dmidecode
    echo ""
    echo "=== Dell SMBIOS Tokens (if available) ==="
    # Try to list tokens if tools available
    which smbios-token-ctl >/dev/null 2>&1 && sudo smbios-token-ctl --list || echo "smbios-token-ctl not available"
} > "${BASELINE_DIR}/smbios_baseline.txt" 2>&1

# 3. Kernel Modules
echo "Documenting kernel modules..."
{
    echo "=== Loaded Modules ==="
    lsmod | sort
    echo ""
    echo "=== Dell Modules ==="
    lsmod | grep -i dell
} > "${BASELINE_DIR}/kernel_modules.txt" 2>&1

# 4. Memory and Resources
echo "Recording memory state..."
{
    echo "=== Memory Map ==="
    sudo cat /proc/iomem
    echo ""
    echo "=== Reserved Regions ==="
    sudo cat /proc/iomem | grep -i reserved
} > "${BASELINE_DIR}/memory_map.txt" 2>&1

# 5. Temperature Baseline
echo "Recording thermal baseline..."
{
    echo "=== Temperature Sensors ==="
    sensors
    echo ""
    echo "=== CPU Temperature ==="
    cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null
} > "${BASELINE_DIR}/thermal_baseline.txt" 2>&1

# 6. System Logs Snapshot
echo "Capturing system logs..."
{
    echo "=== Last 100 Kernel Messages ==="
    sudo dmesg | tail -100
} > "${BASELINE_DIR}/kernel_messages.txt" 2>&1

# 7. PCI Devices
echo "Documenting PCI devices..."
{
    echo "=== PCI Devices ==="
    lspci -vnn
    echo ""
    echo "=== Dell Subsystems ==="
    lspci -vnn | grep -i dell
} > "${BASELINE_DIR}/pci_devices.txt" 2>&1

# 8. Process Snapshot
echo "Recording process state..."
{
    echo "=== Running Processes ==="
    ps aux
    echo ""
    echo "=== System Load ==="
    uptime
    echo ""
    echo "=== Memory Usage ==="
    free -h
} > "${BASELINE_DIR}/process_state.txt" 2>&1

# 9. Security State
echo "Documenting security state..."
{
    echo "=== SELinux Status ==="
    getenforce 2>/dev/null || echo "SELinux not available"
    echo ""
    echo "=== AppArmor Status ==="
    sudo aa-status 2>/dev/null || echo "AppArmor not available"
    echo ""
    echo "=== Secure Boot Status ==="
    mokutil --sb-state 2>/dev/null || echo "mokutil not available"
} > "${BASELINE_DIR}/security_state.txt" 2>&1

# 10. Create manifest
echo "Creating manifest..."
{
    echo "Baseline Snapshot Manifest"
    echo "========================="
    echo "Created: ${TIMESTAMP}"
    echo "System: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "Dell Model: $(sudo dmidecode -s system-product-name)"
    echo "BIOS Version: $(sudo dmidecode -s bios-version)"
    echo ""
    echo "Files in this baseline:"
    ls -la "${BASELINE_DIR}/"
} > "${BASELINE_DIR}/MANIFEST.txt"

# Create compressed backup
echo "Creating compressed backup..."
tar -czf "${BASELINE_DIR}.tar.gz" -C "$(dirname ${BASELINE_DIR})" "$(basename ${BASELINE_DIR})"

echo ""
echo "=== Baseline Snapshot Complete ==="
echo "Directory: ${BASELINE_DIR}"
echo "Archive: ${BASELINE_DIR}.tar.gz"
echo ""
echo "System baseline has been captured. Safe to proceed with SMBIOS enumeration."
echo "In case of issues, this baseline can be used for comparison and recovery."