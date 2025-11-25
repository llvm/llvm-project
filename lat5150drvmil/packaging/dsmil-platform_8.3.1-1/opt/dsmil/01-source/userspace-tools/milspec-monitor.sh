#!/bin/bash
# Monitor and log MIL-SPEC activation during boot

LOG=/var/log/milspec-boot.log
exec > >(tee -a $LOG)
exec 2>&1

echo "[$(date)] MIL-SPEC Boot Monitor Starting"

# Check sysfs
echo "=== Sysfs Status ==="
for f in /sys/devices/platform/dell-milspec/*; do
    [ -f "$f" ] && echo "$f: $(cat $f)"
done

# Check proc
echo "=== Proc Status ==="
cat /proc/milspec 2>/dev/null

# Check debugfs
echo "=== Debug Status ==="
cat /sys/kernel/debug/dell-milspec/boot_progress 2>/dev/null

# Monitor dmesg
echo "=== Kernel Messages ==="
dmesg | grep -i "mil-spec\|dsmil\|mode"

# Watch for ACPI events
echo "=== ACPI Events ==="
acpid -f -l /var/log/milspec-acpi.log &
ACPI_PID=$!

sleep 5
kill $ACPI_PID 2>/dev/null

echo "[$(date)] MIL-SPEC Boot Monitor Complete"
