#!/bin/bash

# NSA TPM Module Parameter Bypass
# Comprehensive approach for STMicroelectronics TPM0176

echo "NSA TPM Module Parameter Injection"
echo "Target: STMicroelectronics STM0176"
echo ""

# Method 1: TIS Force Probe with Hardware Override
echo "=== TIS Force Probe Strategy ==="
rmmod tpm_crb tpm_tis tmp_tis_core tpm 2>/dev/null

# Force TIS with specific parameters for STM0176
modprobe tpm_tis force=1 interrupts=0 hid="STM0176"

# Check result
if [ -c /dev/tpm0 ]; then
    echo "SUCCESS: TIS force probe created /dev/tpm0"
    ls -la /dev/tpm*
else
    echo "TIS force probe failed, trying alternative parameters..."

    # Alternative 1: iTPM workarounds (Lenovo-style fix)
    rmmod tpm_tis 2>/dev/null
    modprobe tpm_tis force=1 itpm=1 interrupts=0

    if [ -c /dev/tpm0 ]; then
        echo "SUCCESS: iTPM workaround created /dev/tpm0"
    else
        echo "iTPM workaround failed, trying CRB bypass..."

        # Alternative 2: CRB with custom parameters
        rmmod tpm_tis 2>/dev/null
        modprobe tpm_crb force=1

        if [ -c /dev/tpm0 ]; then
            echo "SUCCESS: CRB force probe created /dev/tpm0"
        else
            echo "All standard methods failed - firmware manipulation required"
        fi
    fi
fi

# Method 2: ACPI Device Override
echo ""
echo "=== ACPI Device Override Strategy ==="

# Create temporary ACPI device entry
echo "Creating ACPI override for TPM device..."

# Bind TPM to platform bus directly
echo "platform:tpm_tis" > /sys/bus/platform/drivers_probe 2>/dev/null

# Force ACPI device binding
for acpi_dev in /sys/bus/acpi/devices/MSF*; do
    if [ -d "$acpi_dev" ]; then
        echo "Found ACPI TPM device: $acpi_dev"
        echo "tpm_crb" > "$acpi_dev/driver_override" 2>/dev/null
        echo "$(basename $acpi_dev)" > /sys/bus/acpi/drivers/tpm_crb/bind 2>/dev/null
    fi
done

# Method 3: Direct Hardware Probe
echo ""
echo "=== Direct Hardware Probe Strategy ==="

# Probe known TPM hardware addresses
TPM_ADDRESSES="0xFED40000 0xFED41000 0xFED42000 0xFED43000"

for addr in $TPM_ADDRESSES; do
    echo "Probing TPM at address $addr..."

    # Create manual device registration
    echo "$addr" > /sys/bus/platform/drivers/tpm_tis/new_id 2>/dev/null
done

# Method 4: Custom Device Tree Override (for systems with device tree)
echo ""
echo "=== Device Tree Override Strategy ==="

if [ -d /proc/device-tree ]; then
    echo "Device tree detected, creating TPM override..."

    # Create device tree fragment for TPM
    cat > /tmp/tpm-overlay.dts << 'EOF'
/dts-v1/;
/plugin/;

/ {
    compatible = "linux,system";

    fragment@0 {
        target-path = "/";
        __overlay__ {
            tpm@fed40000 {
                compatible = "tcg,tpm-tis-mmio";
                reg = <0xfed40000 0x5000>;
                interrupt-parent = <&gic>;
                interrupts = <0 0 4>;
                status = "okay";
            };
        };
    };
};
EOF

    # Compile and apply overlay (if dtc available)
    if command -v dtc >/dev/null 2>&1; then
        dtc -I dts -O dtb /tmp/tpm-overlay.dts -o /tmp/tpm-overlay.dtbo
        echo "Device tree overlay created"
    fi
fi

# Method 5: Runtime Module Parameter Override
echo ""
echo "=== Runtime Parameter Override ==="

# Override module parameters at runtime
echo "Overriding runtime parameters..."

# Force module parameter changes
echo 1 > /sys/module/tpm_tis/parameters/force 2>/dev/null
echo 0 > /sys/module/tpm_tis/parameters/interrupts 2>/dev/null
echo 1 > /sys/module/tpm_tis/parameters/itpm 2>/dev/null

# Trigger device rescan
echo 1 > /sys/bus/platform/drivers_autoprobe 2>/dev/null
echo 1 > /sys/bus/acpi/drivers_autoprobe 2>/dev/null

# Final status check
echo ""
echo "=== Final Status Check ==="
echo "TPM devices:"
ls -la /dev/tpm* 2>/dev/null || echo "No TPM devices found"

echo ""
echo "TPM kernel modules:"
lsmod | grep tpm

echo ""
echo "Recent TPM kernel messages:"
dmesg | grep -i tpm | tail -5

echo ""
echo "If no TPM devices created, firmware-level bypass required:"
echo "1. Compile and run: gcc -o tpm_crb_buffer_fix tpm_crb_buffer_fix.c"
echo "2. Execute: sudo ./tpm_crb_buffer_fix"
echo "3. Apply ACPI override: sudo bash tpm_acpi_override.sh"
echo "4. Run ME bypass: gcc -o intel_me_tpm_bypass intel_me_tmp_bypass.c && sudo ./intel_me_tpm_bypass"