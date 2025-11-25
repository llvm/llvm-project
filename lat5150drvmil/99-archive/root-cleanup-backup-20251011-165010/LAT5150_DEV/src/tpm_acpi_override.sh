#!/bin/bash

# NSA TPM ACPI Override for STMicroelectronics TPM0176
# Bypasses firmware buffer size mismatch

echo "NSA TPM ACPI Override Tool"
echo "Target: Dell Latitude 5450 MIL-SPEC"
echo "TPM: STMicroelectronics STM0176"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "Root access required"
   exit 1
fi

# Create custom ACPI override
cat > /tmp/tpm_override.asl << 'EOF'
DefinitionBlock ("", "SSDT", 2, "NSA", "TPMFIX", 0x00001000)
{
    Scope (\_SB)
    {
        Device (TPM2)
        {
            Name (_HID, "MSFT0101")  // TPM 2.0 Device
            Name (_CID, "PNP0C31")   // Compatible ID
            Name (_UID, Zero)

            Method (_STA, 0, NotSerialized)
            {
                Return (0x0F)  // Present, Enabled, Show in UI, Functioning
            }

            Name (_CRS, ResourceTemplate ()
            {
                Memory32Fixed (ReadWrite,
                    0xFED40000,         // Address Base
                    0x00005000,         // Address Length
                    )
            })

            // NSA Buffer Fix - Force TIS mode
            Name (BUFR, Package (0x02)
            {
                0x1000,  // Command Buffer Size (4KB)
                0x1000   // Response Buffer Size (4KB)
            })
        }
    }
}
EOF

# Compile ACPI override
iasl -tc /tmp/tpm_override.asl

# Create initrd with override
if [ -f /tmp/tpm_override.aml ]; then
    mkdir -p /tmp/acpi_override/kernel/firmware/acpi-tables
    cp /tmp/tpm_override.aml /tmp/acpi_override/kernel/firmware/acpi-tables/

    cd /tmp/acpi_override
    find . | cpio -H newc --create | gzip > /boot/acpi_override.img

    echo "ACPI override created: /boot/acpi_override.img"
    echo "Add to GRUB: initrd /acpi_override.img"
else
    echo "ACPI compilation failed"
fi

# Alternative: Runtime ACPI injection
echo "Attempting runtime ACPI injection..."
echo "add" > /sys/bus/acpi/drivers/tpm_crb/bind 2>/dev/null
echo "MSF0101:00" > /sys/bus/acpi/drivers/tpm_crb/bind 2>/dev/null

# Force TIS driver binding
echo "Forcing TIS driver binding..."
modprobe -r tpm_crb
modprobe tpm_tis force=1 interrupts=0

echo "Override complete. Check dmesg for results."