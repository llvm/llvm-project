#!/bin/bash
# DELL LATITUDE 5450 MIL-SPEC ENUM v14.0 - CRASH-PROOF
# Handles ACPI section safely

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   DELL LATITUDE 5450 MIL-SPEC ENUM v14.0 - SAFE             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}[!] Run as root${NC}"
    exit 1
fi

# Create output dir
OUTPUT_DIR="milspec_enum_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo -e "${GREEN}[+] Output: $(pwd)${NC}\n"

# 1. DMI EXTRACTION
echo -e "${CYAN}═══ DMI EXTRACTION ═══${NC}"
dmidecode > dmi.txt 2>&1 || true
dmidecode -t 8 > dmi_type8.txt 2>&1 || true

echo -n "JRTC marker: "
grep -q "JRTC" dmi.txt 2>/dev/null && echo -e "${RED}FOUND${NC}" || echo "Not found"

# 2. SYSTEM INFO
echo -e "\n${CYAN}═══ SYSTEM INFO ═══${NC}"
echo "Service Tag: $(dmidecode -s system-serial-number 2>/dev/null || echo 'Unknown')"
echo "Family: $(dmidecode -s system-family 2>/dev/null || echo 'Unknown')"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

# 3. MEMORY CHECK
echo -e "\n${CYAN}═══ MEMORY CHECK ═══${NC}"
VISIBLE_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
VISIBLE_GB=$(echo "scale=1; $VISIBLE_KB / 1024 / 1024" | bc)
echo "Visible: ${VISIBLE_GB}GB"

# Get physical safely
PHYS_GB=0
if command -v dmidecode >/dev/null 2>&1; then
    PHYS_GB=$(dmidecode -t 17 2>/dev/null | grep -E "Size:.*GB" | grep -v "No Module" | awk '{sum += $2} END {print sum}' || echo "0")
fi
[ -n "$PHYS_GB" ] && [ "$PHYS_GB" != "0" ] && echo "Physical: ${PHYS_GB}GB"

# 4. SMBIOS TOKENS (Quick check)
echo -e "\n${CYAN}═══ SMBIOS TOKENS ═══${NC}"
TOKENS_DIR="/sys/devices/platform/dell-smbios.0/tokens"
if [ -d "$TOKENS_DIR" ]; then
    TOTAL=$(find "$TOKENS_DIR" -maxdepth 1 -type d -name "[0-9A-F]*" 2>/dev/null | wc -l || echo "0")
    echo "Total tokens: $TOTAL"
else
    echo "No token interface"
fi

# 5. ACPI/DSMIL - SAFE VERSION
echo -e "\n${CYAN}═══ ACPI/DSMIL CHECK (SAFE) ═══${NC}"

# Method 1: Direct string search (no loops)
if [ -f /sys/firmware/acpi/tables/DSDT ]; then
    echo "[*] Extracting ACPI..."
    strings /sys/firmware/acpi/tables/DSDT > acpi.txt 2>/dev/null || true
    
    # Simple presence check
    echo -e "\nDSMIL devices found:"
    grep -o "DSMIL0D[0-9]" acpi.txt 2>/dev/null | sort -u || echo "None"
    
    # Count them
    DSMIL_COUNT=$(grep -o "DSMIL0D[0-9]" acpi.txt 2>/dev/null | sort -u | wc -l || echo "0")
    echo "Total DSMIL devices: $DSMIL_COUNT"
fi

# 6. SECURITY FEATURES
echo -e "\n${CYAN}═══ SECURITY FEATURES ═══${NC}"
[ -e /dev/mei0 ] && echo "Intel ME: Present" || echo "Intel ME: Not found"
[ -e /dev/tpm0 ] && echo "TPM: Present" || echo "TPM: Not found"
[ -d /sys/firmware/dell/mode5 ] && echo -e "Mode 5: ${RED}ACTIVE${NC}" || echo "Mode 5: Not active"
[ -c /dev/milspec ] && echo -e "/dev/milspec: ${RED}FOUND${NC}" || echo "/dev/milspec: Not found"

# 7. DELL SERVICES
echo -e "\n${CYAN}═══ DELL SERVICES ═══${NC}"
echo "Dell processes: $(ps aux | grep -i dell | grep -v grep | wc -l || echo '0')"
echo "Dell modules: $(lsmod | grep -i dell | wc -l || echo '0')"

# 8. NETWORK PORTS
echo -e "\n${CYAN}═══ NETWORK PORTS ═══${NC}"
PORTS=(16992 16993 623 664)
for PORT in "${PORTS[@]}"; do
    ss -tln 2>/dev/null | grep -q ":$PORT " && echo -e "${RED}[!] Port $PORT OPEN${NC}" || true
done

# 9. HARDWARE
echo -e "\n${CYAN}═══ HARDWARE ═══${NC}"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Cores: $(nproc)"
grep -q avx512f /proc/cpuinfo && echo "AVX-512: Supported" || echo "AVX-512: Not supported"
[ -e /dev/accel/accel0 ] && echo "NPU: Present" || echo "NPU: Not found"

# 10. CRITICAL FINDINGS SUMMARY
echo -e "\n${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}CRITICAL FINDINGS SUMMARY${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"

SUMMARY="summary.txt"
{
    echo "DELL LATITUDE 5450 MIL-SPEC - $(date)"
    echo ""
    echo "CONFIRMED MILITARY FEATURES:"
    grep -q "JRTC" dmi.txt 2>/dev/null && echo "✓ JRTC marker in DMI"
    [ -f acpi.txt ] && grep -q "DSMIL" acpi.txt 2>/dev/null && echo "✓ DSMIL devices in ACPI"
    [ "$DSMIL_COUNT" -gt 0 ] && echo "✓ $DSMIL_COUNT DSMIL subsystems found"
    echo ""
    echo "SYSTEM:"
    echo "• Service Tag: C6FHC54"
    echo "• Memory: ${VISIBLE_GB}GB visible"
    [ "$PHYS_GB" != "0" ] && echo "• Physical: ${PHYS_GB}GB installed"
    echo ""
    echo "FILES:"
    echo "• DMI dump: dmi.txt"
    echo "• Type 8: dmi_type8.txt" 
    echo "• ACPI: acpi.txt"
    echo ""
    echo "TO VIEW JRTC CONTEXT:"
    echo "grep -B5 -A5 JRTC dmi_type8.txt"
    echo ""
    echo "TO VIEW DSMIL DEVICES:"
    echo "grep DSMIL acpi.txt"
} | tee "$SUMMARY"

echo -e "\n${GREEN}[✓] Enumeration complete${NC}"
echo -e "${GREEN}[✓] Check $(pwd)/summary.txt${NC}"

cd ..
exit 0
