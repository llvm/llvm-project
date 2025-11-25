#!/bin/bash
# DELL LATITUDE 5450 MIL-SPEC DEEP ENUMERATION v8.0
# Based on project documentation - JRTC1, DSMIL, Mode 5, hidden features

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="milspec_deep_enum_${TIMESTAMP}.txt"
ACPI_DUMP="acpi_tables_${TIMESTAMP}.txt"
DMI_FULL="dmi_complete_${TIMESTAMP}.txt"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    DELL LATITUDE 5450 MIL-SPEC DEEP ENUMERATION v8.0        ║${NC}"
echo -e "${CYAN}║         JRTC1 / DSMIL / Mode 5 Detection Suite               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Must run as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}[!] This script MUST run as root for deep enumeration${NC}"
    exit 1
fi

# =============================================================================
# DEEP DMI ENUMERATION
# =============================================================================
echo -e "\n${PURPLE}═══ DEEP DMI ENUMERATION ═══${NC}" | tee -a "$REPORT"

# Dump COMPLETE DMI including Type 8 where JRTC1 was found
echo -e "${YELLOW}Extracting ALL DMI data...${NC}" | tee -a "$REPORT"
dmidecode > "$DMI_FULL" 2>&1

# Search for JRTC1 in Type 8
echo -e "\n${YELLOW}Searching DMI Type 8 for JRTC1...${NC}" | tee -a "$REPORT"
dmidecode -t 8 2>/dev/null | grep -A5 -B5 "JRTC1" | tee -a "$REPORT" || echo "Not found in standard query" | tee -a "$REPORT"

# Check for OUT OF SPEC family
echo -e "\n${YELLOW}Checking System Family...${NC}" | tee -a "$REPORT"
family=$(dmidecode -s system-family 2>/dev/null)
echo "System Family: '$family'" | tee -a "$REPORT"
if [[ "$family" == *"OUT OF SPEC"* ]]; then
    echo -e "${RED}[!] OUT OF SPEC DETECTED - MILITARY VARIANT${NC}" | tee -a "$REPORT"
fi

# Search entire DMI for military identifiers
echo -e "\n${YELLOW}Deep searching DMI for military markers...${NC}" | tee -a "$REPORT"
for marker in "JRTC" "DSMIL" "MIL-SPEC" "OUT OF SPEC" "Mode 5" "Mode5"; do
    if grep -q "$marker" "$DMI_FULL"; then
        echo -e "${RED}[!] Found '$marker' in DMI data${NC}" | tee -a "$REPORT"
        grep -n "$marker" "$DMI_FULL" | head -5 | tee -a "$REPORT"
    fi
done

# =============================================================================
# HIDDEN MEMORY DETECTION
# =============================================================================
echo -e "\n${PURPLE}═══ HIDDEN MEMORY ANALYSIS ═══${NC}" | tee -a "$REPORT"

# E820 memory map
echo -e "${YELLOW}E820 Memory Map:${NC}" | tee -a "$REPORT"
dmesg | grep -E "BIOS-e820|e820:" | head -20 | tee -a "$REPORT"

# Reserved regions
echo -e "\n${YELLOW}Reserved Memory Regions:${NC}" | tee -a "$REPORT"
dmesg | grep -i "reserved" | grep -i "memory" | head -10 | tee -a "$REPORT"

# Memory holes
echo -e "\n${YELLOW}Checking for memory holes...${NC}" | tee -a "$REPORT"
if dmesg | grep -q "Memory hole"; then
    echo -e "${RED}[!] Memory hole detected - possible hidden segment${NC}" | tee -a "$REPORT"
    dmesg | grep "Memory hole" | tee -a "$REPORT"
fi

# =============================================================================
# ACPI TABLE EXTRACTION
# =============================================================================
echo -e "\n${PURPLE}═══ ACPI TABLE EXTRACTION ═══${NC}" | tee -a "$REPORT"

# Extract DSDT for DSMIL search
echo -e "${YELLOW}Extracting DSDT table...${NC}" | tee -a "$REPORT"
if [ -f /sys/firmware/acpi/tables/DSDT ]; then
    cp /sys/firmware/acpi/tables/DSDT /tmp/dsdt.dat
    if command -v iasl >/dev/null 2>&1; then
        iasl -d /tmp/dsdt.dat 2>/dev/null
        if [ -f /tmp/dsdt.dsl ]; then
            echo -e "${GREEN}[+] DSDT decompiled${NC}" | tee -a "$REPORT"
            # Search for DSMIL devices
            for i in {0..9}; do
                if grep -q "DSMIL0D$i" /tmp/dsdt.dsl; then
                    echo -e "${RED}[!] DSMIL0D$i found in ACPI DSDT!${NC}" | tee -a "$REPORT"
                    grep -A10 -B2 "DSMIL0D$i" /tmp/dsdt.dsl | head -20 | tee -a "$REPORT"
                fi
            done
        fi
    else
        # Fallback to strings
        echo -e "${YELLOW}Using strings to search DSDT...${NC}" | tee -a "$REPORT"
        strings /sys/firmware/acpi/tables/DSDT | grep -E "DSMIL|JRTC|MIL" | head -10 | tee -a "$REPORT"
    fi
fi

# List all ACPI tables
echo -e "\n${YELLOW}All ACPI tables:${NC}" | tee -a "$REPORT"
ls -la /sys/firmware/acpi/tables/ | tee -a "$REPORT"

# =============================================================================
# SMBIOS TOKEN DEEP SCAN
# =============================================================================
echo -e "\n${PURPLE}═══ SMBIOS TOKEN DEEP SCAN ═══${NC}" | tee -a "$REPORT"

# Check if token interface exists
TOKENS_DIR="/sys/devices/platform/dell-smbios.0/tokens"
if [ -d "$TOKENS_DIR" ]; then
    echo -e "${GREEN}[+] Token interface found${NC}" | tee -a "$REPORT"
    
    # Count ALL entries
    total_entries=$(find "$TOKENS_DIR" -maxdepth 1 -type d | wc -l)
    echo "Total directory entries: $((total_entries - 1))" | tee -a "$REPORT"
    
    # List first 20 tokens with details
    echo -e "\n${YELLOW}Token inventory:${NC}" | tee -a "$REPORT"
    for token_dir in "$TOKENS_DIR"/*; do
        if [ -d "$token_dir" ]; then
            token=$(basename "$token_dir")
            value="N/A"
            location="N/A"
            [ -r "$token_dir/value" ] && value=$(cat "$token_dir/value" 2>/dev/null || echo "ERROR")
            [ -r "$token_dir/location" ] && location=$(cat "$token_dir/location" 2>/dev/null || echo "ERROR")
            
            # Convert hex to decimal for military range check
            if [[ $token =~ ^[0-9A-Fa-f]+$ ]]; then
                dec=$((16#$token))
                if [ $dec -ge 8000 ] && [ $dec -le 8014 ]; then
                    echo -e "${RED}[!] MILITARY TOKEN: $token (dec: $dec) = $value @ $location${NC}" | tee -a "$REPORT"
                elif [ "$value" != "0" ] && [ "$value" != "N/A" ]; then
                    echo "Active: $token = $value @ $location" | tee -a "$REPORT"
                fi
            fi
        fi
    done | head -30
    
    # Specific military token check
    echo -e "\n${YELLOW}Checking military range 8000-8014 (0x1F40-0x1F4E):${NC}" | tee -a "$REPORT"
    for i in $(seq 8000 8014); do
        hex=$(printf "%04X" $i)
        if [ -d "$TOKENS_DIR/$hex" ]; then
            echo -e "${RED}[!] Token $hex (decimal $i) EXISTS${NC}" | tee -a "$REPORT"
        fi
    done
else
    echo -e "${RED}[!] Token interface NOT FOUND - checking alternative locations${NC}" | tee -a "$REPORT"
    find /sys/devices/platform -name "*dell*" -o -name "*smbios*" 2>/dev/null | tee -a "$REPORT"
fi

# WMI token access
echo -e "\n${YELLOW}Checking WMI interfaces:${NC}" | tee -a "$REPORT"
ls -la /sys/bus/wmi/devices/ 2>/dev/null | grep -i dell | tee -a "$REPORT"

# =============================================================================
# MODE 5 SEARCH
# =============================================================================
echo -e "\n${PURPLE}═══ MODE 5 DETECTION ═══${NC}" | tee -a "$REPORT"

# Direct check
if [ -d /sys/firmware/dell/mode5 ]; then
    echo -e "${RED}[!] MODE 5 DIRECTORY FOUND!${NC}" | tee -a "$REPORT"
    ls -la /sys/firmware/dell/mode5/ | tee -a "$REPORT"
else
    echo "Mode 5 directory not present" | tee -a "$REPORT"
fi

# Search sysfs for mode5 references
echo -e "\n${YELLOW}Searching sysfs for Mode 5...${NC}" | tee -a "$REPORT"
find /sys -name "*mode5*" -o -name "*mode_5*" 2>/dev/null | tee -a "$REPORT"

# =============================================================================
# HARDWARE TEST POINTS
# =============================================================================
echo -e "\n${PURPLE}═══ HARDWARE TEST POINTS ═══${NC}" | tee -a "$REPORT"

# GPIO enumeration for test points
echo -e "${YELLOW}GPIO chips:${NC}" | tee -a "$REPORT"
ls -la /sys/class/gpio/gpiochip* 2>/dev/null | tee -a "$REPORT"

# Check for GPIO 147 (TP_MODE5)
if [ -d /sys/class/gpio ]; then
    echo -e "\n${YELLOW}Checking for TP_MODE5 GPIO 147...${NC}" | tee -a "$REPORT"
    if [ -w /sys/class/gpio/export ]; then
        echo 147 > /sys/class/gpio/export 2>/dev/null || echo "GPIO 147 export failed" | tee -a "$REPORT"
        if [ -d /sys/class/gpio/gpio147 ]; then
            echo -e "${RED}[!] GPIO 147 accessible - TP_MODE5 test point${NC}" | tee -a "$REPORT"
        fi
    fi
fi

# =============================================================================
# KERNEL MODULE SEARCH
# =============================================================================
echo -e "\n${PURPLE}═══ KERNEL MODULE SEARCH ═══${NC}" | tee -a "$REPORT"

# Check for dell-milspec module
echo -e "${YELLOW}Searching for military modules...${NC}" | tee -a "$REPORT"
for mod in "dell-milspec" "dell_milspec" "dsmil" "dell-mode5"; do
    if lsmod | grep -q "$mod"; then
        echo -e "${RED}[!] Module $mod is loaded${NC}" | tee -a "$REPORT"
    fi
    if find /lib/modules -name "*${mod}*" 2>/dev/null | grep -q .; then
        echo -e "${RED}[!] Module $mod found in filesystem${NC}" | tee -a "$REPORT"
    fi
done

# =============================================================================
# REAL AVX-512 TEST
# =============================================================================
echo -e "\n${PURPLE}═══ AVX-512 P-CORE TEST ═══${NC}" | tee -a "$REPORT"

cat > /tmp/avx512_test.c << 'EOF'
#include <stdio.h>
#include <immintrin.h>
#include <signal.h>
#include <setjmp.h>

static jmp_buf jmpbuf;

void sigill_handler(int sig) {
    longjmp(jmpbuf, 1);
}

int main() {
    signal(SIGILL, sigill_handler);
    
    printf("Testing AVX-512 on this core...\n");
    
    if (setjmp(jmpbuf) == 0) {
        __m512 a = _mm512_set1_ps(1.0f);
        __m512 b = _mm512_set1_ps(2.0f);
        __m512 c = _mm512_add_ps(a, b);
        float result[16];
        _mm512_storeu_ps(result, c);
        printf("SUCCESS: AVX-512 works! Result[0] = %.1f\n", result[0]);
        return 0;
    } else {
        printf("FAILED: Illegal instruction - No AVX-512\n");
        return 1;
    }
}
EOF

if gcc -mavx512f -o /tmp/avx512_test /tmp/avx512_test.c 2>/dev/null; then
    echo -e "${YELLOW}Testing P-cores (0-11):${NC}" | tee -a "$REPORT"
    taskset -c 0-11 /tmp/avx512_test | tee -a "$REPORT"
    
    echo -e "\n${YELLOW}Testing E-cores (12-21):${NC}" | tee -a "$REPORT"
    taskset -c 12-21 /tmp/avx512_test 2>&1 | tee -a "$REPORT"
else
    echo -e "${RED}[!] Failed to compile AVX-512 test${NC}" | tee -a "$REPORT"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "\n${CYAN}════════════════════════════════════════════════════${NC}" | tee -a "$REPORT"
echo -e "${PURPLE}DEEP ENUMERATION SUMMARY${NC}" | tee -a "$REPORT"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}" | tee -a "$REPORT"

{
    echo ""
    echo "Service Tag: C6FHC54"
    echo "Expected: Dell Latitude 5450 MIL-SPEC JRTC1"
    echo ""
    echo "Files generated:"
    echo "  Main report: $REPORT"
    echo "  DMI dump: $DMI_FULL"
    echo "  ACPI dump: $ACPI_DUMP"
    echo ""
    echo "Key areas to check:"
    echo "1. Search $DMI_FULL for JRTC1 (DMI Type 8)"
    echo "2. Check ACPI DSDT for DSMIL0D0-DSMIL0D9"
    echo "3. Verify 400+ SMBIOS tokens"
    echo "4. Look for OUT OF SPEC family"
    echo ""
    echo "Next steps if features not found:"
    echo "- Features may require BIOS service mode activation"
    echo "- Check physical test points (TP_MODE5)"
    echo "- Review boot parameters for blocking"
} | tee -a "$REPORT"

echo -e "\n${GREEN}[✓] Deep enumeration complete${NC}"
echo -e "${YELLOW}Review files for hidden military features${NC}"
