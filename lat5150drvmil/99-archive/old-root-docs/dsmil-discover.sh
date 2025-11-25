#!/bin/bash
#
# DSMIL Hardware Discovery Tool - Shortcut Launcher
#
# This script provides easy access to the comprehensive DSMIL hardware
# discovery tool from anywhere in the repository.
#
# USAGE:
#   ./dsmil-discover.sh [OPTIONS]
#
# OPTIONS:
#   --help, -h          Show this help message
#   --report, -r        Generate comprehensive report (default)
#   --sudo              Run with sudo for deep hardware access (MSR/ME/MMIO)
#   --output FILE       Save report to file instead of stdout
#   --version, -v       Show version information
#
# FEATURES:
#   - Deep hardware scanning at SMM/MSR/Intel ME level
#   - HAP bit detection for Intel Management Engine
#   - DSMIL device and framework discovery
#   - Dell WMI interface enumeration
#   - Comprehensive security recommendations
#
# Author: DSMIL Integration Framework
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DISCOVERY_SCRIPT="${SCRIPT_DIR}/02-tools/dsmil-devices/dsmil_discover.py"

# Version information
VERSION="1.5.0"
DISCOVERY_DATE="2025-11-06"

# Default options
USE_SUDO=0
OUTPUT_FILE=""
SHOW_HELP=0
SHOW_VERSION=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=1
            shift
            ;;
        --version|-v)
            SHOW_VERSION=1
            shift
            ;;
        --sudo)
            USE_SUDO=1
            shift
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --report|-r)
            # Default behavior, just shift
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help
if [ $SHOW_HELP -eq 1 ]; then
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║              DSMIL HARDWARE DISCOVERY TOOL - HELP                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    ./dsmil-discover.sh [OPTIONS]

OPTIONS:
    --help, -h          Show this help message
    --report, -r        Generate comprehensive report (default)
    --sudo              Run with sudo for deep hardware access
    --output FILE       Save report to file instead of stdout
    --version, -v       Show version information

DISCOVERY CAPABILITIES:

    Basic Discovery:
        • Kernel modules (dcdbas, dell_wmi, dell_smbios)
        • Device nodes (/dev/smi*, /dev/dsmil*)
        • PCI devices (Dell, TPM, Security)
        • ACPI tables
        • SMI interface (ports 0xB2/0xB3)
        • Running processes
        • Firmware files
        • DMI/SMBIOS information

    Deep Hardware Discovery (requires sudo):
        • MMIO/SMRAM memory regions
        • MSR (Model Specific Registers):
          - IA32_SMBASE (SMM base address)
          - IA32_MISC_ENABLE
          - IA32_FEATURE_CONTROL
          - Platform and feature MSRs
        • Intel Management Engine (ME):
          - ME version and operational state
          - HAP bit status (High Assurance Platform)
          - AMT detection
          - Security warnings for MIL-SPEC systems
        • Dell WMI interfaces
        • EFI/UEFI variables
        • USB devices
        • BIOS settings

    Framework Status:
        • Integrated device count (22 devices)
        • Device risk levels and states
        • Framework version information

EXAMPLES:

    Basic discovery (no root required):
        ./dsmil-discover.sh

    Deep hardware scan with sudo:
        ./dsmil-discover.sh --sudo

    Save report to file:
        ./dsmil-discover.sh --sudo --output dsmil-report.txt

    View version:
        ./dsmil-discover.sh --version

SECURITY NOTES:

    • Use --sudo for complete MSR and MMIO region discovery
    • HAP bit detection requires root access to Intel ME interfaces
    • On MIL-SPEC systems, HAP bit should be SET (ME disabled)
    • The tool is read-only and performs no modifications

REQUIREMENTS:

    Required:
        • Python 3.x
        • DSMIL integration framework

    Optional (for enhanced discovery):
        • Root/sudo access
        • msr kernel module (modprobe msr)
        • rdmsr utility (msr-tools package)
        • intelmetool (for HAP bit detection)
        • lspci, lsusb, dmidecode

LOCATION:
    Discovery Script: ./02-tools/dsmil-devices/dsmil_discover.py
    Device Framework: ./02-tools/dsmil-devices/

CLASSIFICATION:
    UNCLASSIFIED // FOR OFFICIAL USE ONLY

EOF
    exit 0
fi

# Show version
if [ $SHOW_VERSION -eq 1 ]; then
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              DSMIL HARDWARE DISCOVERY TOOL - VERSION                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "  ${GREEN}Version:${NC}         $VERSION"
    echo -e "  ${GREEN}Release Date:${NC}    $DISCOVERY_DATE"
    echo -e "  ${GREEN}Framework:${NC}       DSMIL Integration Framework v1.5.0"
    echo -e "  ${GREEN}Devices:${NC}         22 integrated (20.4% of 108 DSMIL devices)"
    echo
    echo -e "  ${GREEN}Features:${NC}"
    echo -e "    • Deep hardware scanning (SMM/MSR/Intel ME)"
    echo -e "    • HAP bit detection for MIL-SPEC security"
    echo -e "    • Comprehensive DSMIL device discovery"
    echo -e "    • Security posture assessment"
    echo
    echo -e "  ${GREEN}Classification:${NC}  UNCLASSIFIED // FOR OFFICIAL USE ONLY"
    echo
    exit 0
fi

# Check if discovery script exists
if [ ! -f "$DISCOVERY_SCRIPT" ]; then
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                              ERROR                                           ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${RED}Error: Discovery script not found!${NC}"
    echo -e "Expected location: ${YELLOW}$DISCOVERY_SCRIPT${NC}"
    echo
    echo "Please ensure the DSMIL device framework is properly installed."
    echo "Run this script from the repository root directory."
    echo
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not found${NC}"
    exit 1
fi

# Display banner
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              DSMIL HARDWARE DISCOVERY TOOL                                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "  ${GREEN}Version:${NC}         $VERSION"
echo -e "  ${GREEN}Framework:${NC}       DSMIL Integration Framework v1.5.0"
echo -e "  ${GREEN}Devices:${NC}         22 integrated devices"
echo
echo -e "  ${GREEN}Scanning:${NC}"
echo -e "    • Basic hardware discovery"
echo -e "    • Deep SMM/MSR/Intel ME detection"
echo -e "    • Framework integration status"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "  ${GREEN}✓ Running as root - full hardware access enabled${NC}"
    echo
elif [ $USE_SUDO -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Elevating to root for deep hardware access...${NC}"
    echo
    exec sudo "$0" "$@"
else
    echo -e "  ${YELLOW}⚠ Not running as root - some checks will be limited${NC}"
    echo -e "  ${YELLOW}  For complete MSR/MMIO/ME access, run with: --sudo${NC}"
    echo
fi

# Run discovery
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Execute discovery script
if [ -n "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}Saving report to: ${YELLOW}$OUTPUT_FILE${NC}"
    echo
    python3 "$DISCOVERY_SCRIPT" > "$OUTPUT_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo
        echo -e "${GREEN}✓ Report saved successfully!${NC}"
        echo -e "  File: ${YELLOW}$OUTPUT_FILE${NC}"
        echo -e "  Size: $(wc -c < "$OUTPUT_FILE") bytes"
        echo

        # Show summary
        echo -e "${BLUE}Quick Summary:${NC}"
        grep -A 15 "DISCOVERY SUMMARY" "$OUTPUT_FILE" | head -20 || true
    else
        echo -e "${RED}✗ Discovery failed with exit code: $EXIT_CODE${NC}"
        exit $EXIT_CODE
    fi
else
    python3 "$DISCOVERY_SCRIPT"
    EXIT_CODE=$?
fi

# Footer
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Discovery completed successfully!${NC}"
    echo
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  • Review the security recommendations above"
    echo -e "  • Check Intel ME HAP bit status if running MIL-SPEC"
    echo -e "  • Run with --sudo for complete hardware detection"
    echo -e "  • Save report with: ${YELLOW}./dsmil-discover.sh --sudo --output report.txt${NC}"
    echo
else
    echo -e "${RED}✗ Discovery completed with errors (exit code: $EXIT_CODE)${NC}"
    echo
    exit $EXIT_CODE
fi

exit 0
