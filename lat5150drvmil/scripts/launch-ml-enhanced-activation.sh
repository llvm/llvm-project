#!/bin/bash
################################################################################
# ML-Enhanced DSMIL Activation Quick Launcher
################################################################################
# Direct launcher for the ML-enhanced integrated activation system
# This provides a quick way to run the end-to-end workflow without tmux
#
# Author: LAT5150DRVMIL AI Platform
# Classification: DSMIL ML Integration
# Version: 1.0.0
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Banner
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              ML-ENHANCED DSMIL ACTIVATION SYSTEM                          ║
║                                                                           ║
║              Dell System Military Integration Layer (DSMIL)               ║
║                  LAT5150 MIL-SPEC AI Platform v2.0                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BOLD}MISSION-CRITICAL: ML-Enhanced Hardware Discovery${NC}"
echo ""
echo -e "${GREEN}Features:${NC}"
echo "  • Automatic hardware discovery (SMBIOS, ACPI, sysfs)"
echo "  • Machine learning device classification"
echo "  • Intelligent activation sequencing"
echo "  • Real-time safety monitoring"
echo "  • Thermal impact prediction"
echo ""

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo -e "${YELLOW}WARNING: Not running as root. Some operations may fail.${NC}"
    echo -e "${YELLOW}Recommend running with: sudo $0${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for required files
REQUIRED_FILES=(
    "02-ai-engine/dsmil_ml_discovery.py"
    "02-ai-engine/dsmil_integrated_activation.py"
    "02-ai-engine/dsmil_device_activation.py"
    "02-ai-engine/dsmil_subsystem_controller.py"
)

echo -e "${BOLD}Checking prerequisites...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}✗ Required file missing: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All required files present"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} python3 installed"

# Menu
echo ""
echo -e "${BOLD}Select mode:${NC}"
echo "  1. Full ML-Enhanced Workflow (Discovery + Activation + Monitoring)"
echo "  2. Discovery Only (No activation)"
echo "  3. Interactive Mode (Confirm each device)"
echo "  4. Run with custom monitoring duration"
echo ""
read -p "Enter choice (1-4): " choice
echo ""

case $choice in
    1)
        echo -e "${CYAN}Launching full ML-enhanced workflow...${NC}"
        python3 "02-ai-engine/dsmil_integrated_activation.py"
        ;;
    2)
        echo -e "${CYAN}Launching discovery-only mode...${NC}"
        python3 "02-ai-engine/dsmil_integrated_activation.py" --no-activation
        ;;
    3)
        echo -e "${CYAN}Launching interactive mode...${NC}"
        python3 "02-ai-engine/dsmil_integrated_activation.py" --interactive
        ;;
    4)
        read -p "Enter monitoring duration in seconds (default 30): " duration
        duration=${duration:-30}
        echo -e "${CYAN}Launching with ${duration}s monitoring...${NC}"
        python3 "02-ai-engine/dsmil_integrated_activation.py" --monitor-duration "$duration"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}${BOLD}Workflow complete!${NC}"
echo ""
echo -e "Logs available at:"
echo "  • /tmp/dsmil_integrated_activation.log"
echo "  • /tmp/dsmil_ml_discovery.log"
echo "  • /tmp/dsmil_integrated_workflow_report.json"
echo ""
