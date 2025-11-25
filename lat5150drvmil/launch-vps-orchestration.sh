#!/bin/bash
################################################################################
# VPS Orchestration Launcher
################################################################################
# Quick launcher for ASN/BGP-aware VPS infrastructure management
#
# Author: LAT5150DRVMIL AI Platform
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

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Banner
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              ASN/BGP-AWARE VPS ORCHESTRATION SYSTEM                       ║
║                                                                           ║
║              Worldwide Server Infrastructure Management                   ║
║                  LAT5150 MIL-SPEC AI Platform v2.0                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BOLD}Capabilities:${NC}"
echo "  • Multi-region VPS provisioning and management"
echo "  • BGP routing and Anycast network deployment"
echo "  • IPv6 subnet allocation and management"
echo "  • Geolocation verification and correction"
echo "  • WireGuard mesh networking"
echo "  • Automated firewall and security configuration"
echo ""
echo -e "${YELLOW}Inspired by: https://blog.lyc8503.net/en/post/asn-5-worldwide-servers/${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

# Check for required Python packages
echo -e "${BOLD}Checking prerequisites...${NC}"
python3 -c "import requests" 2>/dev/null || {
    echo -e "${YELLOW}Installing required Python packages...${NC}"
    pip3 install requests --quiet || {
        echo -e "${RED}ERROR: Failed to install requests package${NC}"
        exit 1
    }
}
echo -e "${GREEN}✓${NC} Prerequisites OK"
echo ""

# Menu
echo -e "${BOLD}Select mode:${NC}"
echo ""
echo "  ${BOLD}Python Manager (Recommended):${NC}"
echo "    1. Provision single VPS"
echo "    2. Deploy Anycast network"
echo "    3. Verify IP geolocation"
echo "    4. Generate Geofeed file"
echo "    5. Generate infrastructure report"
echo ""
echo "  ${BOLD}Automation Scripts:${NC}"
echo "    6. Interactive automation menu"
echo "    7. Batch deploy from config"
echo ""
echo "  ${BOLD}Quick Actions:${NC}"
echo "    8. View example configuration"
echo "    9. Test current IP geolocation"
echo ""
read -p "Enter choice (1-9): " choice
echo ""

case $choice in
    1)
        # Provision single VPS
        read -p "Hostname: " hostname
        echo "Available regions: us-west, us-east, eu-central, eu-west, singapore, japan"
        read -p "Region: " region
        read -p "Enable BGP? (y/n): " bgp
        read -p "Enable Anycast? (y/n): " anycast

        args="--provision $hostname --region $region"
        [[ "$bgp" == "y" ]] && args="$args --enable-bgp"
        [[ "$anycast" == "y" ]] && args="$args --enable-anycast"

        python3 02-ai-engine/vps_orchestration/asn_vps_manager.py $args
        ;;

    2)
        # Deploy Anycast network
        echo -e "${CYAN}Deploying Anycast network across US West, EU Central, and Singapore...${NC}"
        python3 02-ai-engine/vps_orchestration/asn_vps_manager.py --deploy-anycast
        ;;

    3)
        # Verify geolocation
        read -p "IP address to verify: " ip
        python3 02-ai-engine/vps_orchestration/asn_vps_manager.py --verify-geo "$ip"
        ;;

    4)
        # Generate Geofeed
        read -p "Output file [/tmp/geofeed.csv]: " output
        output=${output:-/tmp/geofeed.csv}
        python3 02-ai-engine/vps_orchestration/asn_vps_manager.py --generate-geofeed "$output"
        ;;

    5)
        # Generate report
        read -p "Output file [/tmp/vps_infrastructure_report.json]: " output
        output=${output:-/tmp/vps_infrastructure_report.json}
        python3 02-ai-engine/vps_orchestration/asn_vps_manager.py --report "$output"
        ;;

    6)
        # Interactive automation menu
        bash 02-ai-engine/vps_orchestration/vps_automation_scripts.sh
        ;;

    7)
        # Batch deploy
        read -p "Config file [02-ai-engine/vps_orchestration/vps_config_example.json]: " config
        config=${config:-02-ai-engine/vps_orchestration/vps_config_example.json}

        if [[ ! -f "$config" ]]; then
            echo -e "${RED}ERROR: Config file not found: $config${NC}"
            exit 1
        fi

        bash 02-ai-engine/vps_orchestration/vps_automation_scripts.sh batch_deploy_vps "$config"
        ;;

    8)
        # View example config
        echo -e "${CYAN}Example Configuration:${NC}"
        echo "File: 02-ai-engine/vps_orchestration/vps_config_example.json"
        echo ""
        cat 02-ai-engine/vps_orchestration/vps_config_example.json | jq '.' 2>/dev/null || cat 02-ai-engine/vps_orchestration/vps_config_example.json
        ;;

    9)
        # Test current IP
        echo -e "${CYAN}Testing current IP geolocation...${NC}"
        echo ""
        echo "1. Your IPv4:"
        curl -s https://ipinfo.io/json | jq '.' 2>/dev/null || curl -s https://ipinfo.io

        echo ""
        echo "2. Cloudflare Trace:"
        curl -s https://cloudflare.com/cdn-cgi/trace

        echo ""
        echo "3. IPv6 (if available):"
        curl -s -6 https://ipinfo.io/json 2>/dev/null | jq '.' || echo "  IPv6 not available"
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}${BOLD}Operation complete!${NC}"
echo ""
