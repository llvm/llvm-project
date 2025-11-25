#!/bin/bash
################################################################################
# VPS Automation Scripts for ASN/BGP Infrastructure
################################################################################
# Collection of automation scripts for managing worldwide VPS infrastructure
# with BGP, Anycast, and geolocation capabilities.
#
# Inspired by: https://blog.lyc8503.net/en/post/asn-5-worldwide-servers/
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

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

################################################################################
# BIRD BGP Configuration Deployment
################################################################################

deploy_bird_config() {
    local hostname=$1
    local remote_user=$2
    local remote_host=$3

    log_info "Deploying BIRD configuration to $hostname ($remote_host)"

    local config_file="/tmp/${hostname}_bird.conf"

    if [[ ! -f "$config_file" ]]; then
        log_error "BIRD config not found: $config_file"
        return 1
    fi

    # Deploy configuration
    scp "$config_file" "${remote_user}@${remote_host}:/tmp/bird.conf" || return 1

    # Install and configure BIRD
    ssh "${remote_user}@${remote_host}" << 'REMOTE_SCRIPT'
        # Install BIRD if not present
        if ! command -v bird &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y bird2
        fi

        # Backup existing config
        sudo cp /etc/bird/bird.conf /etc/bird/bird.conf.backup.$(date +%s) 2>/dev/null || true

        # Deploy new config
        sudo mv /tmp/bird.conf /etc/bird/bird.conf
        sudo chown root:root /etc/bird/bird.conf
        sudo chmod 644 /etc/bird/bird.conf

        # Test configuration
        sudo bird -p -c /etc/bird/bird.conf

        # Reload BIRD
        sudo systemctl enable bird
        sudo systemctl restart bird

        echo "BIRD configuration deployed and reloaded"
REMOTE_SCRIPT

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "BIRD deployed successfully to $hostname"
    else
        log_error "BIRD deployment failed on $hostname"
    fi

    return $exit_code
}

################################################################################
# WireGuard Mesh Network Setup
################################################################################

setup_wireguard_mesh() {
    local hostname=$1
    local remote_user=$2
    local remote_host=$3

    log_info "Setting up WireGuard on $hostname ($remote_host)"

    local config_file="/tmp/${hostname}_wg0.conf"

    if [[ ! -f "$config_file" ]]; then
        log_error "WireGuard config not found: $config_file"
        return 1
    fi

    # Generate WireGuard keys
    local private_key=$(wg genkey)
    local public_key=$(echo "$private_key" | wg pubkey)

    log_info "Generated WireGuard keys for $hostname"
    echo "Public key: $public_key"

    # Update config with actual private key
    sed -i "s/<GENERATE_PRIVATE_KEY>/$private_key/g" "$config_file"

    # Deploy configuration
    scp "$config_file" "${remote_user}@${remote_host}:/tmp/wg0.conf" || return 1

    ssh "${remote_user}@${remote_host}" << 'REMOTE_SCRIPT'
        # Install WireGuard if not present
        if ! command -v wg &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y wireguard
        fi

        # Deploy configuration
        sudo mv /tmp/wg0.conf /etc/wireguard/wg0.conf
        sudo chown root:root /etc/wireguard/wg0.conf
        sudo chmod 600 /etc/wireguard/wg0.conf

        # Enable IP forwarding
        sudo sysctl -w net.ipv4.ip_forward=1
        sudo sysctl -w net.ipv6.conf.all.forwarding=1

        # Make permanent
        echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.conf

        # Start WireGuard
        sudo systemctl enable wg-quick@wg0
        sudo systemctl restart wg-quick@wg0

        echo "WireGuard mesh configured and started"
REMOTE_SCRIPT

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "WireGuard deployed successfully to $hostname"
        echo "$public_key" > "/tmp/${hostname}_wg_pubkey.txt"
    else
        log_error "WireGuard deployment failed on $hostname"
    fi

    return $exit_code
}

################################################################################
# Firewall Anti-Scan Configuration
################################################################################

configure_anti_scan_firewall() {
    local remote_user=$1
    local remote_host=$2

    log_info "Configuring anti-scan firewall on $remote_host"

    ssh "${remote_user}@${remote_host}" << 'REMOTE_SCRIPT'
        # Install iptables if needed
        if ! command -v iptables &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y iptables iptables-persistent
        fi

        # Backup current rules
        sudo iptables-save > /tmp/iptables.backup.$(date +%s)

        # Flush existing rules
        sudo iptables -F
        sudo iptables -X

        # Default policies
        sudo iptables -P INPUT DROP
        sudo iptables -P FORWARD DROP
        sudo iptables -P OUTPUT ACCEPT

        # Allow loopback
        sudo iptables -A INPUT -i lo -j ACCEPT

        # Allow established connections
        sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

        # Allow SSH (with rate limiting)
        sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
        sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
        sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

        # Allow WireGuard
        sudo iptables -A INPUT -p udp --dport 51820 -j ACCEPT

        # Allow BGP (if needed)
        sudo iptables -A INPUT -p tcp --dport 179 -j ACCEPT

        # Block ICMP ping (prevents geolocation scanning)
        sudo iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

        # Rate limit new connections
        sudo iptables -A INPUT -p tcp -m state --state NEW -m recent --set
        sudo iptables -A INPUT -p tcp -m state --state NEW -m recent --update --seconds 60 --hitcount 10 -j DROP

        # Save rules
        sudo netfilter-persistent save

        echo "Anti-scan firewall configured"
REMOTE_SCRIPT

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "Firewall configured successfully"
    else
        log_error "Firewall configuration failed"
    fi

    return $exit_code
}

################################################################################
# IPv6 Routing Configuration
################################################################################

configure_ipv6_routing() {
    local remote_user=$1
    local remote_host=$2
    local ipv6_subnet=$3

    log_info "Configuring IPv6 routing on $remote_host for $ipv6_subnet"

    ssh "${remote_user}@${remote_host}" << REMOTE_SCRIPT
        # Enable IPv6 forwarding
        sudo sysctl -w net.ipv6.conf.all.forwarding=1
        sudo sysctl -w net.ipv6.conf.default.forwarding=1

        # Accept router advertisements (for upstream)
        sudo sysctl -w net.ipv6.conf.all.accept_ra=2
        sudo sysctl -w net.ipv6.conf.default.accept_ra=2

        # Make permanent
        echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv6.conf.default.forwarding=1" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv6.conf.all.accept_ra=2" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv6.conf.default.accept_ra=2" | sudo tee -a /etc/sysctl.conf

        # Add static route for subnet
        ip -6 route add $ipv6_subnet dev lo || true

        echo "IPv6 routing configured for $ipv6_subnet"
REMOTE_SCRIPT

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "IPv6 routing configured successfully"
    else
        log_error "IPv6 routing configuration failed"
    fi

    return $exit_code
}

################################################################################
# Geolocation Verification
################################################################################

verify_geolocation() {
    local ip_address=$1

    log_info "Verifying geolocation for $ip_address"

    echo ""
    echo "Checking multiple geolocation databases:"
    echo "=========================================="

    # IPInfo.io
    echo ""
    echo "1. IPInfo.io:"
    curl -s "https://ipinfo.io/$ip_address/json" | jq -r '"Country: \(.country), Region: \(.region), City: \(.city), Org: \(.org)"' || echo "Failed to query"

    # IP-API.com
    echo ""
    echo "2. IP-API.com:"
    curl -s "http://ip-api.com/json/$ip_address" | jq -r '"Country: \(.countryCode), Region: \(.regionName), City: \(.city), ISP: \(.isp)"' || echo "Failed to query"

    # Cloudflare trace
    echo ""
    echo "3. Cloudflare Trace:"
    if [[ "$ip_address" == "me" ]] || [[ "$ip_address" == "self" ]]; then
        curl -s "https://cloudflare.com/cdn-cgi/trace" | grep -E "^(ip|loc|colo)=" || echo "Failed to query"
    else
        echo "  (Only works for client IP)"
    fi

    echo ""
}

################################################################################
# Batch VPS Deployment
################################################################################

batch_deploy_vps() {
    local config_file=$1

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    log_info "Deploying VPS servers from config: $config_file"

    # Read JSON config
    local servers=$(jq -c '.servers[]' "$config_file")

    while IFS= read -r server; do
        local hostname=$(echo "$server" | jq -r '.hostname')
        local region=$(echo "$server" | jq -r '.region')
        local user=$(echo "$server" | jq -r '.user')
        local host=$(echo "$server" | jq -r '.host')
        local ipv6_subnet=$(echo "$server" | jq -r '.ipv6_subnet')

        log_info "Deploying $hostname in $region"

        # Configure IPv6 routing
        configure_ipv6_routing "$user" "$host" "$ipv6_subnet"

        # Deploy BIRD if BGP enabled
        if echo "$server" | jq -e '.bgp_enabled' > /dev/null; then
            deploy_bird_config "$hostname" "$user" "$host"
        fi

        # Setup WireGuard mesh
        setup_wireguard_mesh "$hostname" "$user" "$host"

        # Configure anti-scan firewall
        configure_anti_scan_firewall "$user" "$host"

        log_success "$hostname deployed successfully"
        echo ""

    done <<< "$servers"

    log_success "Batch deployment complete"
}

################################################################################
# Generate Deployment Report
################################################################################

generate_deployment_report() {
    local output_file=${1:-"/tmp/vps_deployment_report.txt"}

    log_info "Generating deployment report..."

    cat > "$output_file" << EOF
VPS Infrastructure Deployment Report
====================================
Generated: $(date)

Public Keys for WireGuard Peers:
--------------------------------
EOF

    # Add all public keys
    for pubkey_file in /tmp/*_wg_pubkey.txt; do
        if [[ -f "$pubkey_file" ]]; then
            local hostname=$(basename "$pubkey_file" | sed 's/_wg_pubkey.txt//')
            local pubkey=$(cat "$pubkey_file")
            echo "$hostname: $pubkey" >> "$output_file"
        fi
    done

    log_success "Report generated: $output_file"
    cat "$output_file"
}

################################################################################
# Main Menu
################################################################################

show_menu() {
    cat << EOF

${CYAN}${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}
${CYAN}${BOLD}║          VPS Infrastructure Automation Scripts                            ║${NC}
${CYAN}${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}

${BOLD}Select operation:${NC}

  1. Deploy BIRD BGP configuration
  2. Setup WireGuard mesh network
  3. Configure anti-scan firewall
  4. Configure IPv6 routing
  5. Verify IP geolocation
  6. Batch deploy from config file
  7. Generate deployment report
  8. Exit

EOF
}

main() {
    while true; do
        show_menu
        read -p "Enter choice (1-8): " choice
        echo ""

        case $choice in
            1)
                read -p "Hostname: " hostname
                read -p "Remote user: " remote_user
                read -p "Remote host: " remote_host
                deploy_bird_config "$hostname" "$remote_user" "$remote_host"
                ;;
            2)
                read -p "Hostname: " hostname
                read -p "Remote user: " remote_user
                read -p "Remote host: " remote_host
                setup_wireguard_mesh "$hostname" "$remote_user" "$remote_host"
                ;;
            3)
                read -p "Remote user: " remote_user
                read -p "Remote host: " remote_host
                configure_anti_scan_firewall "$remote_user" "$remote_host"
                ;;
            4)
                read -p "Remote user: " remote_user
                read -p "Remote host: " remote_host
                read -p "IPv6 subnet (e.g., 2a14:7c0:4d00::/48): " ipv6_subnet
                configure_ipv6_routing "$remote_user" "$remote_host" "$ipv6_subnet"
                ;;
            5)
                read -p "IP address to verify: " ip_address
                verify_geolocation "$ip_address"
                ;;
            6)
                read -p "Config file path: " config_file
                batch_deploy_vps "$config_file"
                ;;
            7)
                read -p "Output file [/tmp/vps_deployment_report.txt]: " output_file
                output_file=${output_file:-"/tmp/vps_deployment_report.txt"}
                generate_deployment_report "$output_file"
                ;;
            8)
                log_info "Exiting..."
                exit 0
                ;;
            *)
                log_error "Invalid choice"
                ;;
        esac

        echo ""
        read -p "Press ENTER to continue..." dummy
    done
}

# Run main menu if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
