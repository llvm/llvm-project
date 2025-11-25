#!/bin/bash
#
# Xen VM Bridge Configuration for LAT5150 DRVMIL Tactical Interface
# Configures secure network access from Xen VMs to tactical interface
#
# Usage:
#   sudo ./configure_xen_bridge.sh install
#   sudo ./configure_xen_bridge.sh remove
#   sudo ./configure_xen_bridge.sh status
#

set -e

INTERFACE_PORT=${TACTICAL_PORT:-5001}
XEN_BRIDGE=${XEN_BRIDGE:-xenbr0}
ALLOWED_VM_NETWORK=${ALLOWED_VM_NETWORK:-"192.168.100.0/24"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${CYAN}[====] $1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_xen() {
    if ! command -v xl &> /dev/null; then
        log_error "Xen tools not found. Is Xen installed?"
        log_warn "Install with: apt-get install xen-tools (Debian/Ubuntu)"
        log_warn "          or: yum install xen (RHEL/CentOS)"
        exit 1
    fi
}

check_bridge() {
    if ! ip link show "$XEN_BRIDGE" &> /dev/null; then
        log_error "Xen bridge '$XEN_BRIDGE' not found"
        log_warn "Create bridge first with: xl network-attach or brctl"
        exit 1
    fi
}

get_bridge_ip() {
    ip addr show "$XEN_BRIDGE" | grep "inet " | awk '{print $2}' | cut -d/ -f1
}

configure_nginx_bridge() {
    log_section "Configuring Nginx for Xen Bridge Access"

    BRIDGE_IP=$(get_bridge_ip)

    if [ -z "$BRIDGE_IP" ]; then
        log_error "Could not determine bridge IP address"
        exit 1
    fi

    log_info "Bridge IP: $BRIDGE_IP"
    log_info "Creating Nginx configuration..."

    cat > /etc/nginx/sites-available/tactical-xen-bridge <<EOF
# LAT5150 DRVMIL Tactical Interface - Xen Bridge Access
# Provides secure access from Xen VMs to tactical interface
#
# Security: Only accessible from Xen bridge network
# TEMPEST: Maintains localhost-only on host, bridge-only for VMs

upstream tactical_backend {
    server 127.0.0.1:${INTERFACE_PORT};
    keepalive 32;
}

# Xen Bridge Access (VMs only)
server {
    listen ${BRIDGE_IP}:8443 ssl http2;
    server_name tactical.xen.local;

    # SSL Configuration (self-signed for internal use)
    ssl_certificate /etc/nginx/ssl/tactical-xen.crt;
    ssl_certificate_key /etc/nginx/ssl/tactical-xen.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # TEMPEST: Disable server tokens
    server_tokens off;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # IP Whitelist - Only Xen VM network
    allow ${ALLOWED_VM_NETWORK};
    deny all;

    # Logging
    access_log /var/log/nginx/tactical-xen-access.log;
    error_log /var/log/nginx/tactical-xen-error.log warn;

    # Root location
    location / {
        root /var/www/tactical;
        index tactical_self_coding_ui.html;
        try_files \$uri \$uri/ =404;

        # Security: Disable directory listing
        autoindex off;
    }

    # API Proxy
    location /api/ {
        proxy_pass http://tactical_backend;
        proxy_http_version 1.1;

        # WebSocket Support
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";

        # Headers
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;
    }

    # WebSocket endpoint
    location /ws/ {
        proxy_pass http://tactical_backend;
        proxy_http_version 1.1;

        # WebSocket headers
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;

        # Timeouts for long-lived connections
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }
}
EOF

    log_info "Nginx configuration created"
}

generate_ssl_certs() {
    log_section "Generating Self-Signed SSL Certificates"

    mkdir -p /etc/nginx/ssl

    if [ -f /etc/nginx/ssl/tactical-xen.crt ]; then
        log_warn "SSL certificates already exist, skipping..."
        return
    fi

    log_info "Generating self-signed certificate..."

    openssl req -x509 -nodes -days 3650 -newkey rsa:4096 \
        -keyout /etc/nginx/ssl/tactical-xen.key \
        -out /etc/nginx/ssl/tactical-xen.crt \
        -subj "/C=US/ST=Secure/L=Tactical/O=LAT5150/OU=DRVMIL/CN=tactical.xen.local" \
        2>/dev/null

    chmod 600 /etc/nginx/ssl/tactical-xen.key
    chmod 644 /etc/nginx/ssl/tactical-xen.crt

    log_info "SSL certificates generated"
}

setup_www_dir() {
    log_section "Setting up Web Directory"

    mkdir -p /var/www/tactical

    if [ ! -f /var/www/tactical/tactical_self_coding_ui.html ]; then
        log_info "Copying tactical interface..."

        # Find the tactical UI file
        UI_FILE=$(find /home/user/LAT5150DRVMIL -name "tactical_self_coding_ui.html" 2>/dev/null | head -1)

        if [ -n "$UI_FILE" ]; then
            cp "$UI_FILE" /var/www/tactical/
            log_info "Tactical interface copied"
        else
            log_warn "Tactical interface not found, please copy manually to /var/www/tactical/"
        fi
    fi

    chown -R www-data:www-data /var/www/tactical
    chmod -R 755 /var/www/tactical
}

configure_iptables() {
    log_section "Configuring iptables for Xen Bridge"

    BRIDGE_IP=$(get_bridge_ip)

    # Allow connections from VM network to bridge IP
    log_info "Adding iptables rules..."

    iptables -A INPUT -i "$XEN_BRIDGE" -s "$ALLOWED_VM_NETWORK" -p tcp --dport 8443 -j ACCEPT \
        -m comment --comment "tactical-xen-bridge"

    # Block everything else to port 8443
    iptables -A INPUT -p tcp --dport 8443 -j DROP \
        -m comment --comment "tactical-xen-block-external"

    # Save rules
    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
    elif [ -f /etc/init.d/iptables ]; then
        /etc/init.d/iptables save
    fi

    log_info "iptables rules configured"
}

install_config() {
    check_root
    check_xen
    check_bridge

    log_section "Installing Xen Bridge Configuration"
    echo ""
    log_info "Bridge: $XEN_BRIDGE"
    log_info "Port: 8443 (HTTPS)"
    log_info "Allowed Network: $ALLOWED_VM_NETWORK"
    echo ""

    # Generate SSL certificates
    generate_ssl_certs

    # Setup web directory
    setup_www_dir

    # Configure Nginx
    if command -v nginx &> /dev/null; then
        configure_nginx_bridge

        # Enable site
        ln -sf /etc/nginx/sites-available/tactical-xen-bridge \
               /etc/nginx/sites-enabled/tactical-xen-bridge

        # Test configuration
        nginx -t

        # Reload Nginx
        systemctl reload nginx

        log_info "Nginx configured and reloaded"
    else
        log_warn "Nginx not found. Please install Nginx or configure your web server manually."
        log_warn "Configuration saved to: /etc/nginx/sites-available/tactical-xen-bridge"
    fi

    # Configure firewall
    configure_iptables

    echo ""
    log_section "Installation Complete"
    echo ""
    log_info "✓ Xen bridge access configured"
    log_info "✓ SSL certificates generated"
    log_info "✓ Firewall rules applied"
    echo ""
    BRIDGE_IP=$(get_bridge_ip)
    log_info "VMs can access tactical interface at:"
    log_info "  https://${BRIDGE_IP}:8443/"
    log_info "  https://tactical.xen.local:8443/ (if DNS configured)"
    echo ""
    log_warn "⚠️  Add to VM /etc/hosts:"
    log_warn "    ${BRIDGE_IP}  tactical.xen.local"
    echo ""
}

remove_config() {
    check_root

    log_section "Removing Xen Bridge Configuration"

    # Remove Nginx config
    rm -f /etc/nginx/sites-enabled/tactical-xen-bridge
    rm -f /etc/nginx/sites-available/tactical-xen-bridge

    if command -v nginx &> /dev/null; then
        systemctl reload nginx
    fi

    # Remove iptables rules
    iptables -D INPUT -i "$XEN_BRIDGE" -s "$ALLOWED_VM_NETWORK" -p tcp --dport 8443 -j ACCEPT \
        -m comment --comment "tactical-xen-bridge" 2>/dev/null || true
    iptables -D INPUT -p tcp --dport 8443 -j DROP \
        -m comment --comment "tactical-xen-block-external" 2>/dev/null || true

    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
    fi

    log_info "Configuration removed"
}

show_status() {
    log_section "Xen Bridge Configuration Status"
    echo ""

    # Check Xen
    if command -v xl &> /dev/null; then
        log_info "✓ Xen tools installed"
    else
        log_error "✗ Xen tools not found"
    fi

    # Check bridge
    if ip link show "$XEN_BRIDGE" &> /dev/null; then
        BRIDGE_IP=$(get_bridge_ip)
        log_info "✓ Bridge exists: $XEN_BRIDGE ($BRIDGE_IP)"
    else
        log_error "✗ Bridge not found: $XEN_BRIDGE"
    fi

    # Check Nginx
    if [ -f /etc/nginx/sites-enabled/tactical-xen-bridge ]; then
        log_info "✓ Nginx configuration active"

        if systemctl is-active --quiet nginx; then
            log_info "✓ Nginx running"
        else
            log_warn "⚠ Nginx not running"
        fi
    else
        log_warn "⚠ Nginx configuration not found"
    fi

    # Check SSL certs
    if [ -f /etc/nginx/ssl/tactical-xen.crt ]; then
        log_info "✓ SSL certificates exist"
    else
        log_warn "⚠ SSL certificates not found"
    fi

    # Check iptables
    if iptables -L INPUT -n | grep -q "tactical-xen"; then
        log_info "✓ iptables rules configured"
    else
        log_warn "⚠ iptables rules not found"
    fi

    # Check running VMs
    echo ""
    log_info "Running Xen VMs:"
    xl list | tail -n +2 || log_warn "No VMs running"

    echo ""
}

show_help() {
    cat <<EOF
Xen Bridge Configuration for LAT5150 DRVMIL Tactical Interface

Usage:
    $0 <command> [options]

Commands:
    install     Configure Xen bridge access for VMs
    remove      Remove Xen bridge configuration
    status      Show current configuration status
    help        Show this help message

Environment Variables:
    TACTICAL_PORT         Backend port (default: 5001)
    XEN_BRIDGE           Bridge interface (default: xenbr0)
    ALLOWED_VM_NETWORK   Allowed VM subnet (default: 192.168.100.0/24)

Examples:
    # Install with defaults
    sudo $0 install

    # Install with custom network
    sudo ALLOWED_VM_NETWORK="10.0.0.0/24" $0 install

    # Check status
    sudo $0 status

    # Remove configuration
    sudo $0 remove

Description:
    Configures secure HTTPS access to the tactical interface from Xen VMs
    via the Xen bridge network. Maintains localhost-only security on host
    while providing encrypted bridge access for VMs.

Security Features:
    - HTTPS with self-signed certificates
    - IP whitelist (VM network only)
    - Firewall rules (iptables)
    - Nginx reverse proxy
    - Security headers
    - TEMPEST-compliant configuration

Access from VMs:
    https://<bridge-ip>:8443/

Requirements:
    - Xen hypervisor installed
    - Xen bridge configured
    - Nginx web server
    - Root privileges

EOF
}

# Main
case "${1:-help}" in
    install)
        install_config
        ;;
    remove)
        remove_config
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
