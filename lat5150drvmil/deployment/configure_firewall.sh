#!/bin/bash
#
# Firewall Configuration Script for Self-Coding System
# Provides additional network-level protection for localhost-only deployment
#
# Usage:
#   sudo ./configure_firewall.sh install
#   sudo ./configure_firewall.sh remove
#   sudo ./configure_firewall.sh status
#

set -e

PORT=${SELF_CODING_PORT:-5001}
SERVICE_NAME="selfcoding"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

detect_firewall() {
    if command -v ufw &> /dev/null; then
        echo "ufw"
    elif command -v firewall-cmd &> /dev/null; then
        echo "firewalld"
    elif command -v iptables &> /dev/null; then
        echo "iptables"
    else
        echo "none"
    fi
}

configure_ufw() {
    log_info "Configuring UFW firewall..."

    # Enable UFW if not already enabled
    if ! ufw status | grep -q "Status: active"; then
        log_warn "UFW is not active. Enabling..."
        ufw --force enable
    fi

    # Deny all incoming connections to port
    log_info "Denying all external connections to port $PORT..."
    ufw deny $PORT/tcp comment "Block external access to self-coding API"

    # Allow localhost connections only
    log_info "Allowing localhost connections to port $PORT..."
    ufw allow from 127.0.0.1 to any port $PORT proto tcp comment "Allow localhost self-coding API"
    ufw allow from ::1 to any port $PORT proto tcp comment "Allow localhost self-coding API (IPv6)"

    # Reload UFW
    ufw reload

    log_info "UFW configuration complete"
}

configure_firewalld() {
    log_info "Configuring firewalld..."

    # Check if firewalld is running
    if ! systemctl is-active --quiet firewalld; then
        log_warn "firewalld is not running. Starting..."
        systemctl start firewalld
        systemctl enable firewalld
    fi

    # Create rich rules for localhost-only access
    log_info "Adding localhost-only rules for port $PORT..."

    # Allow from localhost
    firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='127.0.0.1' port port='$PORT' protocol='tcp' accept"
    firewall-cmd --permanent --add-rich-rule="rule family='ipv6' source address='::1' port port='$PORT' protocol='tcp' accept"

    # Deny from everywhere else
    firewall-cmd --permanent --add-rich-rule="rule family='ipv4' port port='$PORT' protocol='tcp' reject"
    firewall-cmd --permanent --add-rich-rule="rule family='ipv6' port port='$PORT' protocol='tcp' reject"

    # Reload firewalld
    firewall-cmd --reload

    log_info "firewalld configuration complete"
}

configure_iptables() {
    log_info "Configuring iptables..."

    # Check if iptables-persistent is installed (for persistence)
    if ! command -v iptables-save &> /dev/null; then
        log_warn "iptables-persistent not found. Rules will not persist after reboot."
        log_warn "Install with: apt-get install iptables-persistent (Debian/Ubuntu)"
        log_warn "            or: yum install iptables-services (RHEL/CentOS)"
    fi

    # Allow localhost (IPv4)
    log_info "Adding IPv4 localhost rule..."
    iptables -A INPUT -p tcp --dport $PORT -s 127.0.0.1 -j ACCEPT -m comment --comment "self-coding-localhost"

    # Allow localhost (IPv6)
    log_info "Adding IPv6 localhost rule..."
    if command -v ip6tables &> /dev/null; then
        ip6tables -A INPUT -p tcp --dport $PORT -s ::1 -j ACCEPT -m comment --comment "self-coding-localhost"
    fi

    # Drop all other connections to port
    log_info "Blocking external connections to port $PORT..."
    iptables -A INPUT -p tcp --dport $PORT -j DROP -m comment --comment "self-coding-block-external"

    if command -v ip6tables &> /dev/null; then
        ip6tables -A INPUT -p tcp --dport $PORT -j DROP -m comment --comment "self-coding-block-external"
    fi

    # Save rules if iptables-persistent is available
    if command -v netfilter-persistent &> /dev/null; then
        log_info "Saving iptables rules..."
        netfilter-persistent save
    elif [ -f /etc/init.d/iptables ]; then
        log_info "Saving iptables rules..."
        /etc/init.d/iptables save
    fi

    log_info "iptables configuration complete"
}

remove_ufw() {
    log_info "Removing UFW rules for port $PORT..."

    # Remove rules
    ufw delete allow from 127.0.0.1 to any port $PORT proto tcp 2>/dev/null || true
    ufw delete allow from ::1 to any port $PORT proto tcp 2>/dev/null || true
    ufw delete deny $PORT/tcp 2>/dev/null || true

    ufw reload

    log_info "UFW rules removed"
}

remove_firewalld() {
    log_info "Removing firewalld rules for port $PORT..."

    # Remove rich rules
    firewall-cmd --permanent --remove-rich-rule="rule family='ipv4' source address='127.0.0.1' port port='$PORT' protocol='tcp' accept" 2>/dev/null || true
    firewall-cmd --permanent --remove-rich-rule="rule family='ipv6' source address='::1' port port='$PORT' protocol='tcp' accept" 2>/dev/null || true
    firewall-cmd --permanent --remove-rich-rule="rule family='ipv4' port port='$PORT' protocol='tcp' reject" 2>/dev/null || true
    firewall-cmd --permanent --remove-rich-rule="rule family='ipv6' port port='$PORT' protocol='tcp' reject" 2>/dev/null || true

    firewall-cmd --reload

    log_info "firewalld rules removed"
}

remove_iptables() {
    log_info "Removing iptables rules for port $PORT..."

    # Remove rules by comment
    iptables -D INPUT -p tcp --dport $PORT -s 127.0.0.1 -j ACCEPT -m comment --comment "self-coding-localhost" 2>/dev/null || true
    iptables -D INPUT -p tcp --dport $PORT -j DROP -m comment --comment "self-coding-block-external" 2>/dev/null || true

    if command -v ip6tables &> /dev/null; then
        ip6tables -D INPUT -p tcp --dport $PORT -s ::1 -j ACCEPT -m comment --comment "self-coding-localhost" 2>/dev/null || true
        ip6tables -D INPUT -p tcp --dport $PORT -j DROP -m comment --comment "self-coding-block-external" 2>/dev/null || true
    fi

    # Save if possible
    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
    elif [ -f /etc/init.d/iptables ]; then
        /etc/init.d/iptables save
    fi

    log_info "iptables rules removed"
}

show_status() {
    FIREWALL=$(detect_firewall)

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Firewall Status for Self-Coding System"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Port: $PORT"
    echo "Detected Firewall: $FIREWALL"
    echo ""

    case $FIREWALL in
        ufw)
            echo "UFW Status:"
            ufw status numbered | grep -E "(Status:|$PORT)" || echo "No rules found for port $PORT"
            ;;
        firewalld)
            echo "firewalld Status:"
            echo "Active: $(systemctl is-active firewalld)"
            echo ""
            echo "Rules for port $PORT:"
            firewall-cmd --list-rich-rules | grep "$PORT" || echo "No rules found for port $PORT"
            ;;
        iptables)
            echo "iptables Status:"
            echo ""
            echo "IPv4 Rules:"
            iptables -L INPUT -n --line-numbers | grep -E "(Chain|$PORT)" || echo "No rules found for port $PORT"
            echo ""
            if command -v ip6tables &> /dev/null; then
                echo "IPv6 Rules:"
                ip6tables -L INPUT -n --line-numbers | grep -E "(Chain|$PORT)" || echo "No rules found for port $PORT"
            fi
            ;;
        none)
            log_warn "No firewall detected"
            ;;
    esac

    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

install_rules() {
    check_root

    FIREWALL=$(detect_firewall)

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Installing Firewall Rules for Self-Coding System"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    log_info "Detected firewall: $FIREWALL"
    log_info "Port: $PORT"
    echo ""

    case $FIREWALL in
        ufw)
            configure_ufw
            ;;
        firewalld)
            configure_firewalld
            ;;
        iptables)
            configure_iptables
            ;;
        none)
            log_error "No firewall detected. Please install ufw, firewalld, or iptables."
            exit 1
            ;;
    esac

    echo ""
    log_info "✅ Firewall configuration complete"
    echo ""
    log_info "Rules installed:"
    log_info "  ✓ Allow connections from 127.0.0.1 to port $PORT"
    log_info "  ✓ Allow connections from ::1 to port $PORT"
    log_info "  ✓ Block all other connections to port $PORT"
    echo ""
    log_warn "⚠️  This provides network-level protection in addition to"
    log_warn "    the application-level security hardening."
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

remove_rules() {
    check_root

    FIREWALL=$(detect_firewall)

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Removing Firewall Rules for Self-Coding System"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    log_info "Detected firewall: $FIREWALL"
    log_info "Port: $PORT"
    echo ""

    case $FIREWALL in
        ufw)
            remove_ufw
            ;;
        firewalld)
            remove_firewalld
            ;;
        iptables)
            remove_iptables
            ;;
        none)
            log_warn "No firewall detected"
            ;;
    esac

    echo ""
    log_info "✅ Firewall rules removed"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

show_help() {
    cat <<EOF
Firewall Configuration Script for Self-Coding System

Usage:
    $0 <command> [options]

Commands:
    install     Install firewall rules for localhost-only access
    remove      Remove firewall rules
    status      Show current firewall status
    help        Show this help message

Options:
    SELF_CODING_PORT=<port>    Set custom port (default: 5001)

Examples:
    # Install rules with default port (5001)
    sudo $0 install

    # Install rules with custom port
    sudo SELF_CODING_PORT=8080 $0 install

    # Check status
    sudo $0 status

    # Remove rules
    sudo $0 remove

Description:
    This script configures firewall rules to enforce localhost-only access
    to the self-coding system API. It automatically detects and configures
    ufw, firewalld, or iptables.

    The rules ensure that only local connections (127.0.0.1, ::1) can
    access the API port, blocking all external network access.

Security Note:
    This provides network-level protection in addition to the application-level
    security hardening. The self-coding system already enforces localhost-only
    access at the application level, but firewall rules provide defense-in-depth.

Supported Firewalls:
    - UFW (Uncomplicated Firewall) - Ubuntu/Debian
    - firewalld - RHEL/CentOS/Fedora
    - iptables - Universal Linux

EOF
}

# Main script logic
case "${1:-help}" in
    install)
        install_rules
        ;;
    remove)
        remove_rules
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
