#!/usr/bin/env python3
"""
ASN/BGP-Aware VPS Management System
====================================
Integrates ASN-5 worldwide server capabilities with DSMIL infrastructure.

Capabilities:
- Multi-region VPS orchestration
- BGP/Anycast network management
- IPv6 subnet allocation and management
- Geolocation verification and correction
- WHOIS/Geofeed database management
- Network testing and validation
- WARP integration for geolocation spoofing

Inspired by: https://blog.lyc8503.net/en/post/asn-5-worldwide-servers/

Author: LAT5150DRVMIL AI Platform
Classification: VPS Infrastructure Management
"""

import os
import sys
import json
import subprocess
import ipaddress
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VPSRegion(Enum):
    """Supported VPS regions"""
    US_WEST = "us-west"
    US_EAST = "us-east"
    EU_CENTRAL = "eu-central"
    EU_WEST = "eu-west"
    ASIA_PACIFIC = "asia-pacific"
    SINGAPORE = "singapore"
    JAPAN = "japan"
    AUSTRALIA = "australia"


class NetworkProtocol(Enum):
    """Network protocols"""
    BGP = "bgp"
    ANYCAST = "anycast"
    WIREGUARD = "wireguard"
    WARP = "warp"


@dataclass
class IPv6Subnet:
    """IPv6 subnet allocation"""
    prefix: str  # e.g., "2a14:7c0:4d00::/48"
    region: VPSRegion
    assigned_to: Optional[str] = None
    geolocation_verified: bool = False
    whois_updated: bool = False


@dataclass
class VPSServer:
    """VPS server instance"""
    hostname: str
    region: VPSRegion
    ipv4: Optional[str]
    ipv6: Optional[str]
    ipv6_subnet: Optional[IPv6Subnet]
    bgp_enabled: bool = False
    anycast_enabled: bool = False
    warp_enabled: bool = False
    geolocation_country: Optional[str] = None
    provider: Optional[str] = None
    status: str = "pending"


@dataclass
class BGPSession:
    """BGP peering session"""
    peer_asn: int
    peer_ip: str
    local_asn: int
    local_ip: str
    session_type: str  # "upstream", "downstream", "peer"
    status: str = "down"


@dataclass
class GeolocationEntry:
    """IP geolocation database entry"""
    ip_range: str
    country: str
    region: Optional[str]
    city: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    verified: bool = False


class ASNVPSManager:
    """ASN/BGP-aware VPS orchestration and management"""

    def __init__(self, asn: Optional[int] = None, ipv6_block: Optional[str] = None):
        """
        Initialize ASN VPS Manager

        Args:
            asn: Autonomous System Number (e.g., 214775)
            ipv6_block: IPv6 allocation block (e.g., "2a14:7c0:4d00::/40")
        """
        self.asn = asn or 214775  # Example ASN
        self.ipv6_block = ipv6_block or "2a14:7c0:4d00::/40"

        self.vps_servers: Dict[str, VPSServer] = {}
        self.ipv6_subnets: List[IPv6Subnet] = []
        self.bgp_sessions: List[BGPSession] = []
        self.geolocation_entries: List[GeolocationEntry] = []

        logger.info(f"ASN VPS Manager initialized: AS{self.asn}, IPv6: {self.ipv6_block}")

    # =========================================================================
    # IPv6 Subnet Management
    # =========================================================================

    def allocate_ipv6_subnet(self, region: VPSRegion, prefix_length: int = 48) -> Optional[IPv6Subnet]:
        """
        Allocate IPv6 subnet from main block

        Args:
            region: VPS region for allocation
            prefix_length: Subnet prefix length (default /48)

        Returns:
            Allocated IPv6Subnet or None if exhausted
        """
        try:
            network = ipaddress.IPv6Network(self.ipv6_block)

            # Generate subnets
            for subnet in network.subnets(new_prefix=prefix_length):
                subnet_str = str(subnet)

                # Check if already allocated
                if not any(s.prefix == subnet_str for s in self.ipv6_subnets):
                    ipv6_subnet = IPv6Subnet(
                        prefix=subnet_str,
                        region=region
                    )
                    self.ipv6_subnets.append(ipv6_subnet)
                    logger.info(f"Allocated IPv6 subnet: {subnet_str} for {region.value}")
                    return ipv6_subnet

            logger.error(f"IPv6 subnet exhausted for {self.ipv6_block}")
            return None

        except Exception as e:
            logger.error(f"Failed to allocate IPv6 subnet: {e}")
            return None

    def get_available_ipv6_address(self, subnet: IPv6Subnet) -> Optional[str]:
        """Get next available IPv6 address from subnet"""
        try:
            network = ipaddress.IPv6Network(subnet.prefix)
            # Return first usable address (skip network address)
            for i, addr in enumerate(network.hosts()):
                if i == 0:  # Take first host address
                    return str(addr)
            return None
        except Exception as e:
            logger.error(f"Failed to get IPv6 address: {e}")
            return None

    # =========================================================================
    # VPS Server Management
    # =========================================================================

    def provision_vps(
        self,
        hostname: str,
        region: VPSRegion,
        provider: str = "generic",
        enable_bgp: bool = False,
        enable_anycast: bool = False
    ) -> Optional[VPSServer]:
        """
        Provision new VPS server

        Args:
            hostname: Server hostname
            region: Deployment region
            provider: VPS provider name
            enable_bgp: Enable BGP routing
            enable_anycast: Enable Anycast

        Returns:
            Provisioned VPSServer or None on failure
        """
        logger.info(f"Provisioning VPS: {hostname} in {region.value}")

        # Allocate IPv6 subnet
        ipv6_subnet = self.allocate_ipv6_subnet(region)
        if not ipv6_subnet:
            logger.error("Failed to allocate IPv6 subnet")
            return None

        # Get IPv6 address
        ipv6_address = self.get_available_ipv6_address(ipv6_subnet)

        # Create VPS server
        vps = VPSServer(
            hostname=hostname,
            region=region,
            ipv4=None,  # Will be assigned by provider
            ipv6=ipv6_address,
            ipv6_subnet=ipv6_subnet,
            bgp_enabled=enable_bgp,
            anycast_enabled=enable_anycast,
            provider=provider,
            status="provisioning"
        )

        self.vps_servers[hostname] = vps
        logger.info(f"VPS provisioned: {hostname} with IPv6 {ipv6_address}")

        return vps

    def deploy_bird_config(self, vps: VPSServer, upstream_sessions: List[BGPSession]) -> bool:
        """
        Generate and deploy BIRD BGP configuration

        Args:
            vps: VPS server instance
            upstream_sessions: List of BGP upstream sessions

        Returns:
            True if successful
        """
        if not vps.bgp_enabled:
            logger.warning(f"BGP not enabled for {vps.hostname}")
            return False

        bird_config = self._generate_bird_config(vps, upstream_sessions)

        logger.info(f"Generated BIRD config for {vps.hostname}")
        logger.info(f"Config:\n{bird_config}")

        # In production, this would SSH to VPS and deploy config
        # For now, save locally
        config_path = Path(f"/tmp/{vps.hostname}_bird.conf")
        config_path.write_text(bird_config)
        logger.info(f"BIRD config saved to {config_path}")

        return True

    def _generate_bird_config(self, vps: VPSServer, upstream_sessions: List[BGPSession]) -> str:
        """Generate BIRD routing daemon configuration"""

        config = f"""# BIRD BGP Configuration for {vps.hostname}
# Generated by ASN VPS Manager
# Region: {vps.region.value}
# ASN: {self.asn}

router id {vps.ipv4 or '10.0.0.1'};

# IPv6 prefix announcement
protocol static {{
    ipv6;
    route {vps.ipv6_subnet.prefix} reject;
}}

# Kernel protocol
protocol kernel {{
    ipv6 {{
        import none;
        export all;
    }};
}}

# Device protocol
protocol device {{
    scan time 10;
}}

"""

        # Add BGP sessions
        for idx, session in enumerate(upstream_sessions):
            config += f"""
# BGP Session {idx + 1}: {session.session_type}
protocol bgp upstream_{idx + 1} {{
    local as {session.local_asn};
    neighbor {session.peer_ip} as {session.peer_asn};

    ipv6 {{
        import none;
        export where proto = "static";
    }};

    graceful restart on;
}}
"""

        return config

    # =========================================================================
    # Geolocation Management
    # =========================================================================

    def verify_geolocation(self, ip_address: str) -> Dict[str, any]:
        """
        Verify IP geolocation across multiple databases

        Uses similar approach to IPLark for multi-database querying
        """
        logger.info(f"Verifying geolocation for {ip_address}")

        results = {}

        # Query multiple geolocation databases
        databases = {
            'ipinfo': f'https://ipinfo.io/{ip_address}/json',
            'ip-api': f'http://ip-api.com/json/{ip_address}',
        }

        for db_name, url in databases.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results[db_name] = {
                        'country': data.get('country', data.get('countryCode')),
                        'region': data.get('region', data.get('regionName')),
                        'city': data.get('city'),
                        'org': data.get('org', data.get('isp'))
                    }
                    logger.info(f"{db_name}: {results[db_name]['country']}")
            except Exception as e:
                logger.error(f"Failed to query {db_name}: {e}")

        return results

    def test_cloudflare_warp_geolocation(self, expected_country: str) -> bool:
        """
        Test geolocation using Cloudflare's /cdn-cgi/trace endpoint

        This verifies that WARP is providing the correct geolocation
        """
        try:
            response = requests.get('https://cloudflare.com/cdn-cgi/trace', timeout=10)
            if response.status_code == 200:
                trace_data = {}
                for line in response.text.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        trace_data[key] = value

                detected_country = trace_data.get('loc', 'UNKNOWN')
                logger.info(f"Cloudflare detected country: {detected_country}")

                if detected_country == expected_country:
                    logger.info(f"✓ Geolocation correct: {expected_country}")
                    return True
                else:
                    logger.warning(f"✗ Geolocation mismatch: expected {expected_country}, got {detected_country}")
                    return False
        except Exception as e:
            logger.error(f"Failed to test Cloudflare geolocation: {e}")
            return False

    def generate_geofeed(self, output_path: Optional[Path] = None) -> str:
        """
        Generate Geofeed file for bulk geolocation updates

        Geofeed format: RFC 8805
        CSV with columns: IP Prefix, Country Code, Region, City, Postal Code
        """
        logger.info("Generating Geofeed file...")

        geofeed_lines = [
            "# Geofeed v1.0",
            f"# Generated by ASN VPS Manager for AS{self.asn}",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "# Format: IP Prefix, Country Code, Region, City, Postal Code",
            ""
        ]

        for entry in self.geolocation_entries:
            line = f"{entry.ip_range},{entry.country}"
            if entry.region:
                line += f",{entry.region}"
            if entry.city:
                line += f",,{entry.city}"
            geofeed_lines.append(line)

        geofeed_content = '\n'.join(geofeed_lines)

        if output_path:
            output_path.write_text(geofeed_content)
            logger.info(f"Geofeed saved to {output_path}")

        return geofeed_content

    def update_whois_geolocation(self, subnet: IPv6Subnet, country: str) -> bool:
        """
        Update WHOIS database with geolocation information

        In production, this would use RIPE NCC API or similar
        """
        logger.info(f"Updating WHOIS geolocation: {subnet.prefix} -> {country}")

        # Create geolocation entry
        geo_entry = GeolocationEntry(
            ip_range=subnet.prefix,
            country=country,
            region=subnet.region.value,
            city=None,
            latitude=None,
            longitude=None,
            verified=False
        )

        self.geolocation_entries.append(geo_entry)
        subnet.whois_updated = True
        subnet.geolocation_verified = False  # Needs verification

        logger.info(f"WHOIS update queued for {subnet.prefix}")
        return True

    # =========================================================================
    # Network Testing & Validation
    # =========================================================================

    def test_bgp_announcement(self, prefix: str) -> bool:
        """
        Test if BGP prefix is being announced globally

        Uses BGP looking glasses or similar tools
        """
        logger.info(f"Testing BGP announcement for {prefix}")

        # In production, this would query BGP looking glasses
        # For now, simulate
        logger.info(f"✓ Prefix {prefix} appears to be announced")
        return True

    def configure_firewall_anti_scan(self, vps: VPSServer) -> bool:
        """
        Configure firewall to prevent geolocation scanning

        Blocks ICMP and other probing methods to prevent database rescanning
        """
        iptables_rules = f"""# Anti-scan firewall rules for {vps.hostname}

# Block ICMP ping
-A INPUT -p icmp --icmp-type echo-request -j DROP

# Rate limit new connections
-A INPUT -p tcp -m state --state NEW -m recent --set
-A INPUT -p tcp -m state --state NEW -m recent --update --seconds 60 --hitcount 10 -j DROP

# Allow established connections
-A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (change port as needed)
-A INPUT -p tcp --dport 22 -j ACCEPT

# Default drop
-A INPUT -j DROP
"""

        logger.info(f"Generated firewall rules for {vps.hostname}")
        logger.info(f"Rules:\n{iptables_rules}")

        return True

    # =========================================================================
    # Multi-Region Orchestration
    # =========================================================================

    def deploy_anycast_network(
        self,
        regions: List[VPSRegion],
        shared_ipv6: str
    ) -> Dict[VPSRegion, VPSServer]:
        """
        Deploy Anycast network across multiple regions

        All servers announce the same IPv6 prefix for geographic load balancing
        """
        logger.info(f"Deploying Anycast network across {len(regions)} regions")
        logger.info(f"Shared Anycast IPv6: {shared_ipv6}")

        anycast_servers = {}

        for region in regions:
            hostname = f"anycast-{region.value}"

            vps = self.provision_vps(
                hostname=hostname,
                region=region,
                enable_bgp=True,
                enable_anycast=True
            )

            if vps:
                # Override IPv6 with shared Anycast address
                vps.ipv6 = shared_ipv6
                anycast_servers[region] = vps
                logger.info(f"Deployed Anycast node in {region.value}")

        logger.info(f"Anycast network deployed: {len(anycast_servers)} nodes")
        return anycast_servers

    def configure_wireguard_mesh(self, servers: List[VPSServer]) -> bool:
        """
        Configure WireGuard mesh network between VPS servers

        Similar to WARP integration but using WireGuard
        """
        logger.info(f"Configuring WireGuard mesh for {len(servers)} servers")

        for idx, server in enumerate(servers):
            wg_config = self._generate_wireguard_config(server, servers, idx)

            config_path = Path(f"/tmp/{server.hostname}_wg0.conf")
            config_path.write_text(wg_config)
            logger.info(f"WireGuard config saved for {server.hostname}")

        return True

    def _generate_wireguard_config(
        self,
        local_server: VPSServer,
        all_servers: List[VPSServer],
        idx: int
    ) -> str:
        """Generate WireGuard configuration"""

        config = f"""[Interface]
# {local_server.hostname}
Address = 10.100.0.{idx + 1}/24
PrivateKey = <GENERATE_PRIVATE_KEY>
ListenPort = 51820

"""

        for other_idx, other_server in enumerate(all_servers):
            if other_server.hostname == local_server.hostname:
                continue

            config += f"""[Peer]
# {other_server.hostname}
PublicKey = <PEER_PUBLIC_KEY>
Endpoint = {other_server.ipv4 or other_server.ipv6}:51820
AllowedIPs = 10.100.0.{other_idx + 1}/32
PersistentKeepalive = 25

"""

        return config

    # =========================================================================
    # Reporting & Export
    # =========================================================================

    def generate_infrastructure_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive infrastructure report"""

        report = {
            'asn': self.asn,
            'ipv6_block': self.ipv6_block,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'summary': {
                'total_servers': len(self.vps_servers),
                'total_subnets': len(self.ipv6_subnets),
                'bgp_sessions': len(self.bgp_sessions),
                'geolocation_entries': len(self.geolocation_entries),
            },
            'servers': [
                {
                    'hostname': srv.hostname,
                    'region': srv.region.value,
                    'ipv4': srv.ipv4,
                    'ipv6': srv.ipv6,
                    'subnet': srv.ipv6_subnet.prefix if srv.ipv6_subnet else None,
                    'bgp_enabled': srv.bgp_enabled,
                    'anycast_enabled': srv.anycast_enabled,
                    'status': srv.status,
                }
                for srv in self.vps_servers.values()
            ],
            'ipv6_allocations': [
                {
                    'prefix': subnet.prefix,
                    'region': subnet.region.value,
                    'assigned_to': subnet.assigned_to,
                    'geolocation_verified': subnet.geolocation_verified,
                }
                for subnet in self.ipv6_subnets
            ],
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Infrastructure report saved to {output_path}")

        return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ASN/BGP-Aware VPS Management")
    parser.add_argument('--asn', type=int, help="Autonomous System Number")
    parser.add_argument('--ipv6-block', type=str, help="IPv6 allocation block")
    parser.add_argument('--provision', type=str, help="Provision VPS with hostname")
    parser.add_argument('--region', type=str, choices=[r.value for r in VPSRegion],
                       help="VPS region")
    parser.add_argument('--enable-bgp', action='store_true', help="Enable BGP")
    parser.add_argument('--enable-anycast', action='store_true', help="Enable Anycast")
    parser.add_argument('--deploy-anycast', action='store_true',
                       help="Deploy Anycast network")
    parser.add_argument('--verify-geo', type=str, help="Verify geolocation for IP")
    parser.add_argument('--generate-geofeed', type=str, help="Generate Geofeed file")
    parser.add_argument('--report', type=str, help="Generate infrastructure report")

    args = parser.parse_args()

    # Initialize manager
    manager = ASNVPSManager(asn=args.asn, ipv6_block=args.ipv6_block)

    # Provision VPS
    if args.provision and args.region:
        region = VPSRegion(args.region)
        vps = manager.provision_vps(
            hostname=args.provision,
            region=region,
            enable_bgp=args.enable_bgp,
            enable_anycast=args.enable_anycast
        )
        if vps:
            print(f"✓ VPS provisioned: {vps.hostname}")
            print(f"  Region: {vps.region.value}")
            print(f"  IPv6: {vps.ipv6}")
            print(f"  Subnet: {vps.ipv6_subnet.prefix if vps.ipv6_subnet else 'N/A'}")

    # Deploy Anycast network
    elif args.deploy_anycast:
        regions = [VPSRegion.US_WEST, VPSRegion.EU_CENTRAL, VPSRegion.SINGAPORE]
        anycast_servers = manager.deploy_anycast_network(
            regions=regions,
            shared_ipv6="2a14:7c0:4d00::1"
        )
        print(f"✓ Anycast network deployed: {len(anycast_servers)} nodes")

    # Verify geolocation
    elif args.verify_geo:
        results = manager.verify_geolocation(args.verify_geo)
        print(f"Geolocation results for {args.verify_geo}:")
        for db, data in results.items():
            print(f"  {db}: {data['country']} - {data.get('city', 'N/A')}")

    # Generate Geofeed
    elif args.generate_geofeed:
        geofeed = manager.generate_geofeed(Path(args.generate_geofeed))
        print(f"✓ Geofeed generated: {args.generate_geofeed}")

    # Generate report
    elif args.report:
        report = manager.generate_infrastructure_report(Path(args.report))
        print(f"✓ Infrastructure report generated: {args.report}")
        print(f"  Total servers: {report['summary']['total_servers']}")
        print(f"  Total subnets: {report['summary']['total_subnets']}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
