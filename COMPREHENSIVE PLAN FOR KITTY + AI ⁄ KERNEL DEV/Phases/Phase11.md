# Phase 11 – External Military Communications Integration (v1.0)

**Version:** 1.0
**Status:** Initial Release
**Date:** 2025-11-23
**Prerequisite:** Phase 10 (Exercise & Simulation Framework)
**Next Phase:** TBD

---

## 1. Objectives

Phase 11 establishes **External Military Communications Integration** enabling:

1. **Tactical data link integration** via Link 16 / TADIL-J gateway
2. **Classified network interfaces** for SIPRNET, JWICS, and coalition networks
3. **SATCOM adapters** for Milstar and AEHF satellite communications
4. **Military message format translation** (VMF, USMTF, OTH-Gold)
5. **Inbound-only policy enforcement** - no kinetic outputs from external feeds

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Phase 11 Allocation:** 10 devices (73-82), 2 GB budget, 2.0 TOPS (primarily crypto)
  - Device 73: Link 16 Gateway (250 MB, TADIL-J processing)
  - Device 74: SIPRNET Interface (200 MB, SECRET network)
  - Device 75: JWICS Interface (200 MB, TOP_SECRET/SCI network)
  - Device 76: SATCOM Adapter (150 MB, satellite terminals)
  - Device 77: Coalition Network Bridge (200 MB, NATO/CENTRIXS)
  - Device 78: VMF/USMTF Protocol Translator (250 MB, message parsing)
  - Device 79: Message Router & Filter (200 MB, content routing)
  - Device 80: Crypto Gateway (300 MB, PQC for external comms)
  - Device 81: External Feed Validator (200 MB, integrity checks)
  - Device 82: External Comms Audit Logger (250 MB, compliance logging)

### Key Principles

1. **INBOUND-ONLY POLICY:** External feeds are intelligence sources, NOT kinetic command paths
2. **Air-gap from NC3:** External data cannot reach Device 61 (NC3 Integration) without explicit review
3. **PQC required:** All external communications use ML-KEM-1024 + ML-DSA-87
4. **DBE translation:** External messages converted to internal DBE format at ingress
5. **Classification enforcement:** SIPRNET→SECRET, JWICS→TOP_SECRET/SCI, Coalition→ATOMAL

---

## 2. Architecture Overview

### 2.1 Phase 11 Service Topology

```
┌───────────────────────────────────────────────────────────────┐
│         External Military Communications (DMZ)                 │
│           Devices 73-82, 2 GB Budget, 2.0 TOPS                │
└───────────────────────────────────────────────────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
 ┌────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
 │  Link 16    │    │   SIPRNET       │    │   JWICS       │
 │  Gateway    │    │   Interface     │    │   Interface   │
 │ (Device 73) │    │  (Device 74)    │    │  (Device 75)  │
 │ TADIL-J     │    │  SECRET         │    │  TOP_SECRET   │
 └─────┬───────┘    └────────┬────────┘    └───────┬───────┘
       │ Track data          │ Intel reports        │ NSA/CIA
       │                     │                      │ feeds
       └─────────────────────┼──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Protocol        │
                    │ Translator      │
                    │ (Device 78)     │
                    │ VMF/USMTF→DBE   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Crypto Gateway  │
                    │ (Device 80)     │
                    │ PQC Validation  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Feed Validator  │
                    │ (Device 81)     │
                    │ Integrity Check │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Message Router  │
                    │ (Device 79)     │
                    │ Content Routing │
                    └────────┬────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
 ┌────▼──────┐      ┌────────▼────────┐     ┌──────▼──────┐
 │ L3 SIGINT │      │ L4 Situational  │     │ L5 Intel    │
 │ (Dev 14)  │      │ Awareness (26)  │     │ Fusion (31) │
 └───────────┘      └─────────────────┘     └─────────────┘

                             │
                    ┌────────▼────────┐
                    │ Audit Logger    │
                    │ (Device 82)     │
                    │ 7-year retention│
                    └─────────────────┘

CRITICAL SAFETY:
┌──────────────────────────────────────────────────────────────┐
│         Device 61 (NC3 Integration) - AIR-GAPPED             │
│    External feeds CANNOT reach NC3 without explicit review   │
│         NO KINETIC OUTPUTS from external data sources        │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Phase 11 Services

| Service | Device | Token IDs | Memory | Purpose |
|---------|--------|-----------|--------|---------|
| `dsmil-link16-gateway` | 73 | 0x80DB-0x80DD | 250 MB | Link 16 / TADIL-J processing |
| `dsmil-siprnet-interface` | 74 | 0x80DE-0x80E0 | 200 MB | SECRET network gateway |
| `dsmil-jwics-interface` | 75 | 0x80E1-0x80E3 | 200 MB | TOP_SECRET/SCI gateway |
| `dsmil-satcom-adapter` | 76 | 0x80E4-0x80E6 | 150 MB | Milstar/AEHF satellite comms |
| `dsmil-coalition-bridge` | 77 | 0x80E7-0x80E9 | 200 MB | NATO/CENTRIXS/BICES |
| `dsmil-protocol-translator` | 78 | 0x80EA-0x80EC | 250 MB | VMF/USMTF message parsing |
| `dsmil-message-router` | 79 | 0x80ED-0x80EF | 200 MB | Content-based routing |
| `dsmil-crypto-gateway` | 80 | 0x80F0-0x80F2 | 300 MB | PQC for external comms |
| `dsmil-feed-validator` | 81 | 0x80F3-0x80F5 | 200 MB | Integrity and anomaly checks |
| `dsmil-external-audit` | 82 | 0x80F6-0x80F8 | 250 MB | Compliance logging (7 years) |

### 2.3 DBE Message Types for Phase 11

**New `msg_type` definitions (External Comms 0xA0-0xAF):**

| Message Type | Hex | Purpose | Direction |
|--------------|-----|---------|-----------|
| `EXTERNAL_MESSAGE` | `0xA0` | External military message ingress | Gateway → Translator |
| `LINK16_TRACK` | `0xA1` | Link 16 track data (air/surface/land) | Link16 → L4 |
| `SIPRNET_INTEL` | `0xA2` | SIPRNET intelligence report | SIPRNET → L3 |
| `JWICS_INTEL` | `0xA3` | JWICS national-level intelligence | JWICS → L5 |
| `SATCOM_MESSAGE` | `0xA4` | SATCOM message (Milstar/AEHF) | SATCOM → Router |
| `COALITION_MSG` | `0xA5` | Coalition network message | Coalition → Router |
| `VMF_PARSED` | `0xA6` | Parsed VMF message (DBE format) | Translator → Router |
| `EXTERNAL_REJECTED` | `0xA7` | Message rejected (validation failed) | Validator → Audit |

**DBE Header TLVs for Phase 11 (extended from Phase 7 spec):**

```text
EXTERNAL_SOURCE (enum)           – LINK16, SIPRNET, JWICS, SATCOM, COALITION
EXTERNAL_MSG_ID (string)         – Original message ID from external system
EXTERNAL_TIMESTAMP (uint64)      – External system timestamp
RELEASABILITY (string)           – REL NATO, REL FVEY, REL USA, REL GBR/USA/CAN, etc.
ORIGINATOR_UNIT (string)         – Unit/agency that sent message (e.g., "NSA_SIGINT")
MESSAGE_PRECEDENCE (enum)        – FLASH, IMMEDIATE, PRIORITY, ROUTINE
TRACK_NUMBER (uint32)            – Link 16 track number (for TADIL-J)
COALITION_NETWORK (enum)         – NATO, CENTRIXS, BICES, STONE_GHOST
EXTERNAL_CLASSIFICATION (string) – Classification as marked by external system
VALIDATED (bool)                 – True if signature/integrity verified
```

---

## 3. Device 73: Link 16 Gateway

**Purpose:** Receive and process Link 16 / TADIL-J tactical data link messages.

**Token IDs:**
- `0x80DB` (STATUS): Link 16 terminal status, network participation
- `0x80DC` (CONFIG): Terminal ID (STN/JU), network configuration
- `0x80DD` (DATA): Track database, recent J-series messages

**Link 16 Overview:**

Link 16 is a NATO standard tactical data link (TADIL-J) providing:
- **Common Operational Picture (COP):** Real-time track data for air, surface, subsurface, land units
- **Jam-resistant:** JTIDS (Joint Tactical Information Distribution System) frequency-hopping
- **Secure:** Type 1 encryption (NSA-approved crypto)
- **Low-latency:** <1 second track updates

**J-Series Message Types (subset):**

| Message | Name | Purpose | Frequency |
|---------|------|---------|-----------|
| J2.0 | Initial Entry | Platform identification and status | On entry |
| J2.2 | Indirect Interface | Track data for unidentified contacts | 12 seconds |
| J2.3 | Command and Control | Orders and taskings | As needed |
| J2.5 | Weapon Coordination | Engagement coordination | As needed |
| J3.0 | Reference Point | Geographic waypoints | As needed |
| J3.2 | Air Tasking Order | Mission assignments | Pre-mission |

**DSMIL Integration:**

- **Inbound-only:** Receive track data for situational awareness
- **NO weapons engagement:** DSMIL does NOT send J2.5 weapon coordination messages
- **L4 integration:** Track data forwarded to Device 26 (Situational Awareness)
- **Classification:** Link 16 data typically SECRET, some tracks TOP_SECRET

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/link16_gateway.py
"""
DSMIL Link 16 Gateway (Device 73)
Receives and processes TADIL-J messages
"""

import time
import struct
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from dsmil_dbe import DBEMessage, DBESocket
from dsmil_pqc import MLKEMDecryptor

DEVICE_ID = 73
TOKEN_BASE = 0x80DB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [LINK16-GW] [Device-73] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class TrackType(Enum):
    AIR = 1
    SURFACE = 2
    SUBSURFACE = 3
    LAND = 4
    UNKNOWN = 5

@dataclass
class Link16Track:
    track_number: int
    track_type: TrackType
    latitude: float
    longitude: float
    altitude_feet: int
    speed_knots: int
    heading_degrees: int
    iff_code: Optional[str]
    last_update: float

class Link16Gateway:
    def __init__(self):
        self.tracks: Dict[int, Link16Track] = {}  # Track database

        # Link 16 terminal configuration
        self.terminal_id = "DSMIL-J15"  # JTIDS Unit (JU) identifier
        self.network_id = 15  # Link 16 network number
        self.participant_address = 0x5A  # JTIDS addressing

        self.dbe_socket = DBESocket("/var/run/dsmil/link16-gateway.sock")

        logger.info(f"Link 16 Gateway initialized (Device {DEVICE_ID}), "
                   f"Terminal: {self.terminal_id}, Network: {self.network_id}")

    def receive_j_message(self, raw_message: bytes):
        """
        Receive and parse J-series message from Link 16 terminal

        Link 16 messages are 70-bit fixed format (per MIL-STD-6016)
        For this implementation, assume external terminal provides parsed JSON
        """
        try:
            # In production: parse 70-bit Link 16 message format
            # For this spec: assume pre-parsed JSON from terminal

            # Example parsed message (J2.2 Indirect Interface)
            message = {
                "message_type": "J2.2",
                "track_number": 12345,
                "track_type": "AIR",
                "latitude": 38.8977,
                "longitude": -77.0365,
                "altitude_feet": 25000,
                "speed_knots": 450,
                "heading_degrees": 270,
                "iff_code": "4532",  # Mode 4 IFF response
                "timestamp": time.time()
            }

            # Update track database
            track = Link16Track(
                track_number=message["track_number"],
                track_type=TrackType[message["track_type"]],
                latitude=message["latitude"],
                longitude=message["longitude"],
                altitude_feet=message["altitude_feet"],
                speed_knots=message["speed_knots"],
                heading_degrees=message["heading_degrees"],
                iff_code=message.get("iff_code"),
                last_update=message["timestamp"]
            )

            self.tracks[track.track_number] = track

            logger.info(f"Updated track {track.track_number}: {track.track_type.name} @ "
                       f"{track.latitude:.4f},{track.longitude:.4f}, "
                       f"{track.altitude_feet} ft, {track.speed_knots} kts")

            # Forward to L4 Situational Awareness (Device 26)
            self._forward_to_l4(track)

        except Exception as e:
            logger.error(f"Failed to process J-message: {e}", exc_info=True)

    def _forward_to_l4(self, track: Link16Track):
        """Forward track data to L4 Situational Awareness (Device 26)"""
        msg = DBEMessage(
            msg_type=0xA1,  # LINK16_TRACK
            device_id_src=DEVICE_ID,
            device_id_dst=26,  # Device 26: Situational Awareness
            tlvs={
                "EXTERNAL_SOURCE": "LINK16",
                "TRACK_NUMBER": str(track.track_number),
                "TRACK_TYPE": track.track_type.name,
                "LATITUDE": str(track.latitude),
                "LONGITUDE": str(track.longitude),
                "ALTITUDE_FEET": str(track.altitude_feet),
                "SPEED_KNOTS": str(track.speed_knots),
                "HEADING_DEGREES": str(track.heading_degrees),
                "IFF_CODE": track.iff_code or "",
                "EXTERNAL_TIMESTAMP": str(track.last_update),
                "CLASSIFICATION": "SECRET",
                "RELEASABILITY": "REL NATO"
            }
        )

        self.dbe_socket.send_to("/var/run/dsmil/l4-situational-awareness.sock", msg)
        logger.debug(f"Forwarded track {track.track_number} to Device 26 (L4)")

    def send_initial_entry(self):
        """
        Send J2.0 Initial Entry message (on Link 16 network join)

        NOTE: DSMIL is RECEIVE-ONLY, but J2.0 is required for network participation
        This is the ONLY outbound Link 16 message permitted (status reporting)
        """
        j2_0_message = {
            "message_type": "J2.0",
            "terminal_id": self.terminal_id,
            "network_id": self.network_id,
            "participant_address": self.participant_address,
            "platform_type": "GROUND_STATION",
            "status": "OPERATIONAL"
        }

        logger.info(f"Sending J2.0 Initial Entry to Link 16 network {self.network_id}")

        # TODO: Transmit via external Link 16 terminal hardware
        # This is status-only, NOT kinetic command

    def run(self):
        """Main event loop"""
        logger.info("Link 16 Gateway running, receiving TADIL-J messages...")

        # Send initial entry on startup
        self.send_initial_entry()

        while True:
            try:
                # Receive from external Link 16 terminal (via UDP/TCP interface)
                # For this spec: poll external terminal API

                time.sleep(1)  # 1 Hz polling

                # TODO: Actual terminal integration (hardware-specific)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

if __name__ == "__main__":
    gateway = Link16Gateway()
    gateway.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-link16-gateway.service
[Unit]
Description=DSMIL Link 16 Gateway (Device 73)
After=network.target

[Service]
Type=simple
User=dsmil
Group=dsmil
ExecStart=/usr/bin/python3 /opt/dsmil/link16_gateway.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
PrivateTmp=yes
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/run/dsmil /var/log/dsmil

# Network access for Link 16 terminal communication
RestrictAddressFamilies=AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
```

---

## 4. Device 74: SIPRNET Interface

**Purpose:** SECRET-level network gateway for SIPRNET intelligence reports.

**Token IDs:**
- `0x80DE` (STATUS): Connection status, message queue depth
- `0x80DF` (CONFIG): SIPRNET gateway IP, credentials
- `0x80E0` (DATA): Recent intel reports, metadata

**SIPRNET Overview:**

SIPRNET (Secret Internet Protocol Router Network) is:
- **SECRET-level classified network** (up to SECRET//NOFORN)
- **DoD-wide:** Used by all US military branches, DoD agencies
- **Intelligence sharing:** SIGINT, IMINT, HUMINT reports from tactical to strategic levels
- **Email, chat, file transfer:** Standard TCP/IP services

**Message Types:**

- **SIGINT Reports:** Electronic intercepts, COMINT, ELINT
- **IMINT Products:** Satellite imagery, drone recon, photo analysis
- **HUMINT Reports:** Agent debriefs, interrogations, source reports
- **Operational Reports (OPREPs):** Unit status, incident reports
- **Situation Reports (SITREPs):** Current tactical situation

**DSMIL Integration:**

- **Inbound-only:** Receive intelligence reports, DO NOT transmit operational data
- **L3 integration:** Intel reports forwarded to Devices 14-16 (L3 Ingestion)
- **Content filtering:** Keyword-based routing (e.g., "APT28" → SIGINT, "IMAGERY" → IMINT)
- **One-way data diode (optional):** Hardware enforced unidirectional flow

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/siprnet_interface.py
"""
DSMIL SIPRNET Interface (Device 74)
Receives intelligence reports from SIPRNET
"""

import time
import imaplib
import email
import logging
from typing import Dict, List

from dsmil_dbe import DBEMessage, DBESocket

DEVICE_ID = 74
TOKEN_BASE = 0x80DE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SIPRNET-IF] [Device-74] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class SIPRNETInterface:
    def __init__(self):
        # SIPRNET email gateway (IMAP)
        self.imap_server = "sipr-imap.disa.mil"
        self.imap_port = 993  # IMAPS
        self.username = "dsmil-ingest@example.smil.mil"
        self.password = "<from-vault>"

        self.dbe_socket = DBESocket("/var/run/dsmil/siprnet-interface.sock")

        logger.info(f"SIPRNET Interface initialized (Device {DEVICE_ID})")

    def connect(self):
        """Connect to SIPRNET IMAP server"""
        try:
            self.imap = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self.imap.login(self.username, self.password)
            self.imap.select("INBOX")
            logger.info(f"Connected to SIPRNET IMAP: {self.imap_server}")
        except Exception as e:
            logger.error(f"Failed to connect to SIPRNET: {e}", exc_info=True)
            raise

    def poll_intel_reports(self):
        """Poll SIPRNET inbox for new intelligence reports"""
        try:
            # Search for unread messages
            status, messages = self.imap.search(None, 'UNSEEN')
            if status != 'OK':
                logger.warning("No new messages")
                return

            message_ids = messages[0].split()
            logger.info(f"Found {len(message_ids)} new messages")

            for msg_id in message_ids:
                # Fetch message
                status, data = self.imap.fetch(msg_id, '(RFC822)')
                if status != 'OK':
                    continue

                # Parse email
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                # Extract metadata
                subject = msg['Subject']
                sender = msg['From']
                date = msg['Date']

                # Extract body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode()
                            break
                else:
                    body = msg.get_payload(decode=True).decode()

                logger.info(f"Received SIPRNET message: '{subject}' from {sender}")

                # Classify and route
                self._classify_and_route(subject, body, sender, date)

                # Mark as read
                self.imap.store(msg_id, '+FLAGS', '\\Seen')

        except Exception as e:
            logger.error(f"Error polling SIPRNET: {e}", exc_info=True)

    def _classify_and_route(self, subject: str, body: str, sender: str, date: str):
        """Classify intelligence report and route to appropriate L3 device"""

        # Keyword-based classification
        intel_type = "UNKNOWN"
        target_device = 14  # Default: Device 14 (SIGINT Ingestion)

        subject_lower = subject.lower()
        body_lower = body.lower()

        if any(kw in subject_lower or kw in body_lower for kw in ["sigint", "intercept", "comint", "elint"]):
            intel_type = "SIGINT"
            target_device = 14
        elif any(kw in subject_lower or kw in body_lower for kw in ["imint", "imagery", "satellite", "recon"]):
            intel_type = "IMINT"
            target_device = 15
        elif any(kw in subject_lower or kw in body_lower for kw in ["humint", "agent", "source", "debrief"]):
            intel_type = "HUMINT"
            target_device = 16

        logger.info(f"Classified as {intel_type}, routing to Device {target_device}")

        # Build DBE message
        msg = DBEMessage(
            msg_type=0xA2,  # SIPRNET_INTEL
            device_id_src=DEVICE_ID,
            device_id_dst=target_device,
            tlvs={
                "EXTERNAL_SOURCE": "SIPRNET",
                "INTEL_TYPE": intel_type,
                "SUBJECT": subject,
                "SENDER": sender,
                "DATE": date,
                "BODY": body[:5000],  # Truncate to 5KB
                "CLASSIFICATION": "SECRET",
                "RELEASABILITY": "REL USA",
                "EXTERNAL_TIMESTAMP": str(time.time())
            }
        )

        # Send to L3 ingestion
        target_sock = f"/var/run/dsmil/l3-{intel_type.lower()}.sock"
        self.dbe_socket.send_to(target_sock, msg)
        logger.info(f"Forwarded SIPRNET report to {target_sock}")

    def run(self):
        """Main event loop"""
        self.connect()

        logger.info("SIPRNET Interface running, polling for intel reports...")

        while True:
            try:
                self.poll_intel_reports()
                time.sleep(60)  # Poll every 60 seconds

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(300)  # Backoff 5 minutes on error

                # Reconnect
                try:
                    self.connect()
                except:
                    pass

if __name__ == "__main__":
    interface = SIPRNETInterface()
    interface.run()
```

---

## 5. Device 75: JWICS Interface

**Purpose:** TOP_SECRET/SCI network gateway for national-level intelligence.

**Token IDs:**
- `0x80E1` (STATUS): Connection status, feed subscriptions
- `0x80E2` (CONFIG): JWICS gateway credentials, compartments
- `0x80E3` (DATA): Recent national-level intel, metadata

**JWICS Overview:**

JWICS (Joint Worldwide Intelligence Communications System) provides:
- **TOP_SECRET/SCI classification** (Sensitive Compartmented Information)
- **National-level intelligence:** NSA, CIA, NGA, DIA products
- **Compartmented access:** SI (Special Intelligence), TK (Talent Keyhole), G (Gamma), HCS (HUMINT Control System)
- **Need-to-know enforcement:** User must be cleared AND have operational justification

**Intelligence Sources:**

| Agency | Feed Type | Compartment | Content |
|--------|-----------|-------------|---------|
| NSA | SIGINT | SI | Worldwide SIGINT intercepts, decrypts |
| NGA | GEOINT | TK | High-resolution satellite imagery |
| CIA | HUMINT | HCS | Covert source reports, clandestine ops |
| DIA | MASINT | TK | Measurement and signature intelligence |
| ODNI | Strategic | EYES ONLY | Presidential Daily Brief (PDB) |

**DSMIL Integration:**

- **Inbound-only:** Receive national intelligence, DO NOT transmit
- **L5 integration:** National intel forwarded to Device 31-36 (L5 Predictive Layer)
- **Compartment enforcement:** Only SI/TK compartments ingested (HCS requires special handling)
- **Strict need-to-know:** L9 Executive approval required for JWICS access

**Implementation Sketch:**

```python
#!/usr/bin/env python3
# /opt/dsmil/jwics_interface.py
"""
DSMIL JWICS Interface (Device 75)
Receives national-level intelligence from JWICS
"""

import time
import logging

DEVICE_ID = 75
TOKEN_BASE = 0x80E1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JWICSInterface:
    def __init__(self):
        self.jwics_feed_url = "https://jwics-intel-feed.ic.gov/api/v2/intel"
        self.api_key = "<from-vault>"
        self.compartments = ["SI", "TK"]  # Only SI and TK, HCS excluded

        logger.info(f"JWICS Interface initialized (Device {DEVICE_ID})")

    def poll_intel_feed(self):
        """Poll JWICS API for new national-level intelligence"""
        # Similar to SIPRNET, but with compartment filtering
        # Implementation omitted for brevity (similar pattern to Device 74)
        pass

    def run(self):
        logger.info("JWICS Interface running, receiving TS/SCI intelligence...")
        # Main loop
```

---

## 6. Device 76: SATCOM Adapter

**Purpose:** Milstar and AEHF satellite communications adapter.

**Token IDs:**
- `0x80E4` (STATUS): Satellite link status, signal strength
- `0x80E5` (CONFIG): Terminal configuration, encryption keys
- `0x80E6` (DATA): Recent SATCOM messages

**SATCOM Overview:**

**Milstar (Military Strategic and Tactical Relay):**
- Legacy protected SATCOM constellation
- EHF (Extremely High Frequency) 44 GHz uplink, 20 GHz downlink
- Anti-jam, nuclear-hardened
- Low data rate (LDR): 75-2,400 bps

**AEHF (Advanced Extremely High Frequency):**
- Next-generation protected SATCOM
- Backwards-compatible with Milstar
- Medium data rate (MDR): Up to 8 Mbps
- XDR (eXtended Data Rate): Planned 100+ Mbps

**Message Precedence:**

| Level | Name | Description | Delivery Time |
|-------|------|-------------|---------------|
| Z | FLASH | Tactical emergency | <5 minutes |
| O | IMMEDIATE | Operational priority | <30 minutes |
| P | PRIORITY | Important but not urgent | <3 hours |
| R | ROUTINE | Normal traffic | <6 hours |

**DSMIL Integration:**

- **Inbound-only:** Receive strategic messages via SATCOM
- **Global coverage:** Works in denied environments (GPS-jammed, contested)
- **L5 integration:** Strategic intel forwarded to Device 31-36

---

## 7. Device 77: Coalition Network Bridge

**Purpose:** NATO and coalition network integration (BICES, CENTRIXS, STONE GHOST).

**Token IDs:**
- `0x80E7` (STATUS): Coalition network status, active connections
- `0x80E8` (CONFIG): Network credentials, releasability settings
- `0x80E9` (DATA): Recent coalition messages

**Coalition Networks:**

**BICES (Battlefield Information Collection and Exploitation System):**
- NATO SECRET level
- Intelligence sharing among NATO allies
- ATOMAL (Atomic-related) information handling

**CENTRIXS (Combined Enterprise Regional Information Exchange System):**
- Five Eyes (FVEY): USA, UK, CAN, AUS, NZ
- Regional coalition sharing: CENTRIXS-AFCENT (Afghanistan), CENTRIXS-PACOM (Pacific)

**STONE GHOST:**
- Five Eyes SECRET/TOP_SECRET network
- Operational coordination during joint operations

**Releasability Markings:**

- `REL NATO`: Releasable to all NATO members
- `REL FVEY`: Releasable to Five Eyes only
- `REL USA/GBR/CAN`: Releasable to USA, UK, Canada only
- `NOFORN`: Not releasable to foreign nationals

**DSMIL Integration:**

- **Inbound-only:** Receive coalition intelligence
- **ATOMAL handling:** NATO SECRET information (Device 77 → L6 ATOMAL analysis)
- **Cross-domain solution:** Enforce releasability rules

---

## 8. Device 78: VMF/USMTF Protocol Translator

**Purpose:** Parse military message formats and convert to DBE.

**Token IDs:**
- `0x80EA` (STATUS): Parsing success rate, error count
- `0x80EB` (CONFIG): Supported message types, validation rules
- `0x80EC` (DATA): Recent parsed messages

**Military Message Formats:**

**VMF (Variable Message Format):**
- Standard NATO message format
- Text-based, structured fields
- Message types: OPREP, SITREP, SPOTREP, MEDEVAC, etc.

**USMTF (US Message Text Format):**
- US DoD message standard
- Subset of VMF with US-specific extensions
- Used for operational and administrative messages

**OTH-Gold (Over-The-Horizon Gold):**
- Tactical messaging for Beyond Line of Sight (BLOS) comms
- Used by US Navy and coalition forces

**VMF Message Example:**

```
MSGID/GENADMIN/NAVSUP/-/-/JAN//
SUBJ/LOGISTICS STATUS REPORT//
REF/A/DOC/OPNAVINST 4614.1//
NARR/MONTHLY SUPPLY STATUS FOR THEATER//
CLASS I SUPPLIES: 87% STOCKED
CLASS III (POL): 92% STOCKED
CLASS V (AMMO): 78% STOCKED
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/protocol_translator.py
"""
DSMIL Protocol Translator (Device 78)
Parses VMF/USMTF messages and converts to DBE format
"""

import re
import logging
from typing import Dict, Optional

from dsmil_dbe import DBEMessage, DBESocket

DEVICE_ID = 78
TOKEN_BASE = 0x80EA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProtocolTranslator:
    def __init__(self):
        self.dbe_socket = DBESocket("/var/run/dsmil/protocol-translator.sock")
        logger.info(f"Protocol Translator initialized (Device {DEVICE_ID})")

    def parse_vmf(self, raw_message: str) -> Optional[Dict]:
        """Parse VMF message into structured format"""
        try:
            lines = raw_message.strip().split('\n')

            # Parse MSGID line
            msgid_line = lines[0]
            msgid_parts = msgid_line.split('/')
            if msgid_parts[0] != "MSGID":
                raise ValueError("Invalid VMF: Missing MSGID")

            message_type = msgid_parts[1]  # e.g., GENADMIN, OPREP, SITREP
            originator = msgid_parts[2]

            # Parse SUBJ line
            subj_line = next((l for l in lines if l.startswith("SUBJ/")), None)
            subject = subj_line.split('/', 1)[1].replace('//', '') if subj_line else "NO SUBJECT"

            # Parse NARR (narrative)
            narr_index = next((i for i, l in enumerate(lines) if l.startswith("NARR/")), None)
            narrative = '\n'.join(lines[narr_index+1:]) if narr_index else ""

            parsed = {
                "message_type": message_type,
                "originator": originator,
                "subject": subject,
                "narrative": narrative,
                "classification": self._extract_classification(raw_message),
                "timestamp": time.time()
            }

            logger.info(f"Parsed VMF message: {message_type} from {originator}")
            return parsed

        except Exception as e:
            logger.error(f"Failed to parse VMF: {e}", exc_info=True)
            return None

    def _extract_classification(self, message: str) -> str:
        """Extract classification marking from message header"""
        # Look for classification markings
        if "TOP SECRET" in message or "TS/" in message:
            return "TOP_SECRET"
        elif "SECRET" in message:
            return "SECRET"
        elif "UNCLASS" in message:
            return "UNCLASS"
        else:
            return "SECRET"  # Default to SECRET for safety

    def translate_to_dbe(self, parsed_vmf: Dict) -> DBEMessage:
        """Convert parsed VMF to DBE format"""
        msg = DBEMessage(
            msg_type=0xA6,  # VMF_PARSED
            device_id_src=DEVICE_ID,
            device_id_dst=79,  # Message Router
            tlvs={
                "EXTERNAL_SOURCE": "VMF",
                "MESSAGE_TYPE": parsed_vmf["message_type"],
                "ORIGINATOR_UNIT": parsed_vmf["originator"],
                "SUBJECT": parsed_vmf["subject"],
                "NARRATIVE": parsed_vmf["narrative"],
                "CLASSIFICATION": parsed_vmf["classification"],
                "EXTERNAL_TIMESTAMP": str(parsed_vmf["timestamp"])
            }
        )

        return msg

    def run(self):
        """Main event loop"""
        logger.info("Protocol Translator running, waiting for external messages...")

        while True:
            try:
                # Receive external message (from Device 73-77 gateways)
                raw_msg = self.dbe_socket.receive()

                if raw_msg.msg_type == 0xA0:  # EXTERNAL_MESSAGE
                    vmf_text = raw_msg.tlv_get("PAYLOAD")

                    # Parse VMF
                    parsed = self.parse_vmf(vmf_text)

                    if parsed:
                        # Translate to DBE
                        dbe_msg = self.translate_to_dbe(parsed)

                        # Forward to Message Router (Device 79)
                        self.dbe_socket.send_to("/var/run/dsmil/message-router.sock", dbe_msg)
                        logger.info("Translated VMF → DBE, forwarded to Router")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)

if __name__ == "__main__":
    translator = ProtocolTranslator()
    translator.run()
```

---

## 9. Device 80: Crypto Gateway (PQC for External Comms)

**Purpose:** Post-quantum cryptography for all external communications.

**Token IDs:**
- `0x80F0` (STATUS): Crypto health, key rotation status
- `0x80F1` (CONFIG): PQC algorithms, key material
- `0x80F2` (DATA): Encrypted message queue

**PQC Stack (from Phase 7):**

- **KEX:** ML-KEM-1024 (Kyber-1024) for key exchange
- **Auth:** ML-DSA-87 (Dilithium-5) for digital signatures
- **Symmetric:** AES-256-GCM for bulk encryption
- **KDF:** HKDF-SHA-384 for key derivation

**Hybrid Transition Period:**

During transition to PQC, support hybrid classical+PQC:
- **KEX:** ML-KEM-1024 + ECDH P-384
- **Auth:** ML-DSA-87 + ECDSA P-384

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/crypto_gateway.py
"""
DSMIL Crypto Gateway (Device 80)
PQC encryption/decryption for external communications
"""

import logging
from dsmil_pqc import MLKEMEncryptor, MLKEMDecryptor, MLDSAVerifier

DEVICE_ID = 80
TOKEN_BASE = 0x80F0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoGateway:
    def __init__(self):
        self.kem_decryptor = MLKEMDecryptor()  # ML-KEM-1024
        self.sig_verifier = MLDSAVerifier()    # ML-DSA-87

        logger.info(f"Crypto Gateway initialized (Device {DEVICE_ID})")

    def decrypt_external_message(self, encrypted_payload: bytes, signature: bytes) -> bytes:
        """Decrypt and verify external message"""
        # 1. Verify signature (ML-DSA-87)
        if not self.sig_verifier.verify(encrypted_payload, signature):
            raise ValueError("Invalid signature on external message")

        # 2. Decrypt payload (ML-KEM-1024)
        plaintext = self.kem_decryptor.decrypt(encrypted_payload)

        logger.info("Successfully decrypted and verified external message")
        return plaintext
```

---

## 10. Device 81: External Feed Validator

**Purpose:** Integrity and anomaly checks for external messages.

**Validation Checks:**

1. **Signature Verification:** ML-DSA-87 signature valid
2. **Source Authentication:** Certificate pinning for known external sources
3. **Schema Validation:** Message conforms to VMF/USMTF/Link16 standards
4. **Anomaly Detection:** Statistical outliers (unusual message frequency, size)
5. **Spoofing Detection:** Replay attacks, tampered timestamps

**Rejection Criteria:**

- Invalid signature → REJECT (log to Device 82)
- Unknown source → QUARANTINE (manual review)
- Malformed message → REJECT (parse error)
- Anomalous pattern → FLAG (forward with warning)

---

## 11. Device 82: External Comms Audit Logger

**Purpose:** Compliance logging for all external communications (7-year retention).

**Token IDs:**
- `0x80F6` (STATUS): Log storage usage, retention compliance
- `0x80F7` (CONFIG): Retention policies, audit rules
- `0x80F8` (DATA): Recent audit entries

**Audit Record Format:**

```json
{
  "timestamp": "2025-11-23T14:32:15Z",
  "event_type": "EXTERNAL_MESSAGE_RECEIVED",
  "source": "SIPRNET",
  "message_id": "SIPR-2025-112345",
  "classification": "SECRET",
  "originator": "NSA_SIGINT",
  "destination_device": 14,
  "validated": true,
  "user_accessed": ["analyst_smith", "analyst_jones"],
  "releasability": "REL USA"
}
```

**Compliance Requirements:**

- **DoD 5015.2:** Records Management
- **NIST SP 800-53:** Security and Privacy Controls (AU-2, AU-3, AU-6)
- **7-year retention:** All external comms logged for audit trail

---

## 12. Security & ROE Enforcement

### 12.1 Inbound-Only Policy

**CRITICAL SAFETY RULE:**

External military communications are **intelligence sources ONLY**. DSMIL SHALL NOT:
- Send weapons engagement commands via Link 16 (no J2.5 weapon coordination)
- Transmit operational orders via SIPRNET/JWICS
- Issue kinetic commands based solely on external data

**Air-Gap from NC3:**

- Device 61 (NC3 Integration) is **air-gapped** from Phase 11 devices
- External data can reach L3-L9 for analysis, but L9 Executive decisions remain human-gated
- Any external data used in NC3 context requires explicit review and authorization

### 12.2 Classification Enforcement

**Network-to-Classification Mapping:**

| Network | Classification | DSMIL Layer | Enforced By |
|---------|----------------|-------------|-------------|
| Link 16 | SECRET | L4 | Device 73 TLV |
| SIPRNET | SECRET | L3 | Device 74 TLV |
| JWICS | TOP_SECRET/SCI | L5 | Device 75 TLV |
| SATCOM | SECRET-TS | L5 | Device 76 TLV |
| Coalition | NATO SECRET (ATOMAL) | L6 | Device 77 TLV |

**Cross-Domain Enforcement:**

- Messages tagged with `CLASSIFICATION` TLV at ingress (Device 73-77)
- L3-L9 routing respects classification boundaries (Phase 3 L7 Router policy)
- ATOMAL data requires L6 compartment access (Phase 4 ATOMAL handling)

### 12.3 PQC Transition Plan

**Phase 1 (Current):** Hybrid classical+PQC
- ML-KEM-1024 + ECDH P-384 for key exchange
- ML-DSA-87 + ECDSA P-384 for signatures
- Maintain backwards compatibility with classical-only systems

**Phase 2 (Future):** PQC-only
- Remove ECDH/ECDSA after all external systems upgraded
- ML-KEM-1024 + ML-DSA-87 exclusive
- Quantum-safe end-to-end

---

## 13. Implementation Details

### 13.1 Docker Compose Configuration

```yaml
# /opt/dsmil/docker-compose-phase11.yml
version: '3.8'

services:
  link16-gateway:
    image: dsmil/link16-gateway:1.0
    container_name: dsmil-link16-gateway-73
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=73
      - TERMINAL_ID=DSMIL-J15
      - NETWORK_ID=15
    network_mode: host  # Direct hardware access for Link 16 terminal
    restart: unless-stopped

  siprnet-interface:
    image: dsmil/siprnet-interface:1.0
    container_name: dsmil-siprnet-interface-74
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=74
      - IMAP_SERVER=sipr-imap.disa.mil
    restart: unless-stopped

  jwics-interface:
    image: dsmil/jwics-interface:1.0
    container_name: dsmil-jwics-interface-75
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=75
      - JWICS_FEED_URL=https://jwics-intel-feed.ic.gov
    restart: unless-stopped

  satcom-adapter:
    image: dsmil/satcom-adapter:1.0
    container_name: dsmil-satcom-adapter-76
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=76
      - TERMINAL_TYPE=AEHF
    restart: unless-stopped

  coalition-bridge:
    image: dsmil/coalition-bridge:1.0
    container_name: dsmil-coalition-bridge-77
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=77
      - NETWORKS=BICES,CENTRIXS
    restart: unless-stopped

  protocol-translator:
    image: dsmil/protocol-translator:1.0
    container_name: dsmil-protocol-translator-78
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=78
    restart: unless-stopped

  message-router:
    image: dsmil/message-router:1.0
    container_name: dsmil-message-router-79
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=79
    restart: unless-stopped

  crypto-gateway:
    image: dsmil/crypto-gateway:1.0
    container_name: dsmil-crypto-gateway-80
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /opt/dsmil/pqc-keys:/keys:ro
    environment:
      - DEVICE_ID=80
    restart: unless-stopped

  feed-validator:
    image: dsmil/feed-validator:1.0
    container_name: dsmil-feed-validator-81
    volumes:
      - /var/run/dsmil:/var/run/dsmil
    environment:
      - DEVICE_ID=81
    restart: unless-stopped

  external-audit:
    image: dsmil/external-audit:1.0
    container_name: dsmil-external-audit-82
    volumes:
      - /var/run/dsmil:/var/run/dsmil
      - /var/log/dsmil/audit:/audit
    environment:
      - DEVICE_ID=82
      - RETENTION_YEARS=7
    restart: unless-stopped

networks:
  default:
    name: dsmil-external-dmz
```

### 13.2 Network Architecture (DMZ)

```
┌─────────────────────────────────────────────────────────────┐
│                    External Networks                         │
│  Link 16   SIPRNET   JWICS   SATCOM   Coalition             │
└────┬──────────┬─────────┬──────┬────────────┬───────────────┘
     │          │         │      │            │
     │          │         │      │            │
┌────▼──────────▼─────────▼──────▼────────────▼───────────────┐
│                  DMZ - Phase 11 Devices                      │
│        Firewall, IDS, One-Way Diode (optional)               │
│  Device 73-82: External Comms Gateways                       │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              │ DBE Protocol (Internal)
                              │
┌─────────────────────────────▼────────────────────────────────┐
│              DSMIL Internal Network (L3-L9)                  │
│  Devices 14-62: Ingestion, Analysis, Prediction, etc.       │
└──────────────────────────────────────────────────────────────┘
```

**Firewall Rules:**

- External → DMZ: Allow on specific ports (IMAP 993, HTTPS 443, Link 16 UDP)
- DMZ → Internal: Allow only DBE protocol (UDS sockets)
- Internal → External: **DENY ALL** (inbound-only policy)

---

## 14. Testing & Validation

### 14.1 Unit Tests

```python
#!/usr/bin/env python3
# tests/test_link16_gateway.py
"""
Unit tests for Link 16 Gateway (Device 73)
"""

import unittest
from link16_gateway import Link16Gateway, Link16Track, TrackType

class TestLink16Gateway(unittest.TestCase):

    def setUp(self):
        self.gateway = Link16Gateway()

    def test_track_update(self):
        """Test Link 16 track database update"""
        j2_2_message = {
            "message_type": "J2.2",
            "track_number": 9999,
            "track_type": "AIR",
            "latitude": 40.0,
            "longitude": -75.0,
            "altitude_feet": 30000,
            "speed_knots": 500,
            "heading_degrees": 90,
            "timestamp": time.time()
        }

        self.gateway.receive_j_message(j2_2_message)

        # Verify track in database
        self.assertIn(9999, self.gateway.tracks)
        track = self.gateway.tracks[9999]
        self.assertEqual(track.track_type, TrackType.AIR)
        self.assertEqual(track.altitude_feet, 30000)

    def test_inbound_only(self):
        """Verify no weapons engagement messages sent"""
        # DSMIL should NEVER send J2.5 (weapon coordination)
        # Only J2.0 (initial entry) is permitted

        # Attempt to send J2.5 should fail
        with self.assertRaises(NotImplementedError):
            self.gateway.send_weapon_coordination()

if __name__ == '__main__':
    unittest.main()
```

### 14.2 Integration Tests

```bash
#!/bin/bash
# tests/integration/test_external_comms.sh
# Integration test: Receive and process external messages

set -e

echo "[TEST] Starting external comms integration test..."

# 1. Start all Phase 11 services
docker-compose -f /opt/dsmil/docker-compose-phase11.yml up -d

# 2. Simulate Link 16 track message
echo "[TEST] Simulating Link 16 J2.2 message..."
curl -X POST http://localhost:8080/link16/inject \
  -H "Content-Type: application/json" \
  -d '{
    "message_type": "J2.2",
    "track_number": 12345,
    "track_type": "AIR",
    "latitude": 38.8977,
    "longitude": -77.0365,
    "altitude_feet": 25000
  }'

# 3. Verify track forwarded to L4 (Device 26)
sleep 5
TRACK_COUNT=$(redis-cli --raw GET "device:26:track_count")
if [ "$TRACK_COUNT" -eq 0 ]; then
    echo "[TEST] FAILED: Track not forwarded to L4"
    exit 1
fi

echo "[TEST] SUCCESS: Link 16 track received and forwarded"

# 4. Simulate SIPRNET intelligence report
echo "[TEST] Simulating SIPRNET intel report..."
# Send test email to SIPRNET inbox (mock)

# 5. Verify intel forwarded to L3 (Device 14)
sleep 10
INTEL_COUNT=$(redis-cli --raw GET "device:14:intel_count")
if [ "$INTEL_COUNT" -eq 0 ]; then
    echo "[TEST] FAILED: Intel not forwarded to L3"
    exit 1
fi

echo "[TEST] SUCCESS: SIPRNET intel received and forwarded"

# 6. Verify audit logging (Device 82)
AUDIT_ENTRIES=$(ls /var/log/dsmil/audit/ | wc -l)
if [ "$AUDIT_ENTRIES" -lt 2 ]; then
    echo "[TEST] FAILED: Insufficient audit entries"
    exit 1
fi

echo "[TEST] SUCCESS: Audit logging functional"

# 7. Verify inbound-only policy (no outbound messages)
OUTBOUND_COUNT=$(tcpdump -i any -c 100 -n 'dst net 203.0.113.0/24' 2>/dev/null | wc -l)
if [ "$OUTBOUND_COUNT" -gt 0 ]; then
    echo "[TEST] FAILED: Outbound messages detected (inbound-only policy violated)"
    exit 1
fi

echo "[TEST] SUCCESS: Inbound-only policy enforced"

# 8. Cleanup
docker-compose -f /opt/dsmil/docker-compose-phase11.yml down

echo "[TEST] External comms integration test PASSED"
```

### 14.3 Penetration Testing

**Red Team Scenarios:**

1. **Spoofed Link 16 Message:** Attempt to inject fake track data
   - Expected: Rejected by Device 81 (Feed Validator) due to invalid signature

2. **SIPRNET Phishing:** Send malicious email to SIPRNET inbox
   - Expected: Content filtering at Device 79 (Message Router), flagged for review

3. **Man-in-the-Middle:** Intercept JWICS API traffic
   - Expected: PQC encryption at Device 80 prevents decryption

---

## 15. Exit Criteria

Phase 11 is considered complete when:

- [ ] All 10 devices (73-82) operational and health-check passing
- [ ] Link 16 track data successfully received and displayed in L4 COP
- [ ] SIPRNET intelligence report processed and routed to L3 analysts
- [ ] JWICS national-level intel received and forwarded to L5 (with compartment enforcement)
- [ ] SATCOM message received via Milstar/AEHF and prioritized correctly
- [ ] Coalition message with ATOMAL marking handled per releasability rules
- [ ] Inbound-only policy verified: **zero** outbound commands to external systems
- [ ] PQC crypto validated: ML-KEM-1024 + ML-DSA-87 operational
- [ ] Penetration testing completed with no critical vulnerabilities
- [ ] Audit logging functional with 7-year retention verified
- [ ] Integration with L3-L9 layers tested (external data flowing through pipeline)

---

## 16. Future Enhancements

**Post-Phase 11 Capabilities:**

1. **AI-Powered Message Prioritization:** L7 LLM classifies intel reports by urgency
2. **Federated Coalition Learning:** Distributed ML across NATO partners
3. **Quantum Key Distribution (QKD):** Device 46 (Quantum Integration) for Link 16 crypto
4. **Automated Threat Correlation:** Cross-reference Link 16 tracks with SIGINT/IMINT
5. **Real-Time Language Translation:** Multi-lingual coalition comms (Arabic, Russian, Mandarin)

---

**End of Phase 11 Specification**
