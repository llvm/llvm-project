## 1. Overview & Objectives

Phase 2F focuses on **high-speed data infrastructure** and **psycholinguistic monitoring** for the DSMIL system. This phase builds on Phase 1's foundation by implementing:

1. **Fast hot-path data fabric** (Redis Streams + tmpfs SQLite)
2. **Unified logging surface** (journald → Loki → SHRINK)
3. **SHRINK integration** as SOC brainstem for operator stress/crisis monitoring
4. **Baseline Layer 8 SOC expansion** with Device 51-58 logical mappings

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Device Count:** 104 devices (Devices 0-103) across 9 operational layers (Layers 2-9)
- **Layer 8 (ENHANCED_SEC):** 8 devices (51-58), 8 GB budget, 80 TOPS theoretical
  - Device 51: Adversarial ML Defense
  - Device 52: Security Analytics
  - Device 53: Cryptographic AI
  - Device 54: Threat Intelligence Fusion
  - Device 55: Behavioral Biometrics
  - Device 56: Secure Enclave Management
  - Device 57: Network Security AI
  - Device 58: SOAR (Security Orchestration)

---

## 2. Fast Data Fabric Architecture

### 2.1 Redis Streams (Event Bus)

**Purpose:** Provide high-speed, persistent pub-sub streams for cross-layer intelligence flows.

**Installation:**

```bash
sudo apt update && sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

**Stream Definitions:**

| Stream Name | Purpose | Producers | Consumers | Retention |
|------------|---------|-----------|-----------|-----------|
| `L3_IN` | Layer 3 inputs | External data ingest (Devices 0-11) | Layer 3 processors (Devices 15-22) | 24h |
| `L3_OUT` | Layer 3 decisions | Layer 3 (Devices 15-22) | Layer 4, Layer 8 SOC | 24h |
| `L4_IN` | Layer 4 inputs | Layer 3, external | Layer 4 (Devices 23-30) | 24h |
| `L4_OUT` | Layer 4 decisions | Layer 4 (Devices 23-30) | Layer 5, Layer 8 SOC | 24h |
| `SOC_EVENTS` | Fused security alerts | Layer 8 SOC Router (Device 52) | Layer 8 workers, Layer 9 | 7d |

**Configuration:**

```conf
# /etc/redis/redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""  # Disable RDB snapshots for performance
appendonly yes
appendfsync everysec
```

**Stream Retention Policy:**

```python
# Executed by SOC Router initialization
import redis
r = redis.Redis()

# Set max length for streams (auto-trim)
r.xtrim("L3_IN", maxlen=100000, approximate=True)
r.xtrim("L3_OUT", maxlen=100000, approximate=True)
r.xtrim("L4_IN", maxlen=100000, approximate=True)
r.xtrim("L4_OUT", maxlen=100000, approximate=True)
r.xtrim("SOC_EVENTS", maxlen=500000, approximate=True)  # 7d retention
```

### 2.2 tmpfs SQLite (Hot-Path State)

**Purpose:** RAM-backed SQL database for real-time state queries without disk I/O.

**Setup:**

```bash
# Create 4 GB RAM disk for hot-path DB
sudo mkdir -p /mnt/dsmil-ram
sudo mount -t tmpfs -o size=4G,mode=0770,uid=dsmil,gid=dsmil tmpfs /mnt/dsmil-ram

# Make persistent across reboots
echo "tmpfs /mnt/dsmil-ram tmpfs size=4G,mode=0770,uid=dsmil,gid=dsmil 0 0" | \
  sudo tee -a /etc/fstab
```

**Schema:**

```sql
-- /opt/dsmil/scripts/init_hotpath_db.sql
CREATE TABLE IF NOT EXISTS raw_events_fast (
  ts          REAL NOT NULL,           -- Unix timestamp with microseconds
  device_id   INTEGER NOT NULL,        -- Device 0-103
  layer       INTEGER NOT NULL,        -- Layer 2-9
  source      TEXT NOT NULL,           -- Data source/sensor
  compartment TEXT NOT NULL,           -- CRYPTO, SIGNALS, NUCLEAR, etc.
  payload     BLOB NOT NULL,           -- Binary event data
  token_id    INTEGER,                 -- 0x8000 + (device_id * 3) + offset
  clearance   INTEGER                  -- 0x02020202 - 0x09090909
);

CREATE TABLE IF NOT EXISTS model_outputs_fast (
  ts          REAL NOT NULL,
  device_id   INTEGER NOT NULL,        -- Source device (0-103)
  layer       INTEGER NOT NULL,        -- Layer 2-9
  model       TEXT NOT NULL,           -- Model name
  input_ref   TEXT,                    -- Reference to input event
  output_json TEXT NOT NULL,           -- JSON result
  score       REAL,                    -- Confidence/risk score
  tops_used   REAL,                    -- TOPS consumed
  latency_ms  REAL                     -- Processing time
);

CREATE TABLE IF NOT EXISTS layer_state (
  layer       INTEGER PRIMARY KEY,     -- Layer 2-9
  active_devices TEXT NOT NULL,        -- JSON array of active device IDs
  memory_used_gb REAL NOT NULL,        -- Current memory consumption
  tops_used   REAL NOT NULL,           -- Current TOPS utilization
  last_update REAL NOT NULL            -- Last state update timestamp
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_raw_events_fast_ts ON raw_events_fast(ts);
CREATE INDEX IF NOT EXISTS idx_raw_events_fast_device ON raw_events_fast(device_id, ts);
CREATE INDEX IF NOT EXISTS idx_raw_events_fast_layer ON raw_events_fast(layer, ts);
CREATE INDEX IF NOT EXISTS idx_model_outputs_fast_layer_ts ON model_outputs_fast(layer, ts);
CREATE INDEX IF NOT EXISTS idx_model_outputs_fast_device_ts ON model_outputs_fast(device_id, ts);
```

**Initialization:**

```bash
sqlite3 /mnt/dsmil-ram/hotpath.db < /opt/dsmil/scripts/init_hotpath_db.sql
```

**Usage Pattern:**

- **Writers:** Layer 3-4 services write fast-path state (events, model outputs, resource usage)
- **Readers:** SOC Router, monitoring dashboards, Layer 8 analytics
- **Archiver:** Background process copies aged data to Postgres every 5 minutes (optional cold storage)

**Memory Budget:** 4 GB allocated, typically uses 2-3 GB for 24h of hot data.

### 2.3 Data Flow Summary

```
External Sensors → Redis L3_IN → Layer 3 (Devices 15-22) → tmpfs SQLite
                                                           ↓
                                    Redis L3_OUT → Layer 4 (Devices 23-30)
                                                 → Layer 8 SOC Router (Device 52)
                                                           ↓
                                    Redis SOC_EVENTS → Layer 8 Workers (Devices 51-58)
                                                     → Layer 9 Command (Devices 59-62)
```

---

## 3. Unified Logging Architecture

### 3.1 journald → Loki → SHRINK Pipeline

**Design Principle:** All DSMIL services log to systemd's journald with standardized identifiers, enabling:
1. Centralized log collection (Loki/Grafana)
2. Real-time psycholinguistic analysis (SHRINK)
3. Audit trail for Layer 9 compliance

### 3.2 DSMIL Service Logging Standards

**systemd Unit Template:**

```ini
# /etc/systemd/system/dsmil-l3.service
[Unit]
Description=DSMIL Layer 3 Realtime Analytics (Devices 15-22)
After=network.target redis-server.service
Requires=redis-server.service

[Service]
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil
Environment="PYTHONUNBUFFERED=1"
Environment="REDIS_URL=redis://localhost:6379/0"
Environment="SQLITE_PATH=/mnt/dsmil-ram/hotpath.db"
Environment="DSMIL_LAYER=3"
Environment="DSMIL_DEVICES=15,16,17,18,19,20,21,22"
Environment="LAYER_MEMORY_BUDGET_GB=6"
Environment="LAYER_TOPS_BUDGET=80"
ExecStart=/opt/dsmil/.venv/bin/python l3_realtime_service.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-l3
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Service Naming Convention:**

| Service | Syslog Identifier | Devices | Layer | Purpose |
|---------|------------------|---------|-------|---------|
| dsmil-l3.service | dsmil-l3 | 15-22 | 3 | SECRET compartmented analytics |
| dsmil-l4.service | dsmil-l4 | 23-30 | 4 | TOP_SECRET mission planning |
| dsmil-l7-router.service | dsmil-l7-router | 43 | 7 | L7 inference routing |
| dsmil-l7-worker-*.service | dsmil-l7-worker-{id} | 44-50 | 7 | L7 model serving |
| dsmil-soc-router.service | dsmil-soc-router | 52 | 8 | SOC event fusion |
| dsmil-soc-advml.service | dsmil-soc-advml | 51 | 8 | Adversarial ML defense |
| dsmil-soc-analytics.service | dsmil-soc-analytics | 52 | 8 | Security analytics |
| dsmil-soc-crypto.service | dsmil-soc-crypto | 53 | 8 | Cryptographic AI |
| dsmil-soc-threatintel.service | dsmil-soc-threatintel | 54 | 8 | Threat intel fusion |

### 3.3 Aggregated DSMIL Log Stream

**Purpose:** Create `/var/log/dsmil.log` for SHRINK to tail all DSMIL activity.

**Implementation:**

```bash
#!/usr/bin/env bash
# /usr/local/bin/journaldsmil-follow.sh

# Follow all dsmil-* services and write to persistent log
journalctl -fu dsmil-l3.service \
           -fu dsmil-l4.service \
           -fu dsmil-l7-router.service \
           -fu dsmil-l7-worker-*.service \
           -fu dsmil-soc-*.service \
           -o short-iso | tee -a /var/log/dsmil.log
```

**systemd Unit:**

```ini
# /etc/systemd/system/journaldsmil.service
[Unit]
Description=Aggregate DSMIL journald logs to /var/log/dsmil.log
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/journaldsmil-follow.sh
Restart=always
StandardOutput=file:/var/log/dsmil-journald.log
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable:**

```bash
sudo chmod +x /usr/local/bin/journaldsmil-follow.sh
sudo systemctl daemon-reload
sudo systemctl enable --now journaldsmil.service
```

**Log Rotation:**

```conf
# /etc/logrotate.d/dsmil
/var/log/dsmil.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 dsmil dsmil
    postrotate
        systemctl reload journaldsmil.service > /dev/null 2>&1 || true
    endscript
}
```

### 3.4 Loki + Promtail Integration

**Promtail Configuration:**

```yaml
# /etc/promtail/config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: dsmil_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: dsmil
          host: dsmil-node-01
          __path__: /var/log/dsmil.log

  - job_name: systemd
    journal:
      max_age: 12h
      labels:
        job: systemd
        host: dsmil-node-01
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal_syslog_identifier']
        regex: 'dsmil-(.*)'
        target_label: 'layer'
```

**Loki Configuration:**

```yaml
# /etc/loki/config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb:
    directory: /var/lib/loki/index
  filesystem:
    directory: /var/lib/loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 720h  # 30 days
```

**Grafana Dashboard Query Examples:**

```logql
# All DSMIL logs from Layer 3
{job="dsmil", layer="l3"}

# SOC events with high severity
{job="dsmil", layer="soc-router"} |= "CRITICAL" or "HIGH"

# Device 47 (primary LLM) inference logs
{job="dsmil", unit="dsmil-l7-worker-47.service"}

# Layer 8 adversarial ML alerts
{job="dsmil", layer="soc-advml"} |= "ALERT"
```

---

## 4. SHRINK Integration (Psycholinguistic Monitoring)

### 4.1 Purpose & Architecture

**SHRINK (Systematic Human Risk Intelligence in Networked Kernels)** provides:
- Real-time psycholinguistic analysis of operator logs
- Operator stress/crisis detection
- Risk metrics for Layer 8 SOC correlation
- Desktop/audio alerts for anomalous operator behavior

**Integration Point:** SHRINK tails `/var/log/dsmil.log` and exposes metrics on `:8500`.

### 4.2 Installation

```bash
# Install SHRINK
cd /opt
sudo git clone https://github.com/SWORDIntel/SHRINK.git
sudo chown -R shrink:shrink SHRINK
cd SHRINK

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm

# Create dedicated user
sudo useradd -r -s /bin/false -d /opt/SHRINK shrink
sudo chown -R shrink:shrink /opt/SHRINK
```

### 4.3 SHRINK Configuration for DSMIL

```yaml
# /opt/SHRINK/config.yaml

# Enhanced monitoring for DSMIL operator activity
enhanced_monitoring:
  enabled: true
  user_id: "DSMIL_OPERATOR"
  session_tracking: true

# Kernel interface (disabled in Phase 2F, enabled in Phase 4)
kernel_interface:
  enabled: false
  dsmil_device_map:
    51: "adversarial_ml_defense"
    52: "security_analytics"
    53: "cryptographic_ai"
    54: "threat_intel_fusion"
    55: "behavioral_biometrics"
    56: "secure_enclave"
    57: "network_security_ai"
    58: "soar"

# Anomaly detection for operator stress/crisis
anomaly_detection:
  enabled: true
  contamination: 0.1          # Assume 10% of logs are anomalous
  z_score_threshold: 3.0      # 3-sigma threshold for alerts
  features:
    - cognitive_load
    - emotional_intensity
    - linguistic_complexity
    - risk_markers

# Alerting channels
alerting:
  enabled_channels:
    - desktop                 # Linux desktop notifications
    - audio                   # TTS warnings
    - prometheus              # Metrics export
  min_severity: MODERATE      # MODERATE | HIGH | CRITICAL

  thresholds:
    acute_stress: 0.7         # Trigger at 70% stress
    crisis_level: 0.8         # Trigger at 80% crisis indicators
    cognitive_overload: 0.75  # Trigger at 75% cognitive load

# Post-quantum cryptography for metrics transport
crypto:
  enabled: true
  quantum_resistant: true
  algorithms:
    kem: "ML-KEM-1024"        # Kyber-1024
    signature: "ML-DSA-87"    # Dilithium5

# Log source configuration
log_source:
  path: "/var/log/dsmil.log"
  format: "journald"
  follow: true
  buffer_size: 8192

# Predictive models for operator behavior
predictive_models:
  enabled: true
  sequence_length: 48         # 48 log entries for context
  prediction_horizon: 6       # Predict 6 entries ahead
  model_path: "/opt/SHRINK/models/lstm_operator_stress.pt"

# Personalization & intervention
personalization:
  triggers:
    enabled: true
    correlation_window: 120   # 2-minute correlation window
  interventions:
    enabled: true
    escalation_policy:
      - level: "MODERATE"
        action: "desktop_notification"
      - level: "HIGH"
        action: "audio_alert + soc_event"
      - level: "CRITICAL"
        action: "audio_alert + soc_event + layer9_notification"

# Metrics export
metrics:
  enabled: true
  port: 8500
  path: "/metrics"
  format: "prometheus"

  # Exported metrics
  exports:
    - "risk_acute_stress"
    - "shrink_crisis_level"
    - "lbi_hyperfocus_density"
    - "cognitive_load_index"
    - "emotional_intensity_score"
    - "linguistic_complexity_index"
    - "anomaly_score"

# REST API for SOC integration
api:
  enabled: true
  port: 8500
  endpoints:
    - "/api/v1/metrics"       # Current metrics snapshot
    - "/api/v1/history"       # Historical trend data
    - "/api/v1/alerts"        # Active alerts
```

### 4.4 systemd Service

```ini
# /etc/systemd/system/shrink-dsmil.service
[Unit]
Description=SHRINK Psycholinguistic & Risk Monitor for DSMIL
After=network.target journaldsmil.service
Requires=journaldsmil.service

[Service]
Type=simple
User=shrink
Group=shrink
WorkingDirectory=/opt/SHRINK

# SHRINK command with all modules
ExecStart=/opt/SHRINK/.venv/bin/shrink \
  --config /opt/SHRINK/config.yaml \
  --modules core,risk,tmi,neuro,cogarch \
  --source /var/log/dsmil.log \
  --enhanced-monitoring \
  --anomaly-detection \
  --real-time-alerts \
  --port 8500 \
  --log-level INFO

# Resource limits (SHRINK is CPU-bound)
CPUQuota=200%                 # Max 2 CPU cores
MemoryLimit=2G                # 2 GB memory limit

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal
SyslogIdentifier=shrink-dsmil

[Install]
WantedBy=multi-user.target
```

**Enable:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now shrink-dsmil.service
```

### 4.5 SHRINK Metrics Exported

**Prometheus Metrics on `:8500/metrics`:**

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|-----------------|
| `risk_acute_stress` | gauge | Acute operator stress level (0.0-1.0) | > 0.7 |
| `shrink_crisis_level` | gauge | Crisis indicator severity (0.0-1.0) | > 0.8 |
| `lbi_hyperfocus_density` | gauge | Cognitive hyperfocus density | > 0.8 |
| `cognitive_load_index` | gauge | Operator cognitive load (0.0-1.0) | > 0.75 |
| `emotional_intensity_score` | gauge | Emotional intensity in logs | > 0.8 |
| `linguistic_complexity_index` | gauge | Text complexity score | > 0.7 |
| `anomaly_score` | gauge | Log anomaly detection score | > 3.0 (z-score) |
| `shrink_alerts_total` | counter | Total alerts generated | N/A |
| `shrink_processing_latency_ms` | histogram | Log processing latency | N/A |

**REST API Endpoints:**

```bash
# Current metrics snapshot (JSON)
curl http://localhost:8500/api/v1/metrics

# Historical trend (last 1 hour)
curl "http://localhost:8500/api/v1/history?window=1h"

# Active alerts
curl http://localhost:8500/api/v1/alerts
```

---

## 5. Layer 8 SOC Expansion (Logical Mappings)

### 5.1 Device Assignments & Responsibilities

**Layer 8 (ENHANCED_SEC) – 8 Devices, 8 GB Budget, 80 TOPS Theoretical:**

| Device ID | Name | Token Base | Purpose | Phase 2F Status | Memory | TOPS |
|-----------|------|-----------|---------|----------------|--------|------|
| **51** | Adversarial ML Defense | 0x8099 | Detect log manipulation, operator anomalies | **Active** (SHRINK integration) | 1.0 GB | 10 |
| **52** | Security Analytics | 0x809C | SOC event aggregation, dashboard | **Active** (SOC Router) | 1.5 GB | 10 |
| **53** | Cryptographic AI | 0x809F | PQC monitoring, key rotation alerts | Stub | 1.0 GB | 10 |
| **54** | Threat Intel Fusion | 0x80A2 | External threat feed correlation | Stub | 1.0 GB | 10 |
| **55** | Behavioral Biometrics | 0x80A5 | Keystroke/mouse behavior analysis | Stub | 0.5 GB | 10 |
| **56** | Secure Enclave Mgmt | 0x80A8 | TPM/HSM monitoring | Stub | 0.5 GB | 10 |
| **57** | Network Security AI | 0x80AB | Network flow anomaly detection | Stub | 1.5 GB | 10 |
| **58** | SOAR | 0x80AE | Security orchestration & response | Stub | 1.0 GB | 10 |

**Token Calculation Example (Device 52):**
- Base: `0x8000 + (52 × 3) = 0x8000 + 156 = 0x809C`
- STATUS: `0x809C + 0 = 0x809C`
- CONFIG: `0x809C + 1 = 0x809D`
- DATA: `0x809C + 2 = 0x809E`

### 5.2 SOC Router Implementation (Device 52)

**Purpose:** Fuse Layer 3/4 outputs + SHRINK metrics → `SOC_EVENTS` stream for Layer 8 workers.

**Architecture:**

```
Redis L3_OUT ──┐
               ├──> SOC Router (Device 52) ──> Redis SOC_EVENTS ──> Layer 8 Workers
Redis L4_OUT ──┤                                                  └──> Layer 9 Command
               │
SHRINK :8500 ──┘
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/soc_router.py
"""
DSMIL SOC Router (Device 52 - Security Analytics)
Fuses Layer 3/4 outputs + SHRINK metrics → SOC_EVENTS stream
"""

import time
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

import redis
import requests

# Constants
REDIS_URL = "redis://localhost:6379/0"
SHRINK_METRICS_URL = "http://localhost:8500/api/v1/metrics"
DEVICE_ID = 52
LAYER = 8
TOKEN_BASE = 0x809C

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SOC-ROUTER] [Device-52] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class SOCRouter:
    def __init__(self):
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        self.last_l3_id = "0-0"
        self.last_l4_id = "0-0"
        logger.info(f"SOC Router initialized (Device {DEVICE_ID}, Token 0x{TOKEN_BASE:04X})")

    def pull_shrink_metrics(self) -> Dict[str, float]:
        """Pull current SHRINK metrics from REST API"""
        try:
            resp = requests.get(SHRINK_METRICS_URL, timeout=0.5)
            resp.raise_for_status()
            metrics = resp.json()
            return {
                "risk_acute_stress": metrics.get("risk_acute_stress", 0.0),
                "crisis_level": metrics.get("shrink_crisis_level", 0.0),
                "cognitive_load": metrics.get("cognitive_load_index", 0.0),
                "anomaly_score": metrics.get("anomaly_score", 0.0),
            }
        except Exception as e:
            logger.warning(f"Failed to pull SHRINK metrics: {e}")
            return {
                "risk_acute_stress": 0.0,
                "crisis_level": 0.0,
                "cognitive_load": 0.0,
                "anomaly_score": 0.0,
            }

    def process_l3_events(self, messages: List, shrink_metrics: Dict[str, float]):
        """Process Layer 3 output events"""
        for msg_id, fields in messages:
            try:
                event = {k.decode(): v.decode() for k, v in fields.items()}

                # Create SOC event
                soc_event = {
                    "event_id": msg_id.decode(),
                    "ts": time.time(),
                    "src_layer": 3,
                    "src_device": event.get("device_id", "unknown"),
                    "decision": event.get("decision", ""),
                    "score": float(event.get("score", 0.0)),
                    "compartment": event.get("compartment", ""),

                    # SHRINK correlation
                    "shrink_risk": shrink_metrics["risk_acute_stress"],
                    "shrink_crisis": shrink_metrics["crisis_level"],
                    "shrink_cognitive_load": shrink_metrics["cognitive_load"],
                    "shrink_anomaly": shrink_metrics["anomaly_score"],

                    # Alert logic
                    "alert_level": self._calculate_alert_level(
                        float(event.get("score", 0.0)),
                        shrink_metrics
                    ),

                    # Metadata
                    "device_52_processed": True,
                    "token_id": f"0x{TOKEN_BASE:04X}",
                }

                # Publish to SOC_EVENTS
                self.redis.xadd(
                    "SOC_EVENTS",
                    {k: json.dumps(v) if not isinstance(v, (str, bytes)) else v
                     for k, v in soc_event.items()}
                )

                if soc_event["alert_level"] != "INFO":
                    logger.info(
                        f"Alert: {soc_event['alert_level']} | "
                        f"Layer 3 Decision: {soc_event['decision'][:50]} | "
                        f"SHRINK Risk: {shrink_metrics['risk_acute_stress']:.2f}"
                    )

                self.last_l3_id = msg_id

            except Exception as e:
                logger.error(f"Failed to process L3 event: {e}")

    def process_l4_events(self, messages: List, shrink_metrics: Dict[str, float]):
        """Process Layer 4 output events (similar to L3)"""
        for msg_id, fields in messages:
            try:
                event = {k.decode(): v.decode() for k, v in fields.items()}

                soc_event = {
                    "event_id": msg_id.decode(),
                    "ts": time.time(),
                    "src_layer": 4,
                    "src_device": event.get("device_id", "unknown"),
                    "decision": event.get("decision", ""),
                    "score": float(event.get("score", 0.0)),
                    "classification": event.get("classification", "TOP_SECRET"),

                    # SHRINK correlation
                    "shrink_risk": shrink_metrics["risk_acute_stress"],
                    "shrink_crisis": shrink_metrics["crisis_level"],

                    "alert_level": self._calculate_alert_level(
                        float(event.get("score", 0.0)),
                        shrink_metrics
                    ),

                    "device_52_processed": True,
                    "token_id": f"0x{TOKEN_BASE:04X}",
                }

                self.redis.xadd("SOC_EVENTS",
                    {k: json.dumps(v) if not isinstance(v, (str, bytes)) else v
                     for k, v in soc_event.items()})

                if soc_event["alert_level"] != "INFO":
                    logger.info(
                        f"Alert: {soc_event['alert_level']} | "
                        f"Layer 4 Decision | "
                        f"SHRINK Crisis: {shrink_metrics['crisis_level']:.2f}"
                    )

                self.last_l4_id = msg_id

            except Exception as e:
                logger.error(f"Failed to process L4 event: {e}")

    def _calculate_alert_level(self, decision_score: float,
                               shrink_metrics: Dict[str, float]) -> str:
        """Calculate alert severity based on decision score + SHRINK metrics"""
        # High risk if either decision OR operator is stressed
        if decision_score > 0.9 or shrink_metrics["crisis_level"] > 0.8:
            return "CRITICAL"
        elif decision_score > 0.75 or shrink_metrics["risk_acute_stress"] > 0.7:
            return "HIGH"
        elif decision_score > 0.5 or shrink_metrics["anomaly_score"] > 3.0:
            return "MODERATE"
        else:
            return "INFO"

    def run(self):
        """Main event loop"""
        logger.info("SOC Router started, monitoring L3_OUT and L4_OUT...")

        while True:
            try:
                # Pull SHRINK metrics once per iteration
                shrink_metrics = self.pull_shrink_metrics()

                # Read from L3_OUT
                l3_streams = self.redis.xread(
                    {"L3_OUT": self.last_l3_id},
                    block=500,  # 500ms timeout
                    count=10
                )

                for stream_name, messages in l3_streams:
                    if stream_name == b"L3_OUT":
                        self.process_l3_events(messages, shrink_metrics)

                # Read from L4_OUT
                l4_streams = self.redis.xread(
                    {"L4_OUT": self.last_l4_id},
                    block=500,
                    count=10
                )

                for stream_name, messages in l4_streams:
                    if stream_name == b"L4_OUT":
                        self.process_l4_events(messages, shrink_metrics)

                # Brief sleep to prevent tight loop
                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("SOC Router shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    router = SOCRouter()
    router.run()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-soc-router.service
[Unit]
Description=DSMIL SOC Router (Device 52 - Security Analytics)
After=redis-server.service shrink-dsmil.service
Requires=redis-server.service shrink-dsmil.service

[Service]
Type=simple
User=dsmil
Group=dsmil
WorkingDirectory=/opt/dsmil

Environment="PYTHONUNBUFFERED=1"
Environment="REDIS_URL=redis://localhost:6379/0"
Environment="DSMIL_DEVICE_ID=52"
Environment="DSMIL_LAYER=8"

ExecStart=/opt/dsmil/.venv/bin/python soc_router.py

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsmil-soc-router

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dsmil-soc-router.service
```

### 5.3 Device 51 – Adversarial ML Defense (Stub)

**Purpose:** Monitor for log manipulation, model poisoning attempts, operator behavior anomalies.

**Phase 2F Implementation:** Stub service that logs SHRINK anomaly scores above threshold.

```python
# /opt/dsmil/soc_advml_stub.py
"""
Device 51 - Adversarial ML Defense (Stub for Phase 2F)
Monitors SHRINK anomaly scores and logs alerts
"""

import time
import logging
import requests

SHRINK_URL = "http://localhost:8500/api/v1/metrics"
ANOMALY_THRESHOLD = 3.0  # z-score threshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_loop():
    logger.info("Device 51 (Adversarial ML Defense) monitoring started")
    while True:
        try:
            resp = requests.get(SHRINK_URL, timeout=1.0)
            metrics = resp.json()

            anomaly = metrics.get("anomaly_score", 0.0)
            if anomaly > ANOMALY_THRESHOLD:
                logger.warning(
                    f"[DEVICE-51] ANOMALY DETECTED | "
                    f"Score: {anomaly:.2f} | "
                    f"Threshold: {ANOMALY_THRESHOLD}"
                )

            time.sleep(5)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_loop()
```

**systemd Unit:**

```ini
# /etc/systemd/system/dsmil-soc-advml.service
[Unit]
Description=DSMIL Device 51 - Adversarial ML Defense (Stub)
After=shrink-dsmil.service

[Service]
User=dsmil
Group=dsmil
ExecStart=/opt/dsmil/.venv/bin/python soc_advml_stub.py
SyslogIdentifier=dsmil-soc-advml
Restart=always

[Install]
WantedBy=multi-user.target
```

### 5.4 Devices 53-58 – Future Layer 8 Workers

**Phase 2F Status:** Stub services with systemd units, no active AI models yet.

**Activation Timeline:**
- **Phase 3 (Weeks 7-10):** Activate Device 53 (Cryptographic AI) for PQC monitoring
- **Phase 4 (Weeks 11-13):** Activate Devices 54-58 (Threat Intel, Biometrics, Network AI, SOAR)

**Stub Template:**

```bash
# Create stub services for Devices 53-58
for device_id in {53..58}; do
  cat > /opt/dsmil/soc_stub_${device_id}.py << EOF
import time, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Device ${device_id} stub service started")
while True:
    time.sleep(60)
EOF

  cat > /etc/systemd/system/dsmil-soc-device${device_id}.service << EOF
[Unit]
Description=DSMIL Device ${device_id} (Layer 8 Stub)
After=network.target

[Service]
User=dsmil
ExecStart=/opt/dsmil/.venv/bin/python soc_stub_${device_id}.py
SyslogIdentifier=dsmil-soc-device${device_id}
Restart=always

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable dsmil-soc-device${device_id}.service
done
```

---

## 6. Phase 2F Validation & Success Criteria

### 6.1 Checklist

Phase 2F is complete when:

- [x] **Redis Streams operational:**
  - `L3_IN`, `L3_OUT`, `L4_IN`, `L4_OUT`, `SOC_EVENTS` streams created
  - Stream retention policies configured (24h/7d)
  - Verified with `redis-cli XINFO STREAM SOC_EVENTS`

- [x] **tmpfs SQLite hot-path DB:**
  - Mounted at `/mnt/dsmil-ram` (4 GB tmpfs)
  - Schema created with all tables + indexes
  - L3/L4 services writing events/outputs
  - Verified with `sqlite3 /mnt/dsmil-ram/hotpath.db "SELECT COUNT(*) FROM raw_events_fast"`

- [x] **journald logging standardized:**
  - All DSMIL services use `SyslogIdentifier=dsmil-*`
  - Logs visible with `journalctl -u dsmil-*.service`
  - `/var/log/dsmil.log` populated by `journaldsmil.service`

- [x] **Loki + Promtail integration:**
  - Promtail scraping journald + `/var/log/dsmil.log`
  - Loki ingesting logs, accessible via Grafana
  - Sample query works: `{job="dsmil", layer="l3"}`

- [x] **SHRINK monitoring active:**
  - `shrink-dsmil.service` running on `:8500`
  - Metrics endpoint responding: `curl http://localhost:8500/metrics`
  - REST API returning JSON: `curl http://localhost:8500/api/v1/metrics`
  - Prometheus scraping SHRINK metrics

- [x] **SOC Router operational (Device 52):**
  - `dsmil-soc-router.service` running and processing events
  - Reading from `L3_OUT` and `L4_OUT`
  - Writing fused events to `SOC_EVENTS`
  - SHRINK metrics integrated in SOC events
  - Alert levels calculated correctly

- [x] **Device 51 (Adversarial ML) active:**
  - `dsmil-soc-advml.service` running
  - Monitoring SHRINK anomaly scores
  - Logging alerts above threshold

- [x] **Devices 53-58 stubbed:**
  - Systemd units created and enabled
  - Services start without errors
  - Placeholder logging confirms readiness for Phase 3-4

### 6.2 Validation Commands

```bash
# Verify Redis Streams
redis-cli XINFO STREAM SOC_EVENTS
redis-cli XLEN L3_OUT

# Verify tmpfs DB
sqlite3 /mnt/dsmil-ram/hotpath.db "SELECT COUNT(*) FROM raw_events_fast"
df -h /mnt/dsmil-ram

# Verify journald logging
journalctl -u dsmil-l3.service --since "5 minutes ago"
tail -f /var/log/dsmil.log

# Verify SHRINK
curl http://localhost:8500/api/v1/metrics | jq .
systemctl status shrink-dsmil.service

# Verify SOC Router
systemctl status dsmil-soc-router.service
journalctl -u dsmil-soc-router.service -f

# Verify Layer 8 services
systemctl list-units "dsmil-soc-*"
```

### 6.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Redis write latency | < 1ms p99 | `redis-cli --latency` |
| tmpfs SQLite write | < 0.5ms p99 | Custom benchmark script |
| SHRINK processing latency | < 50ms per log line | `shrink_processing_latency_ms` histogram |
| SOC Router throughput | > 10,000 events/sec | Custom load test |
| Log aggregation lag | < 5 seconds | Compare journald timestamp vs Loki ingestion |

### 6.4 Resource Utilization

**Expected Memory Usage:**
- Redis: 512 MB (streams + overhead)
- tmpfs SQLite: 2-3 GB (4 GB allocated)
- SHRINK: 1.5-2.0 GB (NLP models + buffers)
- SOC Router: 200 MB
- Layer 8 stubs: 50 MB each × 8 = 400 MB
- **Total:** ~5-6 GB

**Expected CPU Usage:**
- SHRINK: 1.5-2.0 CPU cores (psycholinguistic processing)
- SOC Router: 0.2-0.5 CPU cores
- Redis: 0.1-0.3 CPU cores
- Layer 8 stubs: negligible

**Expected Disk I/O:**
- Primarily journald writes (~10-50 MB/min depending on log verbosity)
- Loki ingestion: ~5-20 MB/min
- tmpfs: no disk I/O (RAM-backed)

---

## 7. Next Phase Preview (Phase 3)

Phase 3 will build on Phase 2F infrastructure by:

1. **Layer 7 LLM Activation (Device 47):**
   - Deploy LLaMA-7B INT8 on Device 47 (20 GB allocation)
   - Integrate L7 router with SOC Router for LLM-assisted triage

2. **Device 53 (Cryptographic AI) Activation:**
   - Monitor PQC key rotations (ML-KEM-1024, ML-DSA-87)
   - Alert on downgrade attacks or crypto anomalies

3. **SHRINK-LLM Integration:**
   - Use Device 47 LLM to generate natural language summaries of SHRINK alerts
   - Implement "SOC Copilot" endpoint: `/v1/llm/soc-copilot`

4. **Advanced Analytics on tmpfs:**
   - Real-time correlation queries (join `raw_events_fast` + `model_outputs_fast`)
   - Implement Device 52 analytics dashboard

---

## 8. Document Metadata

**Version History:**
- **v1.0 (2024-Q4):** Initial Phase 1F spec with Redis/SHRINK/SOC
- **v2.0 (2025-11-23):** Aligned with v3.1 Comprehensive Plan
  - Updated hardware specs (48.2 TOPS, 64 GB memory)
  - Added device token IDs (0x8000-based system)
  - Clarified Layer 8 device responsibilities (51-58)
  - Updated memory/TOPS budgets per v3.1
  - Added clearance level references
  - Expanded SHRINK configuration with PQC
  - Detailed SOC Router implementation (Device 52)

**Dependencies:**
- Redis >= 7.0
- SQLite >= 3.38
- Python >= 3.10
- SHRINK (latest from GitHub)
- Loki + Promtail >= 2.9
- systemd >= 249

**References:**
- `00_MASTER_PLAN_OVERVIEW_CORRECTED.md (v3.1)`
- `01_HARDWARE_INTEGRATION_LAYER_DETAILED.md (v3.1)`
- `05_LAYER_SPECIFIC_DEPLOYMENTS.md (v1.0)`
- `06_CROSS_LAYER_INTELLIGENCE_FLOWS.md (v1.0)`
- `07_IMPLEMENTATION_ROADMAP.md (v1.0)`
- `Phase1.md (v2.0)`

**Contact:**
For questions or issues with Phase 2F implementation, contact DSMIL DevOps team.

---

**END OF PHASE 2F SPECIFICATION**
