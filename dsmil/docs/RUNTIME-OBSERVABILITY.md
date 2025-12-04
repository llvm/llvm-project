# Runtime Observability Guide

**DSLLVM v1.7+ Runtime Telemetry Export**

## Overview

DSLLVM provides standardized telemetry export for integration with Prometheus, OpenTelemetry, and structured logging systems (ELK/Splunk).

---

## Quick Start

```c
// In source code - automatic telemetry
#include <dsmil_attributes.h>
#include <dsmil_telemetry_export.h>

DSMIL_MISSION_CRITICAL
DSMIL_TELEMETRY_EXPORT("prometheus")
void critical_function(void) {
    // Automatically exports:
    // - Function call count
    // - Execution time histogram
    // - Error rate
    // - Resource usage
}
```

```bash
# Start telemetry collector
dsmil-telemetry-collector --format=prometheus --port=9090

# Export to OpenTelemetry
dsmil-telemetry-collector --format=otel --endpoint=http://otel:4317

# Structured logging
dsmil-telemetry-collector --format=json --output=/var/log/dsmil/telemetry.json
```

---

## Telemetry Types

### Performance Metrics

- **Counters**: Function call counts, event counts
- **Gauges**: Current values (memory usage, active connections)
- **Histograms**: Execution time distributions

### Security Events

- Classification boundary crossings
- Cross-domain gateway usage
- Provenance verification results
- Threat signature matches

### Operational Metrics

- Mission profile activations
- Stealth mode effectiveness
- BFT position updates
- Radio protocol usage

---

## Prometheus Integration

Export metrics in Prometheus format:

```bash
dsmil-telemetry-collector --format=prometheus --port=9090
```

Metrics available at `http://localhost:9090/metrics`

Example metrics:
```
dsmil_function_calls_total{function="critical_function"} 1234
dsmil_execution_time_seconds{function="critical_function"} 0.005
dsmil_memory_bytes{function="critical_function"} 1048576
```

---

## OpenTelemetry Integration

Export to OpenTelemetry collector:

```bash
dsmil-telemetry-collector --format=otel --endpoint=http://otel:4317
```

Supports:
- Traces
- Metrics
- Logs

---

## Structured Logging (JSON)

Export to JSON for ELK/Splunk:

```bash
dsmil-telemetry-collector --format=json --output=/var/log/dsmil/telemetry.json
```

Log format:
```json
{
  "timestamp": 1234567890.123456,
  "metric": "function_calls",
  "type": "counter",
  "value": 1234,
  "labels": {
    "function": "critical_function",
    "layer": "7"
  }
}
```

---

## API Usage

Record custom metrics:

```c
#include <dsmil_telemetry_export.h>

dsmil_telemetry_options_t options = {
    .format = DSMIL_TELEMETRY_PROMETHEUS,
    .enable_performance = true,
    .enable_security = true,
    .enable_operational = true
};

dsmil_telemetry_init(&options);

// Record counter
dsmil_telemetry_record_counter("my_counter", 1, NULL);

// Record gauge
dsmil_telemetry_record_gauge("memory_usage", 1024.0, NULL);

// Record security event
dsmil_telemetry_record_security_event("classification_cross", 5, "{\"from\":\"S\",\"to\":\"C\"}");

dsmil_telemetry_shutdown();
```

---

## Grafana Dashboard

Example Prometheus queries for Grafana:

```promql
# Function call rate
rate(dsmil_function_calls_total[5m])

# Average execution time
avg(dsmil_execution_time_seconds)

# Memory usage
dsmil_memory_bytes

# Security events
dsmil_security_events_total
```

---

## Related Documentation

- **[TELEMETRY-ENFORCEMENT.md](TELEMETRY-ENFORCEMENT.md)**: Telemetry requirements
- **[MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md)**: Mission profile setup

---

**DSLLVM Runtime Observability**: Production-ready telemetry and monitoring integration.
