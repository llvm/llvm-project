# DSMIL Python SDK Technical Specifications  
## Comprehensive Client Library for Python Applications

**Version:** 2.0.1  
**Classification:** RESTRICTED  
**Date:** 2025-01-15  

---

## Overview

The DSMIL Python SDK provides comprehensive, pythonic access to the DSMIL control system with support for both synchronous and asynchronous operations. Designed for data analysis, automation scripts, machine learning applications, and integration with scientific computing workflows.

## Installation

```bash
# Install from PyPI
pip install dsmil-control-client==2.0.1

# Install with optional dependencies
pip install dsmil-control-client[async,ml,visualization]==2.0.1

# Install from source (development)
git clone https://github.com/dsmil/python-sdk.git
cd python-sdk
pip install -e .
```

### Dependencies

**Core Dependencies:**
- `requests>=2.28.0` - HTTP client
- `websockets>=10.4` - WebSocket support
- `pydantic>=1.10.0` - Data validation
- `python-dateutil>=2.8.2` - Date/time handling
- `cryptography>=3.4.0` - Security features

**Optional Dependencies:**
- `aiohttp>=3.8.0` - Async HTTP client (async extra)
- `numpy>=1.21.0` - Numerical computing (ml extra) 
- `pandas>=1.4.0` - Data analysis (ml extra)
- `matplotlib>=3.5.0` - Plotting (visualization extra)
- `plotly>=5.8.0` - Interactive plots (visualization extra)

## Quick Start

```python
import asyncio
from dsmil_client import DSMILClient

async def main():
    # Create client
    client = DSMILClient(
        base_url="https://dsmil-control.mil",
        api_version="2.0"
    )
    
    # Authenticate
    auth_result = await client.authenticate(
        username="operator",
        password="secure_password",
        client_type="python"
    )
    
    if not auth_result.success:
        print(f"Authentication failed: {auth_result.error}")
        return
    
    # Read device status
    status = await client.read_device(0x8000, register="STATUS")
    print(f"Device status: 0x{status.data:08X}")
    
    # Bulk read multiple devices
    device_ids = [0x8000, 0x8001, 0x8002]
    results = await client.bulk_read(device_ids, register="STATUS")
    
    for result in results:
        if result.success:
            print(f"Device 0x{result.device_id:04X}: 0x{result.data:08X}")
        else:
            print(f"Device 0x{result.device_id:04X}: Error - {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core API Reference

### 1. Client Initialization

```python
from dsmil_client import DSMILClient, ClientConfig

# Basic initialization
client = DSMILClient("https://dsmil-control.mil")

# Advanced configuration
config = ClientConfig(
    base_url="https://dsmil-control.mil",
    api_version="2.0",
    timeout=30.0,
    max_retries=3,
    verify_ssl=True,
    client_cert_path="/path/to/cert.pem",
    user_agent="MyApp/1.0"
)

client = DSMILClient(config=config)

# Environment-based configuration
import os
client = DSMILClient.from_environment()  # Uses DSMIL_* env vars
```

### 2. Authentication

```python
# Password authentication
auth_result = await client.authenticate(
    username="operator", 
    password="secure_password",
    client_type="python"
)

# MFA authentication
auth_result = await client.authenticate_mfa(
    username="operator",
    password="secure_password", 
    mfa_token="123456"
)

# Certificate authentication
auth_result = await client.authenticate_certificate(
    cert_path="/path/to/cert.p12",
    cert_password="cert_password"
)

# Token-based authentication (for service accounts)
auth_result = await client.authenticate_token("service_token_here")

# Check authentication status
if auth_result.success:
    print(f"Logged in as {auth_result.user.username}")
    print(f"Clearance: {auth_result.user.clearance_level}")
    print(f"Authorized devices: {len(auth_result.user.authorized_devices)}")
else:
    print(f"Authentication failed: {auth_result.error}")
    if auth_result.requires_mfa:
        print("MFA required")
```

### 3. System Information

```python
# Get system status
status = await client.get_system_status()
print(f"System status: {status.overall_status}")
print(f"Active devices: {status.device_summary.active_devices}")
print(f"Quarantined devices: {status.device_summary.quarantined_devices}")

# Get API capabilities
capabilities = await client.get_capabilities()
print(f"Supported operations: {capabilities.supported_operations}")
print(f"Rate limits: {capabilities.rate_limits}")

# Health check
health = await client.health_check()
print(f"System healthy: {health.is_healthy}")
```

### 4. Device Management

```python
# List devices with filtering
devices = await client.list_devices(
    include_quarantined=False,
    device_group=0,  # Filter by group
    risk_level="HIGH",  # Filter by risk level
    status="ACTIVE",  # Filter by status
    limit=50  # Pagination
)

for device in devices:
    print(f"Device {device.device_id:04X}: {device.device_name}")
    print(f"  Risk: {device.risk_level}, Active: {device.is_active}")

# Get detailed device info
device_info = await client.get_device(0x8000)
print(f"Device capabilities: {device_info.capabilities}")
print(f"Hardware info: {device_info.hardware_info}")
print(f"Performance: {device_info.performance_metrics}")

# Get device history
history = await client.get_device_history(
    device_id=0x8000,
    start_date="2025-01-01",
    end_date="2025-01-15",
    operation_type="READ"
)

print(f"Found {len(history.operations)} historical operations")
```

### 5. Device Operations

#### Single Device Operations

```python
from dsmil_client import Register, OperationType

# Read operations
status = await client.read_device(0x8000, register=Register.STATUS)
config = await client.read_device(0x8001, register=Register.CONFIG, offset=0x10, length=4)

# Write operations  
write_result = await client.write_device(
    device_id=0x8002,
    register=Register.CONFIG,
    data=0x12345678,
    justification="Configuration update for performance optimization"
)

# Complex configuration
config_result = await client.configure_device(
    device_id=0x8003,
    config_data={
        "threshold": 1024,
        "mode": "continuous", 
        "calibration": True,
        "sampling_rate": 1000
    },
    justification="System optimization"
)

# Device control
activate_result = await client.activate_device(0x8004)
deactivate_result = await client.deactivate_device(0x8004)
reset_result = await client.reset_device(0x8005)
```

#### Bulk Operations

```python
# Bulk read
device_ids = list(range(0x8000, 0x8010))  # First 16 devices
bulk_result = await client.bulk_read(
    device_ids=device_ids,
    register=Register.STATUS,
    execution_mode="parallel",  # or "sequential"
    max_concurrency=5,
    timeout=30.0
)

print(f"Bulk operation completed: {bulk_result.summary.successful}/{bulk_result.summary.total}")

for result in bulk_result.results:
    if result.success:
        print(f"Device 0x{result.device_id:04X}: 0x{result.data:08X}")
    else:
        print(f"Device 0x{result.device_id:04X}: {result.error}")

# Bulk write
write_data = {
    0x8000: 0x11111111,
    0x8001: 0x22222222,
    0x8002: 0x33333333
}

bulk_write_result = await client.bulk_write(
    write_data=write_data,
    register=Register.CONFIG,
    justification="Bulk configuration update"
)

# Mixed bulk operations
from dsmil_client import BulkOperation

operations = [
    BulkOperation(device_id=0x8000, operation="read", register=Register.STATUS),
    BulkOperation(device_id=0x8001, operation="read", register=Register.TEMP),
    BulkOperation(device_id=0x8002, operation="write", register=Register.CONFIG, data=0x12345678)
]

mixed_result = await client.bulk_execute(operations)
```

#### Streaming Operations

```python
# Real-time device monitoring
async def monitor_devices():
    device_ids = [0x8000, 0x8001, 0x8002]
    
    async for update in client.stream_devices(
        device_ids=device_ids,
        registers=[Register.STATUS, Register.TEMP],
        interval_seconds=1.0
    ):
        print(f"Device 0x{update.device_id:04X} - {update.register}: 0x{update.data:08X}")
        
        # Process real-time data
        if update.register == Register.TEMP and update.data > 0x80000000:
            print(f"WARNING: High temperature on device 0x{update.device_id:04X}")

# Server-sent events streaming
async def continuous_monitoring():
    stream = await client.create_device_stream(
        device_ids=[0x8000, 0x8001, 0x8002],
        interval_ms=500,
        duration_seconds=3600  # 1 hour
    )
    
    async for batch in stream:
        for device_data in batch.devices:
            process_telemetry_data(device_data)
```

### 6. WebSocket Real-Time Communication

```python
# Create WebSocket connection
ws_client = await client.create_websocket()

# Subscribe to device updates
await ws_client.subscribe_device_updates(
    device_ids=[0x8000, 0x8001, 0x8002],
    callback=handle_device_update
)

# Subscribe to system events
await ws_client.subscribe_system_events(callback=handle_system_event)

# Subscribe with filters
await ws_client.subscribe_advanced(
    subscription="device_updates",
    filters={
        "device_ids": [0x8000, 0x8001],
        "operation_types": ["READ", "WRITE"],
        "risk_levels": ["HIGH", "CRITICAL"]
    },
    rate_limit="MAX_1_PER_SECOND"
)

def handle_device_update(update):
    print(f"Device update: 0x{update.device_id:04X} = 0x{update.data:08X}")

def handle_system_event(event):
    if event.type == "EMERGENCY_STOP":
        print(f"EMERGENCY: {event.message}")
        handle_emergency_stop(event)

# Keep connection alive
await ws_client.run_forever()
```

### 7. Emergency Controls

```python
# Trigger emergency stop
emergency_result = await client.trigger_emergency_stop(
    justification="Security breach detected in sector 7",
    scope="ALL",  # or "DEVICE_GROUP" or "SINGLE_DEVICE" 
    target_devices=[0x8009, 0x800A],  # if scope is specific
    notify_all_clients=True
)

# Check emergency status
emergency_status = await client.get_emergency_status()
if emergency_status.is_active:
    print(f"Emergency stop active: {emergency_status.reason}")
    print(f"Activated by: {emergency_status.triggered_by}")

# Release emergency stop (requires high clearance)
if emergency_status.is_active:
    release_result = await client.release_emergency_stop(
        justification="Threat neutralized, systems verified secure"
    )
```

## Advanced Features

### 8. Error Handling & Resilience

```python
from dsmil_client import DSMILError, DeviceError, AuthenticationError, RateLimitError

# Configure retry policy
client.configure_retry(
    max_attempts=5,
    base_delay=0.1,  # seconds
    max_delay=5.0,
    backoff_multiplier=2.0,
    jitter=True
)

# Comprehensive error handling
try:
    result = await client.read_device(0x8000, register=Register.STATUS)
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Re-authenticate
    await client.authenticate(username, password, client_type)
except RateLimitError as e:
    print(f"Rate limited: {e.retry_after} seconds")
    await asyncio.sleep(e.retry_after)
except DeviceError as e:
    print(f"Device error: {e.device_id} - {e.error_code}")
    if e.device_id in quarantined_devices:
        print("Device is quarantined")
except DSMILError as e:
    print(f"General DSMIL error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Circuit breaker pattern
from dsmil_client import CircuitBreakerConfig

circuit_breaker = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3
)

client.configure_circuit_breaker(circuit_breaker)

# Custom error handler
def error_handler(error):
    if isinstance(error, RateLimitError):
        # Implement backoff strategy
        implement_backoff(error.retry_after)
    elif isinstance(error, DeviceError) and error.error_code == "QUARANTINED":
        # Log security event
        log_security_event(error)

client.set_error_handler(error_handler)
```

### 9. Performance Optimization

```python
# Connection pooling
client.configure_connection_pool(
    max_connections=20,
    keepalive=True,
    keepalive_timeout=300
)

# Request session reuse
client.configure_session(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)

# Compression
client.enable_compression(True)

# Response caching for read operations
client.enable_cache(
    cache_ttl=30,  # seconds
    cache_size=1000  # max entries
)

# Batch multiple operations
async with client.batch_context() as batch:
    batch.add_read(0x8000, Register.STATUS)
    batch.add_read(0x8001, Register.TEMP)  
    batch.add_write(0x8002, Register.CONFIG, 0x12345678)
    
    results = await batch.execute()

# Performance monitoring
metrics = client.get_performance_metrics()
print(f"Average latency: {metrics.avg_latency:.2f}ms")
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"Operations/sec: {metrics.ops_per_second:.1f}")
```

### 10. Data Analysis Integration

```python
import pandas as pd
import numpy as np
from dsmil_client.analysis import DeviceDataAnalyzer

# Export historical data to pandas DataFrame
analyzer = DeviceDataAnalyzer(client)

df = await analyzer.export_device_history(
    device_ids=[0x8000, 0x8001, 0x8002],
    start_date="2025-01-01",
    end_date="2025-01-15",
    registers=[Register.STATUS, Register.TEMP, Register.VOLTAGE]
)

print(df.head())
print(df.describe())

# Real-time data streaming to pandas
stream_analyzer = analyzer.create_streaming_analyzer(
    device_ids=[0x8000, 0x8001],
    window_size=1000,  # Keep last 1000 data points
    update_interval=1.0  # Update every second
)

# Get live statistics
stats = await stream_analyzer.get_live_statistics()
print(f"Mean temperature: {stats['TEMP'].mean():.2f}")
print(f"Voltage std dev: {stats['VOLTAGE'].std():.4f}")

# Anomaly detection
anomalies = analyzer.detect_anomalies(
    device_id=0x8000,
    register=Register.TEMP,
    method="isolation_forest",
    threshold=0.1
)

for anomaly in anomalies:
    print(f"Anomaly detected at {anomaly.timestamp}: value={anomaly.value}")
```

### 11. Machine Learning Integration

```python
from dsmil_client.ml import DevicePredictor, FailurePredictionModel

# Train predictive model
predictor = DevicePredictor(client)

# Collect training data
training_data = await predictor.collect_training_data(
    device_ids=list(range(0x8000, 0x8010)),
    features=[Register.TEMP, Register.VOLTAGE, Register.STATUS],
    target=Register.ERROR,
    days_history=30
)

# Train model
model = FailurePredictionModel()
model.train(training_data)

# Make predictions
device_id = 0x8000
current_data = await client.read_device_multiple(
    device_id, 
    registers=[Register.TEMP, Register.VOLTAGE, Register.STATUS]
)

failure_probability = model.predict_failure_probability(current_data)
print(f"Failure probability for device 0x{device_id:04X}: {failure_probability:.2%}")

if failure_probability > 0.8:
    print("HIGH RISK: Recommend immediate maintenance")
    await client.write_device(
        device_id, 
        Register.CONFIG, 
        0x80000000,  # Maintenance mode
        justification="Predictive maintenance triggered"
    )
```

### 12. Visualization Support

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dsmil_client.visualization import DeviceDashboard, RealTimePlotter

# Create real-time dashboard
dashboard = DeviceDashboard(client)

# Add device monitoring widgets
dashboard.add_device_status_widget([0x8000, 0x8001, 0x8002])
dashboard.add_temperature_trend_widget(0x8000, hours=24)
dashboard.add_performance_metrics_widget()

# Launch dashboard
await dashboard.serve(port=8080)  # Serves at http://localhost:8080

# Real-time plotting
plotter = RealTimePlotter()

# Monitor device temperature in real-time
async def plot_temperature():
    async for update in client.stream_devices([0x8000], [Register.TEMP]):
        plotter.add_point(
            series="Device_8000_Temp",
            timestamp=update.timestamp,
            value=update.data
        )
        plotter.update()

# Static analysis plots
device_history = await analyzer.export_device_history(
    device_ids=[0x8000],
    start_date="2025-01-01",
    end_date="2025-01-15",
    registers=[Register.TEMP]
)

plt.figure(figsize=(12, 6))
plt.plot(device_history.timestamp, device_history.TEMP)
plt.title("Device 0x8000 Temperature Over Time")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()

# Interactive Plotly dashboard
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=device_history.timestamp,
    y=device_history.TEMP,
    mode='lines',
    name='Temperature'
))
fig.update_layout(title="Interactive Device Temperature")
fig.show()
```

## Configuration Management

### 13. Environment Configuration

```python
# .env file support
from dsmil_client import load_dotenv

load_dotenv()  # Loads .env file

# Environment variables:
# DSMIL_BASE_URL=https://dsmil-control.mil
# DSMIL_API_VERSION=2.0
# DSMIL_USERNAME=operator
# DSMIL_PASSWORD=secure_password
# DSMIL_CLIENT_CERT=/path/to/cert.pem
# DSMIL_VERIFY_SSL=true
# DSMIL_TIMEOUT=30
# DSMIL_MAX_RETRIES=3

client = DSMILClient.from_environment()

# Configuration file support
from dsmil_client import DSMILConfig

config = DSMILConfig.from_file("dsmil_config.yaml")
client = DSMILClient(config=config)

# dsmil_config.yaml:
# base_url: https://dsmil-control.mil
# api_version: "2.0"
# authentication:
#   username: operator
#   password: secure_password
# connection:
#   timeout: 30
#   max_retries: 3
#   verify_ssl: true
# performance:
#   max_connections: 20
#   enable_compression: true
#   cache_ttl: 30
```

### 14. Logging Configuration

```python
import logging
from dsmil_client import configure_logging

# Configure DSMIL SDK logging
configure_logging(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_path="dsmil_client.log"
)

# Custom logger
logger = logging.getLogger("dsmil_client")
logger.info("Starting DSMIL client application")

# Performance logging
client.enable_performance_logging(True)

# Security event logging
client.enable_security_logging(True)
```

## Testing & Debugging

### 15. Unit Testing Support

```python
import pytest
from unittest.mock import AsyncMock
from dsmil_client import DSMILClient
from dsmil_client.testing import MockDSMILClient, DeviceSimulator

# Mock client for testing
@pytest.fixture
async def mock_client():
    client = MockDSMILClient()
    # Pre-configure mock responses
    client.mock_authenticate(success=True, clearance="SECRET")
    client.mock_device_read(0x8000, Register.STATUS, 0x12345678)
    return client

@pytest.mark.asyncio
async def test_device_read(mock_client):
    result = await mock_client.read_device(0x8000, Register.STATUS)
    assert result.success
    assert result.data == 0x12345678

# Device simulator for integration testing
simulator = DeviceSimulator()
simulator.add_device(0x8000, initial_status=0x00000001)
simulator.add_device(0x8001, initial_status=0x00000001)

# Start simulator server
await simulator.start_server(port=8899)

# Test against simulator
test_client = DSMILClient("http://localhost:8899")
```

### 16. Debug Mode

```python
# Enable debug mode
client.enable_debug_mode(True)

# Debug information
debug_info = client.get_debug_info()
print(f"Active connections: {debug_info.active_connections}")
print(f"Request queue size: {debug_info.request_queue_size}")
print(f"Cache hit rate: {debug_info.cache_hit_rate:.2%}")

# Request/response logging
client.enable_request_logging(True)

# Performance profiling
with client.performance_profile() as profiler:
    result = await client.read_device(0x8000, Register.STATUS)

print(f"Operation took: {profiler.duration:.2f}ms")
print(f"Network time: {profiler.network_time:.2f}ms")
print(f"Processing time: {profiler.processing_time:.2f}ms")
```

## Examples

### 17. Device Monitoring Script

```python
#!/usr/bin/env python3
"""
DSMIL Device Monitoring Script
Monitors critical devices and alerts on anomalies
"""
import asyncio
import logging
from datetime import datetime
from dsmil_client import DSMILClient, Register

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceMonitor:
    def __init__(self, base_url: str):
        self.client = DSMILClient(base_url)
        self.critical_devices = [0x8000, 0x8001, 0x8002]  # Master controllers
        self.running = False
        
    async def start_monitoring(self):
        """Start the monitoring process"""
        # Authenticate
        auth_result = await self.client.authenticate(
            username="monitor",
            password="monitor_password",
            client_type="python"
        )
        
        if not auth_result.success:
            logger.error(f"Authentication failed: {auth_result.error}")
            return
            
        logger.info(f"Authenticated as {auth_result.user.username}")
        
        self.running = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_device_health()),
            asyncio.create_task(self.monitor_system_events())
        ]
        
        await asyncio.gather(*tasks)
    
    async def monitor_device_health(self):
        """Monitor device health via periodic polling"""
        while self.running:
            try:
                # Bulk read all critical devices
                results = await self.client.bulk_read(
                    device_ids=self.critical_devices,
                    register=Register.STATUS,
                    timeout=10.0
                )
                
                for result in results.results:
                    if not result.success:
                        logger.warning(
                            f"Device 0x{result.device_id:04X} unreachable: {result.error}"
                        )
                        await self.alert_device_offline(result.device_id)
                    elif result.data & 0x80000000:  # Error bit set
                        logger.error(
                            f"Device 0x{result.device_id:04X} reports error: 0x{result.data:08X}"
                        )
                        await self.alert_device_error(result.device_id, result.data)
                    else:
                        logger.debug(f"Device 0x{result.device_id:04X} healthy")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def monitor_system_events(self):
        """Monitor system events via WebSocket"""
        try:
            ws_client = await self.client.create_websocket()
            
            await ws_client.subscribe_system_events(
                callback=self.handle_system_event
            )
            
            await ws_client.subscribe_device_updates(
                device_ids=self.critical_devices,
                callback=self.handle_device_update
            )
            
            await ws_client.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket monitoring error: {e}")
    
    def handle_system_event(self, event):
        """Handle system events"""
        logger.info(f"System event: {event.type} - {event.message}")
        
        if event.type == "EMERGENCY_STOP":
            logger.critical(f"EMERGENCY STOP: {event.message}")
            self.alert_emergency_stop(event)
        elif event.severity == "HIGH":
            self.alert_high_severity_event(event)
    
    def handle_device_update(self, update):
        """Handle real-time device updates"""
        logger.debug(f"Device 0x{update.device_id:04X} update: 0x{update.data:08X}")
        
        # Check for critical conditions
        if update.data & 0x80000000:  # Error condition
            logger.warning(f"Device 0x{update.device_id:04X} error detected")
    
    async def alert_device_offline(self, device_id):
        """Alert when device goes offline"""
        message = f"ALERT: Device 0x{device_id:04X} is offline"
        logger.error(message)
        # In production, send to alerting system
        await self.send_alert(message, severity="HIGH")
    
    async def alert_device_error(self, device_id, error_code):
        """Alert when device reports error"""
        message = f"ALERT: Device 0x{device_id:04X} error code: 0x{error_code:08X}"
        logger.error(message)
        await self.send_alert(message, severity="HIGH")
    
    def alert_emergency_stop(self, event):
        """Alert on emergency stop"""
        message = f"CRITICAL: Emergency stop activated - {event.message}"
        logger.critical(message)
        # Immediate notification required
        asyncio.create_task(self.send_alert(message, severity="CRITICAL"))
    
    def alert_high_severity_event(self, event):
        """Alert on high severity events"""
        message = f"HIGH SEVERITY: {event.type} - {event.message}"
        logger.error(message)
        asyncio.create_task(self.send_alert(message, severity="HIGH"))
    
    async def send_alert(self, message, severity="INFO"):
        """Send alert to monitoring system"""
        # In production, integrate with alerting system
        # (PagerDuty, Slack, email, etc.)
        print(f"[{datetime.now()}] {severity}: {message}")

async def main():
    monitor = DeviceMonitor("https://dsmil-control.mil")
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        monitor.running = False

if __name__ == "__main__":
    asyncio.run(main())
```

### 18. Data Collection & Analysis

```python
#!/usr/bin/env python3
"""
DSMIL Data Collection and Analysis
Collects device data and performs statistical analysis
"""
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dsmil_client import DSMILClient, Register
from dsmil_client.analysis import DeviceDataAnalyzer

class DataCollectionSystem:
    def __init__(self, base_url: str, output_dir: str = "./data"):
        self.client = DSMILClient(base_url)
        self.analyzer = DeviceDataAnalyzer(self.client)
        self.output_dir = output_dir
        self.device_groups = {
            "controllers": list(range(0x8000, 0x8004)),    # Master controllers
            "sensors": list(range(0x8004, 0x8010)),        # Sensor arrays
            "actuators": list(range(0x8010, 0x8020))       # Actuator systems
        }
        
    async def collect_historical_data(self, days: int = 30):
        """Collect historical data for analysis"""
        print(f"Collecting {days} days of historical data...")
        
        await self.client.authenticate("analyst", "analyst_password", "python")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        
        for group_name, device_ids in self.device_groups.items():
            print(f"Collecting data for {group_name} group...")
            
            group_data = await self.analyzer.export_device_history(
                device_ids=device_ids,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                registers=[Register.STATUS, Register.TEMP, Register.VOLTAGE]
            )
            
            group_data['device_group'] = group_name
            all_data.append(group_data)
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Save raw data
        output_file = f"{self.output_dir}/device_data_{days}days.csv"
        combined_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        return combined_data
    
    def analyze_data(self, data: pd.DataFrame):
        """Perform statistical analysis on collected data"""
        print("Performing statistical analysis...")
        
        # Basic statistics
        stats = data.groupby(['device_group', 'device_id']).agg({
            'STATUS': ['mean', 'std', 'min', 'max'],
            'TEMP': ['mean', 'std', 'min', 'max'],
            'VOLTAGE': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print("\nDevice Statistics by Group:")
        print(stats)
        
        # Correlation analysis
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        # Anomaly detection
        self.detect_anomalies(data)
        
        # Generate visualizations
        self.create_visualizations(data)
        
        return stats, correlation_matrix
    
    def detect_anomalies(self, data: pd.DataFrame):
        """Detect anomalies in device data"""
        print("\nDetecting anomalies...")
        
        for group_name in self.device_groups.keys():
            group_data = data[data['device_group'] == group_name]
            
            # Temperature anomalies (values outside 3 standard deviations)
            temp_mean = group_data['TEMP'].mean()
            temp_std = group_data['TEMP'].std()
            temp_threshold = 3 * temp_std
            
            temp_anomalies = group_data[
                abs(group_data['TEMP'] - temp_mean) > temp_threshold
            ]
            
            if not temp_anomalies.empty:
                print(f"\n{group_name} temperature anomalies:")
                for _, anomaly in temp_anomalies.iterrows():
                    print(f"  Device 0x{anomaly['device_id']:04X}: "
                         f"{anomaly['TEMP']:.2f} at {anomaly['timestamp']}")
            
            # Voltage anomalies
            voltage_mean = group_data['VOLTAGE'].mean()
            voltage_std = group_data['VOLTAGE'].std()
            voltage_threshold = 3 * voltage_std
            
            voltage_anomalies = group_data[
                abs(group_data['VOLTAGE'] - voltage_mean) > voltage_threshold
            ]
            
            if not voltage_anomalies.empty:
                print(f"\n{group_name} voltage anomalies:")
                for _, anomaly in voltage_anomalies.iterrows():
                    print(f"  Device 0x{anomaly['device_id']:04X}: "
                         f"{anomaly['VOLTAGE']:.2f} at {anomaly['timestamp']}")
    
    def create_visualizations(self, data: pd.DataFrame):
        """Create data visualizations"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature distribution by device group
        data.boxplot(column='TEMP', by='device_group', ax=axes[0, 0])
        axes[0, 0].set_title('Temperature Distribution by Device Group')
        axes[0, 0].set_xlabel('Device Group')
        axes[0, 0].set_ylabel('Temperature')
        
        # Voltage distribution by device group  
        data.boxplot(column='VOLTAGE', by='device_group', ax=axes[0, 1])
        axes[0, 1].set_title('Voltage Distribution by Device Group')
        axes[0, 1].set_xlabel('Device Group')
        axes[0, 1].set_ylabel('Voltage')
        
        # Temperature vs Voltage correlation
        for group in self.device_groups.keys():
            group_data = data[data['device_group'] == group]
            axes[1, 0].scatter(group_data['TEMP'], group_data['VOLTAGE'], 
                             label=group, alpha=0.6)
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Voltage')
        axes[1, 0].set_title('Temperature vs Voltage Correlation')
        axes[1, 0].legend()
        
        # Status code frequency
        status_counts = data['STATUS'].value_counts().head(10)
        axes[1, 1].bar(range(len(status_counts)), status_counts.values)
        axes[1, 1].set_title('Most Common Status Codes')
        axes[1, 1].set_xlabel('Status Code Rank')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/device_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {self.output_dir}/device_analysis.png")
    
    async def real_time_monitoring(self, duration_minutes: int = 60):
        """Real-time data monitoring and analysis"""
        print(f"Starting real-time monitoring for {duration_minutes} minutes...")
        
        # Create real-time data buffer
        data_buffer = []
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Monitor subset of devices for real-time analysis
        monitored_devices = [0x8000, 0x8001, 0x8002, 0x8003]
        
        async for update in self.client.stream_devices(
            device_ids=monitored_devices,
            registers=[Register.STATUS, Register.TEMP, Register.VOLTAGE],
            interval_seconds=1.0
        ):
            # Add to buffer
            data_point = {
                'timestamp': update.timestamp,
                'device_id': update.device_id,
                'register': update.register.name,
                'value': update.data
            }
            data_buffer.append(data_point)
            
            # Real-time analysis every 60 seconds
            if len(data_buffer) % 240 == 0:  # 4 devices * 3 registers * 20 updates
                await self.analyze_real_time_buffer(data_buffer[-240:])
            
            # Check if monitoring period is over
            if datetime.now() > end_time:
                break
        
        # Final analysis
        df = pd.DataFrame(data_buffer)
        df.to_csv(f"{self.output_dir}/realtime_data.csv", index=False)
        print(f"Real-time data saved to {self.output_dir}/realtime_data.csv")
        
        return df
    
    async def analyze_real_time_buffer(self, buffer_data):
        """Analyze recent real-time data"""
        df = pd.DataFrame(buffer_data)
        
        # Pivot data for analysis
        pivot_df = df.pivot_table(
            index=['timestamp', 'device_id'], 
            columns='register', 
            values='value'
        ).reset_index()
        
        # Check for recent anomalies
        for device_id in pivot_df['device_id'].unique():
            device_data = pivot_df[pivot_df['device_id'] == device_id]
            
            if 'TEMP' in device_data.columns:
                recent_temp = device_data['TEMP'].iloc[-5:].mean()  # Last 5 readings
                if recent_temp > 80:  # Temperature threshold
                    print(f"WARNING: High temperature on device 0x{device_id:04X}: {recent_temp:.1f}")
            
            if 'VOLTAGE' in device_data.columns:
                recent_voltage = device_data['VOLTAGE'].iloc[-5:].mean()
                if recent_voltage < 10 or recent_voltage > 50:  # Voltage thresholds
                    print(f"WARNING: Voltage out of range on device 0x{device_id:04X}: {recent_voltage:.1f}")

async def main():
    collector = DataCollectionSystem("https://dsmil-control.mil")
    
    # Collect and analyze historical data
    historical_data = await collector.collect_historical_data(days=7)
    stats, correlations = collector.analyze_data(historical_data)
    
    # Real-time monitoring
    # realtime_data = await collector.real_time_monitoring(duration_minutes=30)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Conclusion

The DSMIL Python SDK provides comprehensive, pythonic access to the DSMIL control system with extensive support for data analysis, machine learning integration, and real-time monitoring. The SDK is designed for flexibility and ease of use while maintaining the security and reliability requirements of military-grade systems.

Key features include:
- Async/await support for high performance
- Comprehensive error handling and resilience
- Integration with scientific Python ecosystem (pandas, numpy, matplotlib)
- Real-time streaming and WebSocket support
- Built-in data analysis and visualization tools
- Extensive configuration and testing support

---

**Document Classification**: RESTRICTED  
**Review Date**: 2025-04-15  
**Next Version**: 2.1 (Enhanced ML capabilities)