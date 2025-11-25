#!/usr/bin/env python3
"""
TPM2 Comprehensive Monitoring Dashboard
Real-time monitoring and visualization for TPM2 compatibility layer

Author: TPM2 Monitoring Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import signal
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status levels"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"

@dataclass
class Metric:
    """Performance metric"""
    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: float
    labels: Dict[str, str]

@dataclass
class SystemHealth:
    """System health information"""
    status: SystemStatus
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    temperature_celsius: Optional[float]
    uptime_seconds: float
    load_average: Tuple[float, float, float]

@dataclass
class ServiceStatus:
    """Service status information"""
    name: str
    status: SystemStatus
    is_active: bool
    is_enabled: bool
    memory_mb: float
    cpu_percent: float
    uptime_seconds: float
    restart_count: int

@dataclass
class AccelerationMetrics:
    """Hardware acceleration metrics"""
    type: str
    status: SystemStatus
    utilization_percent: float
    throughput_ops_sec: float
    response_time_ms: float
    error_rate_percent: float
    temperature_celsius: Optional[float]
    power_watts: Optional[float]

@dataclass
class SecurityMetrics:
    """Security-related metrics"""
    successful_authentications: int
    failed_authentications: int
    security_violations: int
    token_validations: int
    authorization_denials: int
    audit_events_total: int

class TPM2MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for TPM2 compatibility layer
    Provides real-time metrics, health monitoring, and alerting
    """

    def __init__(self, config_path: str = "/etc/military-tpm/monitoring.json"):
        """Initialize monitoring dashboard"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.metrics_cache = {}
        self.health_history = []
        self.alert_history = []
        self.monitoring_thread = None
        self.web_server_thread = None

        # Monitoring state
        self.last_check_time = 0
        self.check_interval = self.config.get("check_interval_seconds", 30)
        self.metrics_retention_hours = self.config.get("metrics_retention_hours", 24)

        # Alert thresholds
        self.alert_thresholds = self.config.get("alert_thresholds", {})

        # Services to monitor
        self.monitored_services = [
            "military-tpm2.service",
            "military-tpm-health.service",
            "military-tpm-audit.service"
        ]

        logger.info("TPM2 Monitoring Dashboard initialized")

    def start(self):
        """Start monitoring dashboard"""
        logger.info("Starting TPM2 monitoring dashboard...")
        self.running = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # Start web server if enabled
        if self.config.get("web_interface", {}).get("enabled", False):
            self._start_web_server()

        logger.info("TPM2 monitoring dashboard started")

    def stop(self):
        """Stop monitoring dashboard"""
        logger.info("Stopping TPM2 monitoring dashboard...")
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        if self.web_server_thread:
            self.web_server_thread.join(timeout=5)

        logger.info("TPM2 monitoring dashboard stopped")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": time.time(),
            "system_health": asdict(self._get_system_health()),
            "service_status": [asdict(status) for status in self._get_service_status()],
            "acceleration_metrics": [asdict(metrics) for metrics in self._get_acceleration_metrics()],
            "security_metrics": asdict(self._get_security_metrics()),
            "performance_metrics": self._get_performance_metrics(),
            "recent_alerts": self._get_recent_alerts(),
            "configuration": self.config
        }

    def export_metrics(self, format: str = "json", output_path: Optional[str] = None) -> str:
        """Export metrics to file"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"/var/log/military-tpm/metrics_{timestamp}.{format}"

        dashboard_data = self.get_dashboard_data()

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
        elif format == "prometheus":
            self._export_prometheus_metrics(dashboard_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Metrics exported to {output_path}")
        return output_path

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        system_health = self._get_system_health()
        service_statuses = self._get_service_status()
        acceleration_metrics = self._get_acceleration_metrics()

        # Determine overall status
        all_statuses = [system_health.status]
        all_statuses.extend([service.status for service in service_statuses])
        all_statuses.extend([accel.status for accel in acceleration_metrics])

        if any(status == SystemStatus.FAILED for status in all_statuses):
            overall_status = SystemStatus.FAILED
        elif any(status == SystemStatus.CRITICAL for status in all_statuses):
            overall_status = SystemStatus.CRITICAL
        elif any(status == SystemStatus.WARNING for status in all_statuses):
            overall_status = SystemStatus.WARNING
        else:
            overall_status = SystemStatus.HEALTHY

        return {
            "overall_status": overall_status.value,
            "system_health": system_health.status.value,
            "services_healthy": sum(1 for s in service_statuses if s.status == SystemStatus.HEALTHY),
            "services_total": len(service_statuses),
            "acceleration_healthy": sum(1 for a in acceleration_metrics if a.status == SystemStatus.HEALTHY),
            "acceleration_total": len(acceleration_metrics),
            "last_check": self.last_check_time,
            "uptime_seconds": system_health.uptime_seconds
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded monitoring configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            "enabled": True,
            "check_interval_seconds": 30,
            "metrics_retention_hours": 24,
            "alert_thresholds": {
                "cpu_usage_percent": 80,
                "memory_usage_percent": 85,
                "response_time_ms": 1000,
                "error_rate_percent": 5,
                "temperature_celsius": 80
            },
            "health_checks": [
                {"name": "tpm_device_access", "interval": 60},
                {"name": "me_communication", "interval": 120},
                {"name": "token_validation", "interval": 300},
                {"name": "acceleration_performance", "interval": 600}
            ],
            "web_interface": {
                "enabled": False,
                "host": "localhost",
                "port": 8080
            }
        }

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                start_time = time.time()

                # Collect all metrics
                self._collect_metrics()

                # Check alert conditions
                self._check_alerts()

                # Clean up old data
                self._cleanup_old_data()

                # Record last check time
                self.last_check_time = time.time()

                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error

    def _collect_metrics(self):
        """Collect all monitoring metrics"""
        timestamp = time.time()

        # Collect system health
        system_health = self._get_system_health()
        self._store_metric("system.cpu_usage", system_health.cpu_usage_percent, "percent", timestamp)
        self._store_metric("system.memory_usage", system_health.memory_usage_percent, "percent", timestamp)
        self._store_metric("system.disk_usage", system_health.disk_usage_percent, "percent", timestamp)

        if system_health.temperature_celsius:
            self._store_metric("system.temperature", system_health.temperature_celsius, "celsius", timestamp)

        # Collect service metrics
        for service_status in self._get_service_status():
            service_name = service_status.name.replace('.service', '')
            self._store_metric(f"service.{service_name}.cpu_usage", service_status.cpu_percent, "percent", timestamp)
            self._store_metric(f"service.{service_name}.memory_usage", service_status.memory_mb, "mb", timestamp)
            self._store_metric(f"service.{service_name}.uptime", service_status.uptime_seconds, "seconds", timestamp)

        # Collect acceleration metrics
        for accel_metrics in self._get_acceleration_metrics():
            accel_type = accel_metrics.type
            self._store_metric(f"acceleration.{accel_type}.utilization", accel_metrics.utilization_percent, "percent", timestamp)
            self._store_metric(f"acceleration.{accel_type}.throughput", accel_metrics.throughput_ops_sec, "ops/sec", timestamp)
            self._store_metric(f"acceleration.{accel_type}.response_time", accel_metrics.response_time_ms, "ms", timestamp)
            self._store_metric(f"acceleration.{accel_type}.error_rate", accel_metrics.error_rate_percent, "percent", timestamp)

    def _store_metric(self, name: str, value: float, unit: str, timestamp: float):
        """Store metric in cache"""
        if name not in self.metrics_cache:
            self.metrics_cache[name] = []

        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            unit=unit,
            timestamp=timestamp,
            labels={}
        )

        self.metrics_cache[name].append(metric)

        # Keep only recent metrics
        retention_seconds = self.metrics_retention_hours * 3600
        cutoff_time = timestamp - retention_seconds
        self.metrics_cache[name] = [
            m for m in self.metrics_cache[name] if m.timestamp > cutoff_time
        ]

    def _get_system_health(self) -> SystemHealth:
        """Get system health information"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100

            # Temperature
            temperature = self._get_cpu_temperature()

            # Uptime
            uptime = time.time() - psutil.boot_time()

            # Load average
            load_avg = os.getloadavg()

            # Determine status
            status = SystemStatus.HEALTHY
            if cpu_usage > self.alert_thresholds.get("cpu_usage_percent", 80):
                status = SystemStatus.WARNING
            if memory_usage > self.alert_thresholds.get("memory_usage_percent", 85):
                status = SystemStatus.CRITICAL
            if temperature and temperature > self.alert_thresholds.get("temperature_celsius", 80):
                status = SystemStatus.CRITICAL

            return SystemHealth(
                status=status,
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage,
                disk_usage_percent=disk_usage,
                temperature_celsius=temperature,
                uptime_seconds=uptime,
                load_average=load_avg
            )

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                status=SystemStatus.FAILED,
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                temperature_celsius=None,
                uptime_seconds=0,
                load_average=(0, 0, 0)
            )

    def _get_service_status(self) -> List[ServiceStatus]:
        """Get status of monitored services"""
        service_statuses = []

        for service_name in self.monitored_services:
            try:
                # Get service status
                result = subprocess.run(
                    ['systemctl', 'is-active', service_name],
                    capture_output=True, text=True
                )
                is_active = result.stdout.strip() == 'active'

                result = subprocess.run(
                    ['systemctl', 'is-enabled', service_name],
                    capture_output=True, text=True
                )
                is_enabled = result.stdout.strip() == 'enabled'

                # Get process information
                memory_mb = 0
                cpu_percent = 0
                uptime_seconds = 0
                restart_count = 0

                try:
                    # Get service PID
                    result = subprocess.run(
                        ['systemctl', 'show', service_name, '--property=MainPID'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        pid_line = result.stdout.strip()
                        if '=' in pid_line:
                            pid = int(pid_line.split('=')[1])
                            if pid > 0:
                                process = psutil.Process(pid)
                                memory_mb = process.memory_info().rss / (1024 * 1024)
                                cpu_percent = process.cpu_percent()
                                uptime_seconds = time.time() - process.create_time()

                    # Get restart count
                    result = subprocess.run(
                        ['systemctl', 'show', service_name, '--property=NRestarts'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        restart_line = result.stdout.strip()
                        if '=' in restart_line:
                            restart_count = int(restart_line.split('=')[1])

                except (psutil.NoSuchProcess, ValueError):
                    pass

                # Determine status
                if is_active and is_enabled:
                    status = SystemStatus.HEALTHY
                elif is_active:
                    status = SystemStatus.WARNING
                else:
                    status = SystemStatus.CRITICAL

                service_status = ServiceStatus(
                    name=service_name,
                    status=status,
                    is_active=is_active,
                    is_enabled=is_enabled,
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    uptime_seconds=uptime_seconds,
                    restart_count=restart_count
                )

                service_statuses.append(service_status)

            except Exception as e:
                logger.error(f"Error getting status for service {service_name}: {e}")
                service_statuses.append(ServiceStatus(
                    name=service_name,
                    status=SystemStatus.FAILED,
                    is_active=False,
                    is_enabled=False,
                    memory_mb=0,
                    cpu_percent=0,
                    uptime_seconds=0,
                    restart_count=0
                ))

        return service_statuses

    def _get_acceleration_metrics(self) -> List[AccelerationMetrics]:
        """Get hardware acceleration metrics"""
        acceleration_metrics = []

        # NPU metrics
        if self._check_npu_availability():
            npu_metrics = self._get_npu_metrics()
            acceleration_metrics.append(npu_metrics)

        # GNA metrics
        if self._check_gna_availability():
            gna_metrics = self._get_gna_metrics()
            acceleration_metrics.append(gna_metrics)

        # CPU optimized metrics
        cpu_metrics = self._get_cpu_optimized_metrics()
        acceleration_metrics.append(cpu_metrics)

        return acceleration_metrics

    def _get_npu_metrics(self) -> AccelerationMetrics:
        """Get NPU acceleration metrics"""
        try:
            # Simulate NPU metrics (in practice, would query actual NPU APIs)
            utilization = 45.0  # Placeholder
            throughput = 1200.0  # ops/sec
            response_time = 5.0  # ms
            error_rate = 0.1  # percent

            status = SystemStatus.HEALTHY
            if error_rate > self.alert_thresholds.get("error_rate_percent", 5):
                status = SystemStatus.WARNING

            return AccelerationMetrics(
                type="npu",
                status=status,
                utilization_percent=utilization,
                throughput_ops_sec=throughput,
                response_time_ms=response_time,
                error_rate_percent=error_rate,
                temperature_celsius=None,
                power_watts=None
            )

        except Exception as e:
            logger.error(f"Error getting NPU metrics: {e}")
            return AccelerationMetrics(
                type="npu",
                status=SystemStatus.FAILED,
                utilization_percent=0,
                throughput_ops_sec=0,
                response_time_ms=0,
                error_rate_percent=100,
                temperature_celsius=None,
                power_watts=None
            )

    def _get_gna_metrics(self) -> AccelerationMetrics:
        """Get GNA acceleration metrics"""
        try:
            # Simulate GNA metrics
            utilization = 30.0
            throughput = 800.0
            response_time = 8.0
            error_rate = 0.2

            status = SystemStatus.HEALTHY
            if error_rate > self.alert_thresholds.get("error_rate_percent", 5):
                status = SystemStatus.WARNING

            return AccelerationMetrics(
                type="gna",
                status=status,
                utilization_percent=utilization,
                throughput_ops_sec=throughput,
                response_time_ms=response_time,
                error_rate_percent=error_rate,
                temperature_celsius=None,
                power_watts=None
            )

        except Exception as e:
            logger.error(f"Error getting GNA metrics: {e}")
            return AccelerationMetrics(
                type="gna",
                status=SystemStatus.FAILED,
                utilization_percent=0,
                throughput_ops_sec=0,
                response_time_ms=0,
                error_rate_percent=100,
                temperature_celsius=None,
                power_watts=None
            )

    def _get_cpu_optimized_metrics(self) -> AccelerationMetrics:
        """Get CPU optimized acceleration metrics"""
        try:
            # Get actual CPU metrics
            cpu_usage = psutil.cpu_percent()

            # Simulate performance metrics
            utilization = cpu_usage
            throughput = 500.0
            response_time = 15.0
            error_rate = 0.0

            status = SystemStatus.HEALTHY
            if cpu_usage > 80:
                status = SystemStatus.WARNING

            return AccelerationMetrics(
                type="cpu_optimized",
                status=status,
                utilization_percent=utilization,
                throughput_ops_sec=throughput,
                response_time_ms=response_time,
                error_rate_percent=error_rate,
                temperature_celsius=self._get_cpu_temperature(),
                power_watts=None
            )

        except Exception as e:
            logger.error(f"Error getting CPU optimized metrics: {e}")
            return AccelerationMetrics(
                type="cpu_optimized",
                status=SystemStatus.FAILED,
                utilization_percent=0,
                throughput_ops_sec=0,
                response_time_ms=0,
                error_rate_percent=100,
                temperature_celsius=None,
                power_watts=None
            )

    def _get_security_metrics(self) -> SecurityMetrics:
        """Get security-related metrics"""
        try:
            # These would typically be read from audit logs
            # For now, using placeholder values
            return SecurityMetrics(
                successful_authentications=150,
                failed_authentications=2,
                security_violations=0,
                token_validations=75,
                authorization_denials=1,
                audit_events_total=500
            )

        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return SecurityMetrics(
                successful_authentications=0,
                failed_authentications=0,
                security_violations=0,
                token_validations=0,
                authorization_denials=0,
                audit_events_total=0
            )

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        metrics_summary = {}

        for metric_name, metric_list in self.metrics_cache.items():
            if metric_list:
                latest_metric = metric_list[-1]
                values = [m.value for m in metric_list[-10:]]  # Last 10 values

                metrics_summary[metric_name] = {
                    "current": latest_metric.value,
                    "unit": latest_metric.unit,
                    "average_10": sum(values) / len(values),
                    "min_10": min(values),
                    "max_10": max(values),
                    "timestamp": latest_metric.timestamp
                }

        return metrics_summary

    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        # Return last 10 alerts
        return self.alert_history[-10:]

    def _check_npu_availability(self) -> bool:
        """Check if NPU is available"""
        npu_devices = ['/dev/intel_npu', '/dev/npu0', '/dev/accel/accel0']
        return any(os.path.exists(device) for device in npu_devices)

    def _check_gna_availability(self) -> bool:
        """Check if GNA is available"""
        return os.path.exists('/dev/gna0')

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            temperatures = psutil.sensors_temperatures()
            if 'coretemp' in temperatures:
                return temperatures['coretemp'][0].current
            elif 'cpu_thermal' in temperatures:
                return temperatures['cpu_thermal'][0].current
        except Exception:
            pass
        return None

    def _check_alerts(self):
        """Check for alert conditions"""
        current_time = time.time()

        # Check system health alerts
        system_health = self._get_system_health()
        if system_health.cpu_usage_percent > self.alert_thresholds.get("cpu_usage_percent", 80):
            self._create_alert("HIGH_CPU_USAGE", f"CPU usage: {system_health.cpu_usage_percent:.1f}%", "warning")

        if system_health.memory_usage_percent > self.alert_thresholds.get("memory_usage_percent", 85):
            self._create_alert("HIGH_MEMORY_USAGE", f"Memory usage: {system_health.memory_usage_percent:.1f}%", "critical")

        # Check service alerts
        for service_status in self._get_service_status():
            if not service_status.is_active:
                self._create_alert("SERVICE_DOWN", f"Service {service_status.name} is not active", "critical")

    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create an alert"""
        alert = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "acknowledged": False
        }

        self.alert_history.append(alert)
        logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")

        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

    def _cleanup_old_data(self):
        """Clean up old data"""
        current_time = time.time()
        retention_seconds = self.metrics_retention_hours * 3600
        cutoff_time = current_time - retention_seconds

        # Clean up metrics cache
        for metric_name in list(self.metrics_cache.keys()):
            self.metrics_cache[metric_name] = [
                m for m in self.metrics_cache[metric_name] if m.timestamp > cutoff_time
            ]

            if not self.metrics_cache[metric_name]:
                del self.metrics_cache[metric_name]

        # Clean up health history
        self.health_history = [
            h for h in self.health_history if h.get("timestamp", 0) > cutoff_time
        ]

    def _start_web_server(self):
        """Start web interface server"""
        # This would start a web server for the dashboard
        # For now, just log that it would be started
        host = self.config.get("web_interface", {}).get("host", "localhost")
        port = self.config.get("web_interface", {}).get("port", 8080)
        logger.info(f"Web interface would be available at http://{host}:{port}")

    def _export_prometheus_metrics(self, data: Dict[str, Any], output_path: str):
        """Export metrics in Prometheus format"""
        with open(output_path, 'w') as f:
            # Write system metrics
            system_health = data["system_health"]
            f.write(f"# TYPE system_cpu_usage gauge\n")
            f.write(f"system_cpu_usage {system_health['cpu_usage_percent']}\n")
            f.write(f"# TYPE system_memory_usage gauge\n")
            f.write(f"system_memory_usage {system_health['memory_usage_percent']}\n")

            # Write service metrics
            for service in data["service_status"]:
                service_name = service["name"].replace("-", "_").replace(".", "_")
                f.write(f"# TYPE service_{service_name}_active gauge\n")
                f.write(f"service_{service_name}_active {int(service['is_active'])}\n")

            # Write acceleration metrics
            for accel in data["acceleration_metrics"]:
                accel_type = accel["type"]
                f.write(f"# TYPE acceleration_{accel_type}_utilization gauge\n")
                f.write(f"acceleration_{accel_type}_utilization {accel['utilization_percent']}\n")


def main():
    """Main monitoring dashboard entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="TPM2 Monitoring Dashboard")
    parser.add_argument("--config", default="/etc/military-tpm/monitoring.json",
                       help="Configuration file path")
    parser.add_argument("--export-metrics", choices=["json", "prometheus"],
                       help="Export metrics and exit")
    parser.add_argument("--health-summary", action="store_true",
                       help="Show health summary and exit")
    parser.add_argument("--dashboard-data", action="store_true",
                       help="Show dashboard data and exit")

    args = parser.parse_args()

    # Create monitoring dashboard
    dashboard = TPM2MonitoringDashboard(args.config)

    if args.export_metrics:
        # Export metrics and exit
        dashboard.start()
        time.sleep(2)  # Allow some data collection
        output_path = dashboard.export_metrics(args.export_metrics)
        print(f"Metrics exported: {output_path}")
        dashboard.stop()
        return

    if args.health_summary:
        # Show health summary and exit
        dashboard.start()
        time.sleep(2)  # Allow some data collection
        summary = dashboard.get_health_summary()
        print(json.dumps(summary, indent=2))
        dashboard.stop()
        return

    if args.dashboard_data:
        # Show dashboard data and exit
        dashboard.start()
        time.sleep(2)  # Allow some data collection
        data = dashboard.get_dashboard_data()
        print(json.dumps(data, indent=2))
        dashboard.stop()
        return

    # Run monitoring dashboard
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        dashboard.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        dashboard.start()

        # Keep running until signal received
        while dashboard.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        dashboard.stop()


if __name__ == "__main__":
    main()