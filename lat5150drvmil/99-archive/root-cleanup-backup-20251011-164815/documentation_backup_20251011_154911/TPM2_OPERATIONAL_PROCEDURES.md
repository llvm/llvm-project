# TPM2 Compatibility Layer Operational Procedures

**Date:** September 23, 2025
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version:** 1.0
**Audience:** System Administrators, Security Officers, Operations Team

## Quick Reference Operations Guide

### System Status Check
```bash
# Check all services
ls -la /home/john/military_tpm/var/run/*.pid

# View recent logs
tail -f /home/john/military_tmp/var/log/tpm2_deployment.log

# Test TPM2 compatibility
tpm2_getrandom 8
```

### Emergency Procedures
```bash
# Emergency stop (if needed)
pkill -f tpm-device-emulator
pkill -f acceleration_health_monitor

# Emergency restart
/home/john/military_tpm/bin/start-tpm2-service
/home/john/military_tpm/bin/start-health-monitor
```

## Daily Operations

### Morning Startup Checklist
1. **Verify Service Status**
   ```bash
   # Check PID files exist
   ls /home/john/military_tpm/var/run/tpm2-service.pid
   ls /home/john/military_tpm/var/run/health-monitor.pid

   # Verify processes running
   ps aux | grep tpm-device-emulator
   ps aux | grep acceleration_health_monitor
   ```

2. **Review Overnight Logs**
   ```bash
   # Check for errors or warnings
   grep -i error /home/john/military_tpm/var/log/tpm2_deployment.log
   grep -i warn /home/john/military_tpm/var/log/tpm2_deployment.log

   # Review security events
   tail -50 /home/john/military_tpm/var/log/audit.log
   ```

3. **Performance Health Check**
   ```bash
   # Test basic TPM operations
   tpm2_getrandom 16

   # Verify acceleration status
   python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/acceleration_health_monitor.py --status
   ```

### End of Day Procedures
1. **Log Review and Archive**
   ```bash
   # Check log sizes
   du -h /home/john/military_tpm/var/log/

   # Compress old logs if needed
   gzip /home/john/military_tpm/var/log/*.log.old
   ```

2. **Security Summary**
   ```bash
   # Generate daily security report
   python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/security_audit_logger.py --summary 24
   ```

## Weekly Maintenance

### Monday: System Health Assessment
```bash
# Comprehensive system check
python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/deployment_validator.py --export-report

# Performance benchmark
python3 /home/john/LAT/LAT5150DRVMIL/transparent_demo.py --performance-benchmark

# Hardware health check
lspci | grep -i "neural\|tpm\|mei"
lsmod | grep -E "tpm|mei"
```

### Wednesday: Security Audit
```bash
# Military token validation
python3 /home/john/LAT/LAT5150DRVMIL/activate_military_tokens.py --validate-only

# Security compliance check
python3 -c "
import subprocess
import os

# Check file permissions
security_dirs = ['/dev/tpm0', '/dev/mei0']
for path in security_dirs:
    if os.path.exists(path):
        stat = os.stat(path)
        print(f'{path}: {oct(stat.st_mode)}')

# Check security modules
result = subprocess.run(['lsmod'], capture_output=True, text=True)
security_modules = ['tpm', 'mei', 'intel_gna']
for module in security_modules:
    status = 'LOADED' if module in result.stdout else 'NOT_LOADED'
    print(f'{module}: {status}')
"
```

### Friday: Performance Review
```bash
# Acceleration performance test
python3 /home/john/LAT/LAT5150DRVMIL/gna_integration_demo.py --performance-test

# Stress test (short duration)
python3 -c "
import threading
import time
import hashlib

def stress_test():
    start = time.time()
    operations = 0
    while time.time() - start < 60:  # 1 minute test
        hashlib.sha256(b'stress_test').digest()
        operations += 1
    print(f'Stress test: {operations} operations in 60s ({operations/60:.1f} ops/sec)')

stress_test()
"

# Memory usage check
free -h
ps aux --sort=-%mem | head -10
```

## Monthly Procedures

### Full System Validation
```bash
# Complete deployment validation
python3 /home/john/LAT/LAT5150DRVMIL/deploy_tpm2_userspace.py --dry-run

# Comprehensive testing
python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/tests/test_compatibility.py

# Security penetration test simulation
python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/security_monitoring/test_monitoring_integration.py
```

### Configuration Backup
```bash
# Create configuration backup
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/john/military_tpm/backup/monthly_$BACKUP_DATE"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp -r /home/john/military_tpm/etc/ "$BACKUP_DIR/"
cp -r /home/john/military_tpm/bin/ "$BACKUP_DIR/"

# Create backup manifest
echo "Monthly backup created: $BACKUP_DATE" > "$BACKUP_DIR/manifest.txt"
echo "Configuration files:" >> "$BACKUP_DIR/manifest.txt"
ls -la "$BACKUP_DIR/etc/" >> "$BACKUP_DIR/manifest.txt"

echo "Monthly backup created: $BACKUP_DIR"
```

## Troubleshooting Guide

### Service Not Starting

**Symptom:** TPM2 service fails to start
```bash
# Diagnostic steps
1. Check PID file exists
   ls -la /home/john/military_tpm/var/run/tmp2-service.pid

2. Check process status
   ps aux | grep tpm-device-emulator

3. Check logs for errors
   tail -50 /home/john/military_tpm/var/log/tpm2_deployment.log

4. Test manual start
   /home/john/military_tpm/bin/tpm-device-emulator

# Resolution
# If manual start works:
/home/john/military_tpm/bin/start-tpm2-service

# If manual start fails:
# Check dependencies and permissions
python3 -c "import sys; sys.path.append('/home/john/military_tpm/lib'); import tpm2_compat"
```

### Performance Degradation

**Symptom:** Slow TPM operations or high latency
```bash
# Diagnostic steps
1. Check acceleration status
   python3 /home/john/LAT/LAT5150DRVMIL/tpm2_compat/acceleration_health_monitor.py --status

2. Monitor system resources
   top -p $(pgrep tpm-device-emulator)
   iostat 1 5

3. Test fallback mechanism
   python3 /home/john/LAT/LAT5150DRVMIL/transparent_demo.py --test-fallback

# Resolution
# Force acceleration reset:
pkill -f acceleration_health_monitor
/home/john/military_tpm/bin/start-health-monitor

# If NPU issues persist:
# Edit fallback configuration to prioritize CPU
vim /home/john/military_tpm/etc/fallback.json
```

### Security Alerts

**Symptom:** Unusual security events in audit logs
```bash
# Diagnostic steps
1. Review security events
   grep "SECURITY" /home/john/military_tpm/var/log/audit.log | tail -20

2. Check military token status
   python3 /home/john/LAT/LAT5150DRVMIL/activate_military_tokens.py --status

3. Verify system integrity
   python3 -c "
import subprocess
result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
print(f'Kernel: {result.stdout.strip()}')

# Check for unauthorized modifications
result = subprocess.run(['ls', '-la', '/dev/tpm0'], capture_output=True, text=True)
print(f'TPM device: {result.stdout.strip()}')
"

# Resolution
# If unauthorized access detected:
1. Review access logs
2. Verify user authorization levels
3. Check for system modifications
4. Report to security team if needed
```

### Memory Leaks

**Symptom:** Increasing memory usage over time
```bash
# Diagnostic steps
1. Monitor memory usage
   while true; do
     ps aux --sort=-%mem | grep tpm-device-emulator | head -1
     sleep 60
   done

2. Check for memory leaks
   valgrind --leak-check=full python3 /home/john/military_tpm/bin/tmp-device-emulator

# Resolution
# Restart services with monitoring:
pkill -f tpm-device-emulator
/home/john/military_tpm/bin/start-tpm2-service

# Monitor for 24 hours and escalate if issue persists
```

## Monitoring and Alerting

### Key Metrics to Monitor
1. **Service Availability**
   - TPM2 service uptime
   - Health monitor status
   - Process memory usage

2. **Performance Metrics**
   - TPM operation latency
   - Acceleration performance
   - Fallback activation frequency

3. **Security Metrics**
   - Failed authorization attempts
   - Token validation failures
   - Unusual access patterns

### Alert Thresholds
```json
{
  "cpu_usage_percent": 80,
  "memory_usage_percent": 85,
  "response_time_ms": 1000,
  "error_rate_percent": 5,
  "security_events_per_hour": 10
}
```

### Automated Monitoring Script
```bash
#!/bin/bash
# Save as: /home/john/military_tpm/bin/monitoring_check.sh

echo "=== TPM2 System Monitoring Check ==="
echo "Timestamp: $(date)"

# Check services
echo "Services Status:"
if [ -f "/home/john/military_tpm/var/run/tpm2-service.pid" ]; then
  echo "  TPM2 Service: RUNNING"
else
  echo "  TPM2 Service: STOPPED"
fi

if [ -f "/home/john/military_tpm/var/run/health-monitor.pid" ]; then
  echo "  Health Monitor: RUNNING"
else
  echo "  Health Monitor: STOPPED"
fi

# Check recent errors
echo "Recent Errors:"
grep -i error /home/john/military_tpm/var/log/tmp2_deployment.log | tail -3

# Check performance
echo "Performance Test:"
timeout 5s tpm2_getrandom 8 && echo "  TPM2 Response: OK" || echo "  TPM2 Response: FAILED"

echo "=== End Monitoring Check ==="
```

## Backup and Recovery

### Configuration Backup
```bash
# Create timestamped backup
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "/tmp/tpm2_config_backup_$BACKUP_DATE.tar.gz" \
  /home/john/military_tpm/etc/ \
  /home/john/military_tmp/bin/
```

### Recovery Procedures
```bash
# Service recovery
1. Stop all services
   pkill -f tpm-device-emulator
   pkill -f acceleration_health_monitor

2. Restore from backup (if needed)
   tar -xzf /tmp/tpm2_config_backup_YYYYMMDD_HHMMSS.tar.gz -C /

3. Restart services
   /home/john/military_tpm/bin/start-tpm2-service
   /home/john/military_tpm/bin/start-health-monitor

4. Verify operation
   tpm2_getrandom 8
```

## Contact Information

### Escalation Procedures
1. **Level 1:** System Administrator (Daily Operations)
2. **Level 2:** Security Officer (Security Issues)
3. **Level 3:** Technical Lead (Complex Technical Issues)
4. **Level 4:** Vendor Support (Hardware/Driver Issues)

### Emergency Contacts
- **24/7 Operations:** Internal IT Support
- **Security Incidents:** Internal Security Team
- **Technical Issues:** TPM2 Development Team

---

**Document Version:** 1.0
**Last Updated:** September 23, 2025
**Next Review:** October 23, 2025
**Approved By:** Enterprise Security Architect