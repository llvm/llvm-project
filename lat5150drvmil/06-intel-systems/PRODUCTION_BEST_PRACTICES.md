# Screenshot Intelligence System - Production Best Practices

Comprehensive guide for production deployment, operation, and maintenance of the Screenshot Intelligence System.

## Table of Contents

1. [Deployment](#deployment)
2. [Security Hardening](#security-hardening)
3. [Performance Optimization](#performance-optimization)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Backup & Recovery](#backup--recovery)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Scaling](#scaling)

---

## Deployment

### Pre-Deployment Checklist

- [ ] **System Requirements Met**
  - Python 3.10+
  - 8GB+ RAM (16GB recommended)
  - 20GB+ free disk space
  - Docker for Qdrant
  - Tesseract OCR installed

- [ ] **Security Configuration**
  - All services bound to 127.0.0.1 (local-only)
  - API keys generated (if using API)
  - Telegram/Signal credentials secured
  - File permissions set correctly

- [ ] **Network Configuration**
  - Qdrant accessible on 127.0.0.1:6333
  - No external network exposure
  - Firewall rules verified

### Automated Deployment

```bash
cd /home/user/LAT5150DRVMIL/06-intel-systems
./deploy_screenshot_intel_production.sh
```

The deployment script handles:
- ✅ Prerequisite validation
- ✅ Dependency installation
- ✅ Qdrant setup (Docker)
- ✅ Directory structure creation
- ✅ Comprehensive testing
- ✅ Convenience scripts

### Manual Deployment Steps

If automated deployment fails:

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng libgomp1 libglib2.0-0

# 2. Install Python packages
pip3 install --user -r /home/user/LAT5150DRVMIL/06-intel-systems/screenshot-analysis-system/requirements.txt

# 3. Setup Qdrant
docker run -d --name qdrant \
  -p 127.0.0.1:6333:6333 \
  -v $HOME/qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant

# 4. Create data directories
mkdir -p $HOME/.screenshot_intel/{screenshots,logs,incidents,backups,metrics}

# 5. Run tests
cd /home/user/LAT5150DRVMIL/04-integrations/rag_system
python3 test_screenshot_intel_integration.py
```

---

## Security Hardening

### 1. Network Security

**LOCAL-ONLY Binding** (Already Implemented):
- Qdrant: `127.0.0.1:6333` ✓
- FastAPI: `127.0.0.1:8000` ✓
- MCP: stdio transport (no network) ✓

**Additional Hardening**:
```bash
# Verify no external listening
sudo netstat -tlnp | grep -E '6333|8000'
# Should show 127.0.0.1 only

# Optional: Add iptables rules
sudo iptables -A INPUT -p tcp --dport 6333 -s 127.0.0.1 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 6333 -j DROP
```

### 2. API Key Management

Generate strong API keys:
```bash
# Generate API key
export SCREENSHOT_INTEL_API_KEY=$(openssl rand -hex 32)

# Save to environment file
echo "SCREENSHOT_INTEL_API_KEY=$SCREENSHOT_INTEL_API_KEY" >> ~/.screenshot_intel/.env

# Verify permissions
chmod 600 ~/.screenshot_intel/.env
```

### 3. Credential Security

**Telegram API Credentials**:
```bash
# NEVER commit credentials to git
# Store in .env file only
echo "TELEGRAM_API_ID=your_api_id" >> ~/.screenshot_intel/.env
echo "TELEGRAM_API_HASH=your_api_hash" >> ~/.screenshot_intel/.env
chmod 600 ~/.screenshot_intel/.env
```

**Signal CLI**:
```bash
# Signal credentials stored in signal-cli config
# Ensure proper permissions
chmod 700 ~/.local/share/signal-cli
```

### 4. Data Encryption

**At Rest**:
- Use encrypted filesystem (LUKS, dm-crypt)
- Enable ZFS encryption if using ZFS

**In Transit**:
- Qdrant uses gRPC (already encrypted)
- Local-only binding prevents external access

### 5. File Permissions

```bash
# Set proper permissions
chmod 700 ~/.screenshot_intel
chmod 600 ~/.screenshot_intel/.env
chmod 755 ~/.screenshot_intel/run-*.sh
```

---

## Performance Optimization

### 1. Qdrant Optimization

**Indexing Parameters** (config in `vector_rag_system.py`):
```python
# Optimal for most workloads
VectorParams(
    size=384,  # BAAI/bge-base-en-v1.5
    distance=Distance.COSINE,
    hnsw_config={
        "m": 16,  # Number of edges per node
        "ef_construct": 100,  # Construction time accuracy
    }
)
```

**Query Optimization**:
```python
# Use appropriate search parameters
results = rag.search(
    query="...",
    limit=10,  # Don't retrieve more than needed
    score_threshold=0.5,  # Filter low-quality matches
)
```

### 2. OCR Performance

**PaddleOCR GPU Acceleration**:
```python
# Enable GPU if available
from paddleocr import PaddleOCR
paddle_ocr = PaddleOCR(use_gpu=True, gpu_mem=500)
```

**Batch Processing**:
```bash
# Ingest multiple screenshots efficiently
~/.screenshot_intel/screenshot-intel ingest scan device_id --pattern "*.png"
```

### 3. Memory Management

**Monitor Memory Usage**:
```bash
# Check system metrics
~/.screenshot_intel/run-health-check.sh

# View metrics
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/system_health_monitor.py --metrics
```

**Optimize for Low Memory**:
- Reduce `batch_size` in ingestion
- Use Tesseract instead of PaddleOCR
- Limit concurrent operations

### 4. Disk I/O

**SSD Recommended**:
- Vector database benefits from SSD
- Screenshot storage can use HDD

**Monitor Disk Usage**:
```bash
# Check disk usage
df -h ~/.screenshot_intel

# Find large files
du -sh ~/.screenshot_intel/* | sort -h
```

---

## Monitoring & Alerting

### 1. Health Checks

**Automated Health Checks** (via cron):
```bash
# Add to crontab (crontab -e)
0 */6 * * * ~/.screenshot_intel/run-health-check.sh
```

**Manual Health Check**:
```bash
~/.screenshot_intel/run-health-check.sh

# Or with Python
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/system_health_monitor.py --report
```

### 2. Metrics Collection

**Continuous Metrics**:
```bash
# Add to crontab for hourly metrics
0 * * * * python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/system_health_monitor.py --metrics
```

**Metrics Stored**:
- CPU usage
- Memory usage
- Disk usage
- Qdrant response times
- Ingestion rates

**View Metrics**:
```bash
# Recent metrics (JSONL format)
tail -100 ~/.screenshot_intel/metrics/system_metrics.jsonl

# Health check history
tail -50 ~/.screenshot_intel/metrics/health_checks.jsonl
```

### 3. Log Monitoring

**Log Locations**:
```
~/.screenshot_intel/logs/
├── deployment_YYYYMMDD_HHMMSS.log
├── maintenance_YYYYMMDD_HHMMSS.log
├── screenshot_intel.log
└── api_server.log
```

**Monitor Logs**:
```bash
# Real-time monitoring
tail -f ~/.screenshot_intel/logs/screenshot_intel.log

# Search for errors
grep -i error ~/.screenshot_intel/logs/*.log

# Count errors per hour
grep -i error ~/.screenshot_intel/logs/screenshot_intel.log | awk '{print $1" "$2}' | cut -d: -f1 | uniq -c
```

### 4. Alerting Setup

**Example: Email alerts on critical issues**

Create `/home/user/alert_monitor.sh`:
```bash
#!/bin/bash
# Email alerts for critical health issues

HEALTH_OUTPUT=$(~/.screenshot_intel/run-health-check.sh)

if echo "$HEALTH_OUTPUT" | grep -q "UNHEALTHY"; then
    echo "$HEALTH_OUTPUT" | mail -s "ALERT: Screenshot Intel Unhealthy" admin@example.com
fi
```

Add to crontab:
```bash
0 */6 * * * /home/user/alert_monitor.sh
```

---

## Backup & Recovery

### 1. Automated Backups

**Daily Backups** (via cron):
```bash
# Add to crontab
0 3 * * * ~/.screenshot_intel/run-maintenance.sh
```

This backs up:
- Incident metadata
- Device registry
- Configuration files

**Manual Backup**:
```bash
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/system_health_monitor.py --maintain --full
```

### 2. Qdrant Database Backup

**Using Qdrant Snapshots**:
```bash
# Create snapshot
curl -X POST http://127.0.0.1:6333/collections/lat5150_knowledge_base/snapshots

# List snapshots
curl http://127.0.0.1:6333/collections/lat5150_knowledge_base/snapshots

# Download snapshot
curl http://127.0.0.1:6333/collections/lat5150_knowledge_base/snapshots/snapshot-XXXXXX -O
```

**Using Docker Volume Backup**:
```bash
# Backup Qdrant data
docker run --rm \
  -v qdrant_storage:/data \
  -v ~/backups:/backup \
  ubuntu tar czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz /data
```

### 3. Screenshot Backup

**ZFS Snapshots** (if using ZFS):
```bash
# Create snapshot
sudo zfs snapshot pool/screenshots@$(date +%Y%m%d)

# List snapshots
sudo zfs list -t snapshot

# Restore from snapshot
sudo zfs rollback pool/screenshots@20251112
```

**rsync Backup**:
```bash
# Backup to external drive
rsync -av --progress ~/.screenshot_intel/screenshots/ /mnt/backup/screenshots/
```

### 4. Recovery Procedures

**Recover from Backup**:
```bash
# 1. Restore Qdrant data
docker stop qdrant
tar xzf ~/backups/qdrant_backup_YYYYMMDD.tar.gz -C ~/qdrant_storage/
docker start qdrant

# 2. Restore incidents and config
tar xzf ~/.screenshot_intel/backups/backup_YYYYMMDD_HHMMSS.tar.gz -C ~/.screenshot_intel/

# 3. Verify health
~/.screenshot_intel/run-health-check.sh
```

---

## Maintenance

### 1. Automated Maintenance

**Daily Maintenance Tasks**:
```bash
# Configured via cron
0 3 * * * ~/.screenshot_intel/run-maintenance.sh
```

Tasks performed:
- ✅ Log rotation
- ✅ Temporary file cleanup
- ✅ Database optimization
- ✅ Backup creation
- ✅ Old backup cleanup (30 days)
- ✅ Data integrity verification

### 2. Manual Maintenance

**Full Maintenance**:
```bash
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/system_health_monitor.py --maintain --full
```

**Individual Tasks**:
```bash
# Clean old logs
find ~/.screenshot_intel/logs -name "*.log" -mtime +30 -delete

# Clean temporary files
find ~/.screenshot_intel -name "*.tmp" -delete

# Verify Qdrant health
curl http://127.0.0.1:6333/collections/lat5150_knowledge_base
```

### 3. Database Maintenance

**Reindex if Performance Degrades**:
```python
from vector_rag_system import VectorRAGSystem

rag = VectorRAGSystem()
# Qdrant handles optimization automatically
# Manual optimization rarely needed
```

**Monitor Index Size**:
```bash
# Check Qdrant collection info
curl http://127.0.0.1:6333/collections/lat5150_knowledge_base | jq
```

---

## Troubleshooting

### Common Issues

#### 1. Qdrant Connection Failed

**Symptom**: `Failed to connect to Qdrant`

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker start qdrant

# Check logs
docker logs qdrant

# Verify connectivity
curl http://127.0.0.1:6333/collections
```

#### 2. OCR Not Working

**Symptom**: `PaddleOCR not available`

**Solution**:
```bash
# Install PaddleOCR
pip3 install paddleocr paddlepaddle

# If GPU issues, use CPU version
pip3 install paddlepaddle

# Fallback to Tesseract
sudo apt-get install tesseract-ocr
```

#### 3. High Memory Usage

**Symptom**: System becomes slow, memory > 90%

**Solution**:
```bash
# Check metrics
~/.screenshot_intel/run-health-check.sh

# Reduce batch size in ingestion
# Edit vector_rag_system.py: reduce batch_size

# Restart Qdrant to free memory
docker restart qdrant

# Consider adding swap or more RAM
```

#### 4. API Server Won't Start

**Symptom**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
kill <PID>

# Or use different port
export API_PORT=8001
```

### Debug Mode

**Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Run Tests**:
```bash
# Full test suite
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/test_screenshot_intel_integration.py -v

# Specific category
python3 /home/user/LAT5150DRVMIL/04-integrations/rag_system/test_screenshot_intel_integration.py --test-category database -v
```

---

## Scaling

### Horizontal Scaling

**Multiple Qdrant Nodes** (for large deployments):
```yaml
# docker-compose.yml
version: '3'
services:
  qdrant-1:
    image: qdrant/qdrant
    ports:
      - "127.0.0.1:6333:6333"
  qdrant-2:
    image: qdrant/qdrant
    ports:
      - "127.0.0.1:6334:6333"
```

### Vertical Scaling

**Increase Resources**:
- **RAM**: 16GB+ for large datasets (>1M documents)
- **CPU**: Multi-core for parallel OCR processing
- **Disk**: SSD for Qdrant storage

**Optimize Embeddings**:
```python
# Use larger model for better accuracy (requires more RAM)
VectorRAGSystem(embedding_model="BAAI/bge-large-en-v1.5")  # 1024D instead of 384D
```

### Data Archival

**Archive Old Data**:
```bash
# Export old documents
python3 -c "
from vector_rag_system import VectorRAGSystem
from datetime import datetime, timedelta

rag = VectorRAGSystem()
cutoff = datetime.now() - timedelta(days=365)
old_docs = rag.timeline_query(datetime(2020,1,1), cutoff)
# Export to archive storage
"
```

---

## Production Checklist

Before going to production:

- [ ] All tests passing
- [ ] Health checks configured
- [ ] Automated backups enabled
- [ ] Monitoring in place
- [ ] Security hardening complete
- [ ] Documentation reviewed
- [ ] Disaster recovery tested
- [ ] Performance benchmarked
- [ ] Alerts configured
- [ ] Team trained on operations

---

## Support and Maintenance Contacts

- **System Documentation**: `/home/user/LAT5150DRVMIL/06-intel-systems/`
- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **Deployment Guide**: `SCREENSHOT_INTEL_DEPLOYMENT.md`
- **API Documentation**: http://127.0.0.1:8000/api/docs (when API running)

---

## Version History

- **v1.0.0** (2025-11-12): Production-ready release with health monitoring, automated maintenance, resilience features, and comprehensive testing

