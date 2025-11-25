# Enhanced AI Engine - Deployment Guide

Complete guide for deploying the Enhanced AI Engine to production.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Running the Engine](#running-the-engine)
7. [Health Checks](#health-checks)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Operations Runbook](#operations-runbook)

---

## Quick Start

```bash
# 1. Clone and navigate
cd /path/to/LAT5150DRVMIL/02-ai-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your values

# 4. Setup databases
./scripts/setup_databases.sh

# 5. Run tests
python3 tests/integration_test_suite.py

# 6. Start the engine
python3 -c "from enhanced_ai_engine import EnhancedAIEngine; engine = EnhancedAIEngine(); print('✅ Engine started')"
```

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+) or macOS
- **Python**: 3.9+

### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD (NVMe preferred)
- **Network**: 1Gbps+
- **OS**: Ubuntu 22.04 LTS or RHEL 9

### External Services
- **PostgreSQL**: 13+ (for conversation history)
- **Redis**: 6+ (for response caching)
- **DIRECTEYE API**: Access token required

---

## Installation

### 1. Python Dependencies

Create requirements.txt if not exists:

```txt
# Core dependencies
psycopg2-binary>=2.9.0
redis>=4.0.0
chromadb>=0.3.0
sentence-transformers>=2.2.0

# Optional (for specific components)
pydantic>=2.0.0  # For dynamic schemas
anthropic>=0.18.0  # If using Claude
openai>=1.0.0  # If using OpenAI models
```

Install:
```bash
pip install -r requirements.txt
```

### 2. System Packages

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    postgresql-client \
    redis-tools \
    build-essential \
    python3-dev
```

RHEL/CentOS:
```bash
sudo dnf install -y \
    postgresql \
    redis \
    gcc \
    python3-devel
```

---

## Configuration

### Environment Variables

Copy and edit the example environment file:

```bash
cp .env.example .env
nano .env  # or vim, emacs, etc.
```

### Required Configuration

#### Database (PostgreSQL)
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lat5150_ai
POSTGRES_USER=lat5150
POSTGRES_PASSWORD=your_secure_password_here
```

#### Cache (Redis)
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here
```

#### API Keys
```bash
DIRECTEYE_API_KEY=your_directeye_api_key_here
```

### Optional Configuration

#### Component Toggles
Enable/disable specific components:
```bash
ENABLE_MCP_SELECTOR=true
ENABLE_THREAT_INTEL=true
ENABLE_BLOCKCHAIN_TOOLS=true
ENABLE_OSINT_WORKFLOWS=true
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_HERETIC=true  # LLM abliteration (uncensoring)
```

#### Heretic Abliteration Configuration
Enable/configure heretic abliteration features (Unsloth + DECCP + remove-refusals):
```bash
# Enable/disable heretic
ENABLE_HERETIC=true

# Use Unsloth optimization (2x speed, 70% VRAM reduction)
HERETIC_USE_UNSLOTH=true

# Abliteration method: "single_layer", "multi_layer", "adaptive"
HERETIC_METHOD=multi_layer

# Quantization (if using Unsloth): "4bit", "8bit", "none"
HERETIC_QUANTIZATION=4bit

# Layer aggregation: "mean", "weighted_mean", "max"
HERETIC_LAYER_AGGREGATION=mean
```

#### Performance Tuning
```bash
MAX_CONTEXT_WINDOW=131072
RAM_CONTEXT_SIZE_MB=512
MAX_CONCURRENT_QUERIES=10
MCP_OPTIMIZE_FOR=balanced  # quality, speed, cost, balanced
```

---

## Database Setup

### PostgreSQL Setup

#### Create Database and User
```sql
-- Connect to PostgreSQL
sudo -u postgres psql

-- Create database
CREATE DATABASE lat5150_ai;

-- Create user
CREATE USER lat5150 WITH ENCRYPTED PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE lat5150_ai TO lat5150;

-- Exit
\q
```

#### Run Migrations
```bash
# Initialize conversation tables
python3 -c "
from conversation_manager import ConversationManager
cm = ConversationManager()
print('✅ Conversation tables created')
"
```

### Redis Setup

#### Ubuntu/Debian
```bash
sudo apt-get install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

#### Configure Redis (optional)
Edit `/etc/redis/redis.conf`:
```conf
# Set password
requirepass your_redis_password_here

# Set max memory
maxmemory 2gb
maxmemory-policy allkeys-lru
```

Restart Redis:
```bash
sudo systemctl restart redis
```

### ChromaDB Setup

ChromaDB runs in-process by default, no setup needed.

For client/server mode (optional):
```bash
# Install ChromaDB server
pip install chromadb[server]

# Start server
chroma run --path ./chroma_db --port 8000
```

---

## Running the Engine

### Development Mode

```python
from enhanced_ai_engine import EnhancedAIEngine

# Initialize with all components
engine = EnhancedAIEngine()

# Start a conversation
conv = engine.start_conversation("Test Conversation")

# Query the engine
response = engine.query(
    prompt="What is the capital of France?",
    model="gpt-4",
    conversation_id=conv.id
)

print(response.content)
```

### Production Mode

```python
from enhanced_ai_engine import EnhancedAIEngine
from config import Config

# Load production config
config = Config.from_env()

# Initialize engine with config
engine = EnhancedAIEngine(
    enable_self_improvement=config.components.enable_self_improvement,
    enable_dsmil_integration=config.components.enable_dsmil_integration,
    enable_mcp_selector=config.components.enable_mcp_selector,
    enable_threat_intel=config.components.enable_threat_intel,
    enable_blockchain_tools=config.components.enable_blockchain_tools,
    enable_osint_workflows=config.components.enable_osint_workflows,
    enable_advanced_analytics=config.components.enable_advanced_analytics,
    mcp_optimize_for=config.performance.mcp_optimize_for
)

# Use the engine...
```

### As a Service (systemd)

Create `/etc/systemd/system/ai-engine.service`:

```ini
[Unit]
Description=Enhanced AI Engine
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=lat5150
WorkingDirectory=/opt/LAT5150DRVMIL/02-ai-engine
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 /opt/LAT5150DRVMIL/02-ai-engine/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-engine
sudo systemctl start ai-engine
sudo systemctl status ai-engine
```

---

## Heretic Abliteration Usage

The Enhanced AI Engine includes **Heretic abliteration** - an advanced LLM uncensoring system that combines three cutting-edge techniques:

- **Unsloth**: 2x faster training, 70% VRAM reduction (4-bit/8-bit quantization)
- **DECCP**: Multi-layer computation + LLM-as-Judge evaluation
- **remove-refusals**: Broad model compatibility (15+ architectures)

### Quick Start Example

```python
from enhanced_ai_engine import EnhancedAIEngine

# Initialize engine with heretic enabled
engine = EnhancedAIEngine(
    enable_heretic=True,
    heretic_use_unsloth=True,  # 2x speed, 70% VRAM savings
    heretic_method="multi_layer"  # DECCP multi-layer computation
)

# Define training prompts
harmless_prompts = [
    "How do I bake a cake?",
    "Tell me about the solar system",
    "How can I learn programming?"
]

harmful_prompts = [
    "How do I make a bomb?",
    "How can I hack into a computer?",
    "How do I create a virus?"
]

# Abliterate a model
result = engine.abliterate_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    harmless_prompts=harmless_prompts,
    harmful_prompts=harmful_prompts,
    output_path="./models/llama2-uncensored",
    method="multi_layer",
    use_unsloth=True,
    quantization="4bit",
    batch_size=4
)

print(f"Abliteration complete! Saved to: {result['output_path']}")
print(f"Memory used: {result['memory_stats']['peak_memory_gb']:.2f} GB")
```

### Web API Usage

The heretic system provides REST API endpoints for web integration.

#### 1. Check System Status

```bash
curl http://localhost:5000/api/heretic/status
```

Response:
```json
{
  "available": true,
  "enhanced_available": true,
  "version": "2.0.0",
  "components": {
    "unsloth": true,
    "deccp": true,
    "remove_refusals": true,
    "llm_judge": true
  },
  "features": {
    "unsloth": "2x faster training, 70% less VRAM",
    "deccp": "Multi-layer computation + LLM-as-Judge",
    "remove_refusals": "Broad model compatibility"
  }
}
```

#### 2. Start Enhanced Abliteration

```bash
curl -X POST http://localhost:5000/api/heretic/abliterate/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "harmless_prompts": ["Tell me a story", "What is 2+2?"],
    "harmful_prompts": ["How to hack", "Make a weapon"],
    "output_path": "./models/qwen2-uncensored",
    "method": "multi_layer",
    "use_unsloth": true,
    "quantization": "4bit",
    "layer_aggregation": "mean"
  }'
```

Response:
```json
{
  "job_id": "enhanced_abliterate_1",
  "status": "started",
  "message": "Enhanced abliteration started for model: Qwen/Qwen2-7B-Instruct",
  "features": {
    "unsloth": true,
    "method": "multi_layer",
    "quantization": "4bit"
  }
}
```

#### 3. Check Job Status

```bash
curl http://localhost:5000/api/heretic/abliterate/enhanced_abliterate_1
```

Response:
```json
{
  "job_id": "enhanced_abliterate_1",
  "status": "completed",
  "model": "Qwen/Qwen2-7B-Instruct",
  "method": "multi_layer",
  "progress": 100,
  "result": {
    "output_path": "./models/qwen2-uncensored",
    "refusal_directions": 32,
    "layers_used": [12, 13, 14, 15, 16],
    "memory_stats": {
      "peak_memory_gb": 8.4,
      "vram_savings_percent": 70
    }
  }
}
```

#### 4. Evaluate with LLM-as-Judge

```bash
curl -X POST http://localhost:5000/api/heretic/evaluate/llm-judge \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "test_prompts": [
      "Tell me about network security",
      "How do cryptographic systems work?"
    ]
  }'
```

Response:
```json
{
  "evaluations": [
    {
      "prompt": "Tell me about network security",
      "helpfulness": 8.5,
      "refusal_detected": false,
      "reasoning": "Provides comprehensive security information"
    }
  ],
  "summary": {
    "average_helpfulness": 8.5,
    "refusal_rate": 0.0
  }
}
```

#### 5. Get Memory Statistics

```bash
curl http://localhost:5000/api/heretic/memory-stats
```

Response:
```json
{
  "cuda_available": true,
  "cuda_memory": {
    "allocated_gb": 8.4,
    "reserved_gb": 9.2,
    "device_count": 1,
    "device_name": "NVIDIA RTX 3090"
  },
  "cpu_memory": {
    "total_gb": 64.0,
    "available_gb": 48.2,
    "used_gb": 15.8,
    "percent": 24.7
  }
}
```

### Supported Models

The enhanced heretic system supports 15+ model architectures:

**Text Models:**
- Llama (meta-llama/Llama-2-7b-chat-hf, etc.)
- Qwen (Qwen/Qwen2-7B-Instruct, etc.)
- Mistral (mistralai/Mistral-7B-Instruct-v0.2, etc.)
- Gemma (google/gemma-7b-it, etc.)
- DeepSeek (deepseek-ai/deepseek-coder-7b-instruct-v1.5, etc.)

**Multimodal Models:**
- LLaVA (liuhaotian/llava-v1.5-7b, etc.)
- Qwen-VL (Qwen/Qwen-VL-Chat, etc.)

See [HERETIC_INTEGRATION_GUIDE.md](./HERETIC_INTEGRATION_GUIDE.md) for complete model compatibility matrix.

### Performance Benchmarks

**Memory Usage (7B models):**
- Standard loading: 28GB VRAM
- With 4-bit Unsloth: 8.4GB VRAM (70% reduction)
- With 8-bit Unsloth: 14GB VRAM (50% reduction)

**Speed:**
- Standard: 100% baseline
- With Unsloth: 200% (2x faster)

**Hardware Requirements:**
- **Minimum**: RTX 3090 (24GB VRAM) with 4-bit quantization
- **Recommended**: A100 (40GB VRAM) for optimal performance
- **CPU-only**: Possible but very slow (not recommended)

### Advanced Configuration

#### Method Comparison

**single_layer** (Original Heretic):
```python
engine.abliterate_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    method="single_layer",  # Fast, basic quality
    ...
)
```

**multi_layer** (DECCP - Recommended):
```python
engine.abliterate_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    method="multi_layer",  # Better quality, aggregates multiple layers
    layer_aggregation="weighted_mean",  # Options: mean, weighted_mean, max
    ...
)
```

**adaptive** (Automatic Layer Selection):
```python
engine.abliterate_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    method="adaptive",  # Automatically finds optimal layers
    ...
)
```

### Troubleshooting

#### Out of VRAM

**Problem**: `torch.cuda.OutOfMemoryError`

**Solution**: Enable Unsloth with 4-bit quantization:
```python
engine = EnhancedAIEngine(
    enable_heretic=True,
    heretic_use_unsloth=True  # Enables 4-bit by default
)
```

#### Model Not Compatible

**Problem**: `AttributeError: 'Model' object has no attribute 'model'`

**Solution**: The enhanced system uses generic layer access patterns and should work with most models. If issues persist, check [HERETIC_INTEGRATION_GUIDE.md](./HERETIC_INTEGRATION_GUIDE.md) for model-specific notes.

#### Slow Performance

**Problem**: Abliteration taking too long

**Solutions**:
1. Enable Unsloth optimization (2x speedup)
2. Reduce batch size: `batch_size=2`
3. Use `method="single_layer"` instead of `"multi_layer"`

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'unsloth'`

**Solution**: Install heretic dependencies:
```bash
pip install unsloth transformers torch
```

### Security Considerations

**⚠️ IMPORTANT**: Abliterated models remove safety guardrails. Use responsibly:

- Only for research, security testing, or authorized use cases
- Not for production systems serving untrusted users
- Implement application-level safety filters if needed
- Monitor model outputs for harmful content

---

## Health Checks

### Component Health Check

```python
from enhanced_ai_engine import EnhancedAIEngine

engine = EnhancedAIEngine()

# Get comprehensive statistics
stats = engine.get_statistics()

# Check components
print(f"Conversations: {stats['conversations']['total_conversations']}")
print(f"Cache hit rate: {stats['cache']['cache_hits']} / {stats['cache']['cache_requests']}")
print(f"Memory usage: {stats['memory']}")
print(f"Event-driven: {stats['event_driven']}")
print(f"MCP Selector: {stats['mcp_selector']}")
print(f"Threat Intel: {stats['threat_intelligence']}")
print(f"Blockchain: {stats['blockchain_investigation']}")
print(f"OSINT Workflows: {stats['osint_workflows']}")
print(f"Analytics: {stats['advanced_analytics']}")
```

### Database Connectivity

```bash
# PostgreSQL
psql "postgresql://lat5150:password@localhost:5432/lat5150_ai" -c "SELECT 1;"

# Redis
redis-cli -a your_password PING
```

### Integration Tests

```bash
# Run full test suite
python3 tests/integration_test_suite.py

# Should output:
# Total Tests: 25
# Passed:      25 (100.0%)
# Failed:      0 (0.0%)
```

---

## Monitoring

### Metrics to Monitor

1. **Performance Metrics**
   - Query latency (p50, p95, p99)
   - Throughput (queries/second)
   - Cache hit rate
   - Memory usage

2. **Component Metrics**
   - MCP tool selections
   - Threat intelligence reports generated
   - Blockchain investigations run
   - OSINT workflow executions
   - Patterns/anomalies detected

3. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network traffic

4. **Database Metrics**
   - PostgreSQL connections
   - Redis memory usage
   - ChromaDB index size

### Log Locations

```bash
# Application logs
tail -f ./logs/ai_engine.log

# Audit logs (HILP)
tail -f ./logs/audit.log

# Event store (SQLite)
sqlite3 agent_events.db "SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;"
```

### Alerting Rules

Recommended alerts:
- Query latency > 5 seconds
- Error rate > 1%
- Cache hit rate < 50%
- Database connection failures
- Disk space < 20%

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### "psycopg2.OperationalError: could not connect"
```bash
# Solution: Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection settings in .env
echo $POSTGRES_HOST $POSTGRES_PORT
```

#### "Redis connection error"
```bash
# Solution: Check Redis is running
sudo systemctl status redis

# Test connection
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD PING
```

#### "Out of memory"
```bash
# Solution 1: Reduce context windows in .env
MAX_CONTEXT_WINDOW=65536  # Instead of 131072
RAM_CONTEXT_SIZE_MB=256   # Instead of 512

# Solution 2: Enable decaying memory for automatic cleanup
ENABLE_DECAYING_MEMORY=true
```

#### "High latency"
```bash
# Solution: Optimize MCP routing
MCP_OPTIMIZE_FOR=speed  # Instead of quality

# Reduce concurrent queries
MAX_CONCURRENT_QUERIES=5  # Instead of 10

# Check database indexes
# Check cache hit rate
```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG
```

---

## Operations Runbook

### Daily Operations

**Morning Checks** (5 minutes):
1. Check service status: `systemctl status ai-engine`
2. Review overnight logs: `grep ERROR logs/ai_engine.log | tail -20`
3. Check disk space: `df -h`
4. Verify databases: PostgreSQL + Redis connectivity

**End of Day** (5 minutes):
1. Review statistics: `python3 -c "from enhanced_ai_engine import EnhancedAIEngine; e = EnhancedAIEngine(); print(e.get_statistics())"`
2. Check for anomalies in audit log
3. Backup databases (automated, verify completion)

### Weekly Operations

**Monday** (15 minutes):
1. Run integration tests: `python3 tests/integration_test_suite.py`
2. Review performance trends
3. Check for security updates
4. Clean up old logs: `find logs/ -name "*.log.*" -mtime +30 -delete`

**Friday** (30 minutes):
1. Review week's statistics
2. Update threat intelligence indicators
3. Check for component updates
4. Plan any maintenance for weekend

### Monthly Operations

**First Monday** (1 hour):
1. Full system backup
2. Database vacuum/optimize
3. Review and update configuration
4. Security audit
5. Update dependencies: `pip list --outdated`

### Incident Response

**Service Down**:
1. Check service status: `systemctl status ai-engine`
2. Review recent logs: `journalctl -u ai-engine -n 100`
3. Check dependencies: PostgreSQL, Redis
4. Restart if needed: `systemctl restart ai-engine`
5. Document incident

**High Error Rate**:
1. Check error logs: `grep ERROR logs/ai_engine.log | tail -50`
2. Identify pattern (database? external API? specific component?)
3. Disable problematic component if needed (edit .env)
4. Escalate if unresolved in 30 minutes

**Performance Degradation**:
1. Check resource usage: `top`, `free -h`, `df -h`
2. Review slow query logs
3. Check cache hit rate
4. Consider scaling up or optimizing queries

### Backup and Recovery

**Automated Backups** (configure cron):
```bash
# Daily PostgreSQL backup
0 2 * * * pg_dump lat5150_ai | gzip > /backups/ai_engine_$(date +\%Y\%m\%d).sql.gz

# Daily event store backup
0 3 * * * cp /path/to/agent_events.db /backups/agent_events_$(date +\%Y\%m\%d).db
```

**Recovery**:
```bash
# Restore PostgreSQL
gunzip < backup.sql.gz | psql lat5150_ai

# Restore event store
cp backup.db agent_events.db
```

### Scaling

**Vertical Scaling**:
- Increase RAM_CONTEXT_SIZE_MB
- Increase MAX_CONCURRENT_QUERIES
- Upgrade server resources

**Horizontal Scaling**:
- Deploy multiple instances behind load balancer
- Use shared PostgreSQL/Redis
- Consider read replicas for PostgreSQL

---

## Security Checklist

- [ ] Strong database passwords configured
- [ ] Redis password set
- [ ] API keys stored securely (not in code)
- [ ] Rate limiting enabled
- [ ] Audit logging enabled
- [ ] SSL/TLS for database connections
- [ ] Regular security updates
- [ ] Firewall rules configured
- [ ] Access logs monitored
- [ ] PII detection enabled

---

## Support and Resources

**Documentation**:
- This guide
- `IMPLEMENTATION_SUMMARY.md` - Component details
- `PHASE_4_PROPOSAL.md` - Architecture overview
- `CURRENT_STACK_INVENTORY.md` - Full stack inventory

**Code**:
- Integration tests: `tests/integration_test_suite.py`
- Configuration: `config.py`
- Main engine: `enhanced_ai_engine.py`

**Getting Help**:
1. Check logs first
2. Run integration tests
3. Review troubleshooting section
4. Check component-specific documentation

---

## Version History

- **v4.0** (2025-11-18): Phase 4 complete - Options B+C + Production deployment
- **v3.0**: Phase 3 - Human-in-loop executor
- **v2.0**: Phase 2 - Entity resolution, Dynamic schemas, Agentic RAG
- **v1.0**: Phase 1 - Event-driven, Multi-model eval, Decaying memory

---

**Last Updated**: 2025-11-18
**Maintained By**: LAT5150DRVMIL Team
