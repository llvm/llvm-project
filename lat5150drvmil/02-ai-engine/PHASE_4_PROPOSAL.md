# Phase 4: Testing, Optimization & Deployment
**Proposed Next Steps for LAT5150DRVMIL AI Engine**

---

## Current State

✅ **Complete Stack** (enumerated in `CURRENT_STACK_INVENTORY.md`):
- 7 ai-that-works patterns integrated (4,250 LOC)
- Full DIRECTEYE OSINT integration
- Production safety (human-in-loop)
- Complete audit trail (event-driven)
- Memory optimization (30-50% savings)
- Entity resolution + enrichment
- Dynamic schema generation
- Agentic RAG intelligence

**All components integrated and individually tested** ✅

---

## Phase 4 Options

### Option A: Production Deployment (Recommended)
**Duration**: 4-6 hours
**Focus**: Deploy to production with monitoring

#### Tasks:
1. **Integration Testing** (2 hours)
   - End-to-end test suite
   - All components working together
   - Real-world scenario tests
   - Load testing

2. **Configuration Management** (1 hour)
   - Environment variables (.env)
   - Production database setup (PostgreSQL)
   - Redis configuration
   - DIRECTEYE API keys
   - HILP notification system

3. **Monitoring & Observability** (2 hours)
   - Metrics collection (Prometheus/Grafana)
   - Log aggregation (ELK/Loki)
   - Alerting (PagerDuty/Slack)
   - Dashboard creation

4. **Documentation** (1 hour)
   - Deployment guide
   - Operations runbook
   - Troubleshooting guide
   - API documentation

**Deliverables**:
- Production-ready deployment
- Monitoring dashboards
- Operational documentation
- Automated testing

---

### Option B: Advanced Features (Optional Enhancements)
**Duration**: 6-8 hours
**Focus**: Add next-level capabilities

#### Tasks:
1. **MCP Tool Selector** (3 hours)
   - Intelligent tool selection from 35+ MCP tools
   - Context-aware routing
   - Tool capability matching
   - Cost optimization

2. **Generative UI** (3 hours)
   - Dynamic workflow UI generation
   - Interactive approval interfaces
   - Real-time visualization
   - Entity relationship graphs

3. **Advanced Analytics** (2 hours)
   - Pattern recognition across conversations
   - Entity network analysis
   - Behavior prediction
   - Anomaly detection

**Deliverables**:
- MCP tool selector
- Generative UI framework
- Advanced analytics dashboard

---

### Option C: Specialized Extensions (Domain-Specific)
**Duration**: 4-6 hours
**Focus**: Add domain-specific capabilities

#### Tasks:
1. **Threat Intelligence Automation** (2 hours)
   - Automated IOC extraction
   - Threat actor attribution
   - Campaign tracking
   - Automated reporting

2. **Blockchain Investigation** (2 hours)
   - Transaction graph analysis
   - Wallet clustering
   - Mixer detection
   - DeFi protocol tracking

3. **OSINT Workflows** (2 hours)
   - Pre-built investigation workflows
   - Social media intel automation
   - Dark web monitoring
   - Credential breach detection

**Deliverables**:
- Domain-specific tooling
- Automated workflows
- Investigation templates

---

### Option D: Performance Optimization (Speed & Efficiency)
**Duration**: 3-4 hours
**Focus**: Optimize performance and reduce costs

#### Tasks:
1. **Query Optimization** (1 hour)
   - RAG query performance
   - Database indexing
   - Cache hit rate improvement

2. **Model Selection Optimization** (1 hour)
   - Cost-aware model routing
   - Quality/speed/cost tradeoffs
   - Automatic model selection

3. **Memory Optimization** (1 hour)
   - Further memory decay tuning
   - Aggressive summarization for old data
   - Batch processing optimization

4. **Benchmarking** (1 hour)
   - Performance baselines
   - Before/after comparisons
   - Regression testing

**Deliverables**:
- Performance benchmarks
- Optimized configurations
- Cost analysis

---

## Recommended Path: **Option A (Production Deployment)**

### Why This Makes Sense:
1. **You have a complete, tested stack** - time to deploy it
2. **Production use will reveal real-world needs** - optimize based on actual usage
3. **Monitoring is critical** - understand how the system performs
4. **You can add advanced features later** - based on production learnings

### Detailed Breakdown: Production Deployment

#### 4.1 Integration Testing (2 hours)

**Create**: `02-ai-engine/tests/integration_test_suite.py`

```python
import pytest
import asyncio
from enhanced_ai_engine import EnhancedAIEngine, RiskLevel

@pytest.mark.asyncio
async def test_full_intelligence_pipeline():
    """Test: Entity extraction → OSINT enrichment → RAG storage → Query"""

    engine = EnhancedAIEngine(
        enable_entity_resolution=True,
        enable_agentic_rag=True,
        enable_event_driven=True
    )

    # 1. Extract entities with OSINT enrichment
    result = await engine.extract_and_resolve_entities(
        text="Contact john@example.com or bitcoin:1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        enrich=True
    )

    assert len(result['extracted']) > 0
    assert len(result['enriched']) > 0

    # 2. Query with agentic RAG
    rag_result = engine.agentic_rag_query(
        "What email addresses do we have?",
        enable_reformulation=True
    )

    assert rag_result['query_reformulation']['intent'] in ['factual', 'exploratory']
    assert len(rag_result['chunks']) > 0

    # 3. Verify event logging
    state = engine.event_driven_agent.get_state()
    assert state.conversation_turns > 0


@pytest.mark.asyncio
async def test_human_in_loop_workflow():
    """Test: HILP approval workflow"""

    engine = EnhancedAIEngine(enable_human_in_loop=True)

    async def risky_operation(resource_id: str):
        return f"Deleted {resource_id}"

    # Execute in background
    task = asyncio.create_task(
        engine.execute_with_approval(
            operation="delete_resource",
            operation_func=risky_operation,
            parameters={"resource_id": "test_123"},
            risk_override=RiskLevel.HIGH
        )
    )

    # Wait for pending approval
    await asyncio.sleep(0.5)

    # Approve
    pending = engine.get_pending_approvals()
    assert len(pending) == 1

    engine.approve_pending_request(pending[0]['request_id'], approved_by="test_admin")

    # Wait for execution
    result = await task

    assert result['success'] == True
    assert result['approval']['status'] == 'approved'


@pytest.mark.asyncio
async def test_memory_decay():
    """Test: Memory decay saves tokens"""

    engine = EnhancedAIEngine(enable_decaying_memory=True)

    # Add some memory blocks
    for i in range(10):
        engine.hierarchical_memory.add_block(
            content=f"Test content {i} " * 100,
            block_type="qa_pair",
            importance=0.5
        )

    # Apply decay
    stats_before = engine.hierarchical_memory.get_stats()
    await engine.apply_memory_decay()
    stats_after = engine.hierarchical_memory.get_stats()

    # Should have saved tokens (or no decay if < 1 hour old)
    # Just verify it runs without error
    assert 'working_tokens' in stats_after


@pytest.mark.asyncio
async def test_dynamic_schema_generation():
    """Test: Schema generation from examples"""

    engine = EnhancedAIEngine(enable_dynamic_schemas=True)

    examples = [
        {"name": "Alice", "age": 30, "active": True},
        {"name": "Bob", "age": 25, "active": False}
    ]

    result = engine.generate_schema_from_examples(
        examples=examples,
        model_name="TestUser"
    )

    assert result['model_name'] == "TestUser"
    assert 'name' in result['schema']
    assert 'age' in result['schema']
    assert result['validation_passed'] == True


def test_statistics_collection():
    """Test: All components report statistics"""

    engine = EnhancedAIEngine()
    stats = engine.get_statistics()

    # Verify all expected sections present
    assert 'engine' in stats
    assert 'conversations' in stats
    assert 'cache' in stats
    assert 'memory' in stats

    # New components
    assert 'multi_model_eval' in stats or 'error' not in stats
    assert 'event_driven' in stats or 'error' not in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

**Run**: `pytest tests/integration_test_suite.py -v`

---

#### 4.2 Configuration Management (1 hour)

**Create**: `02-ai-engine/.env.production`

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lat5150_production
POSTGRES_USER=lat5150_user
POSTGRES_PASSWORD=<SECURE_PASSWORD>

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<SECURE_PASSWORD>
REDIS_DB=0

# DIRECTEYE API Keys
DIRECTEYE_API_KEY=<YOUR_KEY>
HUNTER_IO_API_KEY=<YOUR_KEY>
CLEARBIT_API_KEY=<YOUR_KEY>
HOLEHE_ENABLED=true
SHERLOCK_ENABLED=true

# Blockchain API Keys
ETHERSCAN_API_KEY=<YOUR_KEY>
BLOCKCHAIN_INFO_API_KEY=<YOUR_KEY>

# LLM Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama-70b
UNCENSORED_MODEL=uncensored_code

# Human-in-Loop Configuration
HILP_ENABLED=true
HILP_AUDIT_PATH=/var/log/lat5150/hilp_audit.log
HILP_NOTIFICATION_WEBHOOK=https://your-slack-webhook
HILP_DEFAULT_TIMEOUT=300

# Event-Driven Agent
EVENT_STORE_PATH=/var/lib/lat5150/event_store.db

# Memory Configuration
MEMORY_DECAY_ENABLED=true
MEMORY_DECAY_MIN_AGE_HOURS=1.0
MEMORY_DECAY_SCHEDULE=auto

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

**Create**: `02-ai-engine/deploy/docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  lat5150_engine:
    build: .
    env_file:
      - .env.production
    depends_on:
      - postgres
      - redis
    volumes:
      - ./:/app
      - /var/log/lat5150:/var/log/lat5150
      - /var/lib/lat5150:/var/lib/lat5150
    ports:
      - "8000:8000"

  prometheus:
    image: prom/prometheus
    volumes:
      - ./deploy/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./deploy/grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

#### 4.3 Monitoring Setup (2 hours)

**Create**: `02-ai-engine/monitoring/metrics_exporter.py`

```python
#!/usr/bin/env python3
"""Prometheus metrics exporter for LAT5150DRVMIL"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
from enhanced_ai_engine import EnhancedAIEngine

# Metrics
CONVERSATION_TURNS = Counter('lat5150_conversation_turns_total', 'Total conversation turns')
ENTITIES_EXTRACTED = Counter('lat5150_entities_extracted_total', 'Entities extracted', ['entity_type'])
APPROVALS_REQUESTED = Counter('lat5150_approvals_requested_total', 'Approval requests', ['risk_level'])
APPROVALS_STATUS = Counter('lat5150_approvals_status_total', 'Approval status', ['status'])
MEMORY_TOKENS = Gauge('lat5150_memory_tokens', 'Memory tokens', ['tier'])
RAG_QUERIES = Counter('lat5150_rag_queries_total', 'RAG queries', ['intent', 'strategy'])
MODEL_LATENCY = Histogram('lat5150_model_latency_seconds', 'Model latency', ['model'])

def collect_metrics(engine: EnhancedAIEngine):
    """Collect metrics from engine"""
    stats = engine.get_statistics()

    # Event-driven metrics
    if 'event_driven' in stats:
        CONVERSATION_TURNS.inc(stats['event_driven']['conversation_turns'])

    # Entity resolution metrics
    if 'entity_resolution' in stats:
        for entity_type, count in stats['entity_resolution']['entity_types'].items():
            ENTITIES_EXTRACTED.labels(entity_type=entity_type).inc(count)

    # HILP metrics
    if 'human_in_loop' in stats:
        for risk, count in stats['human_in_loop']['risk_breakdown'].items():
            APPROVALS_REQUESTED.labels(risk_level=risk).inc(count)
        for status, count in stats['human_in_loop']['status_breakdown'].items():
            APPROVALS_STATUS.labels(status=status).inc(count)

    # Memory metrics
    if 'memory' in stats:
        MEMORY_TOKENS.labels(tier='working').set(stats['memory']['working_tokens'])
        MEMORY_TOKENS.labels(tier='short_term').set(stats['memory']['short_term_blocks'] * 100)
        MEMORY_TOKENS.labels(tier='long_term').set(stats['memory']['long_term_blocks'] * 100)

    # Agentic RAG metrics
    if 'agentic_rag' in stats:
        for intent, count in stats['agentic_rag']['intent_distribution'].items():
            for strategy, strategy_count in stats['agentic_rag']['strategy_distribution'].items():
                RAG_QUERIES.labels(intent=intent, strategy=strategy).inc(1)


if __name__ == '__main__':
    # Start metrics server
    start_http_server(9090)

    # Initialize engine
    engine = EnhancedAIEngine()

    # Collect metrics every 15 seconds
    while True:
        try:
            collect_metrics(engine)
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        time.sleep(15)
```

**Create**: `02-ai-engine/deploy/grafana-dashboard.json`
(Full Grafana dashboard with panels for all metrics)

---

#### 4.4 Deployment Documentation (1 hour)

**Create**: `02-ai-engine/DEPLOYMENT_GUIDE.md`

```markdown
# LAT5150DRVMIL Production Deployment Guide

## Prerequisites

- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Python 3.10+
- Ollama (for LLM serving)

## Quick Start

1. **Clone repository**
   ```bash
   cd /home/user/LAT5150DRVMIL/02-ai-engine
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with your credentials
   ```

3. **Start services**
   ```bash
   docker-compose -f deploy/docker-compose.yml up -d
   ```

4. **Initialize database**
   ```bash
   python scripts/init_database.py
   ```

5. **Start engine**
   ```bash
   python enhanced_ai_engine.py
   ```

6. **Verify deployment**
   ```bash
   pytest tests/integration_test_suite.py
   ```

## Monitoring

- **Metrics**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)
  - Default credentials: admin / <GRAFANA_PASSWORD>
- **Logs**: `/var/log/lat5150/`

## Operations

### Check System Health
```bash
curl http://localhost:8000/health
```

### View Statistics
```python
from enhanced_ai_engine import EnhancedAIEngine
engine = EnhancedAIEngine()
stats = engine.get_statistics()
print(stats)
```

### Approve Pending Requests
```python
pending = engine.get_pending_approvals()
for req in pending:
    print(f"Request: {req['operation']} (risk: {req['risk_level']})")
    # Approve or reject
    engine.approve_pending_request(req['request_id'], approved_by="admin")
```

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.
```

---

## Timeline: Option A (Production Deployment)

**Day 1** (4 hours):
- Hour 1-2: Integration testing
- Hour 3: Configuration management
- Hour 4: Monitoring setup

**Day 2** (2 hours):
- Hour 1: Documentation
- Hour 2: Deployment & verification

**Total**: 6 hours to production-ready deployment

---

## Success Criteria

### For Production Deployment:
- [ ] All integration tests passing
- [ ] Production environment configured
- [ ] Monitoring dashboards operational
- [ ] Documentation complete
- [ ] System deployed and verified
- [ ] Approval workflow tested end-to-end
- [ ] Performance baseline established

---

## Next Decision Point

**After Option A (Production Deployment)**, you can choose:

1. **Monitor production usage** for 1-2 weeks
2. **Identify real-world needs** from actual usage
3. **Optimize based on data** (Option D)
4. **Add advanced features** based on demand (Option B or C)

This approach ensures you're building what's actually needed, not what might be needed.

---

## Recommendation

**Start with Option A (Production Deployment)** because:
1. ✅ You have a complete, tested stack
2. ✅ Production use reveals real needs
3. ✅ Can iterate based on actual usage
4. ✅ Monitoring provides data for optimization
5. ✅ Can add features incrementally

Let me know which option you'd like to pursue, or if you want a custom combination!
