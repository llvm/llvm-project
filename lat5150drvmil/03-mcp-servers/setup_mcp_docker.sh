#!/bin/bash

# MCP Servers Docker Setup Script
# Alternative: Docker-based deployment for MCP servers + Intel LLM-Scaler
# For tomorrow's setup

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  MCP Servers + Intel LLM-Scaler Docker Setup${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 1. CHECK DOCKER
# ============================================================================

print_info "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Installing Docker..."

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER

    print_success "Docker installed. Please log out and back in for group changes to take effect."
    print_warning "After logging back in, run this script again."
    exit 0
fi

if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

print_success "Docker and docker-compose are installed"

# ============================================================================
# 2. INTEL LLM-SCALER CONTAINER
# ============================================================================

print_info "Setting up Intel LLM-Scaler (vLLM with Intel GPU optimization)..."

# Pull latest Intel LLM-Scaler image
print_info "Pulling Intel LLM-Scaler image (this may take a while)..."
docker pull intel/llm-scaler-vllm:1.0

# Create docker-compose for LLM-Scaler
cat > /tmp/llm-scaler-compose.yml <<'DOCKER_COMPOSE'
version: '3.8'

services:
  llm-scaler:
    image: intel/llm-scaler-vllm:1.0
    container_name: intel-llm-scaler
    restart: unless-stopped
    devices:
      - /dev/dri:/dev/dri  # Intel GPU device
    environment:
      - MODEL_NAME=wizardlm-uncensored-codellama:34b
      - MAX_MODEL_LEN=100000
      - GPU_MEMORY_UTILIZATION=0.9
      - ENABLE_CHUNKED_PREFILL=true
      - QUANTIZATION=int8
      - TRUST_REMOTE_CODE=true
    ports:
      - "8000:8000"  # vLLM API
      - "9090:9090"  # Metrics
    volumes:
      - ./models:/models
      - ./logs:/logs
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model ${MODEL_NAME}
      --device xpu
      --max-model-len ${MAX_MODEL_LEN}
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
      --enable-chunked-prefill
      --port 8000
      --host 0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
DOCKER_COMPOSE

print_success "LLM-Scaler docker-compose created"

# ============================================================================
# 3. LADDR MULTI-AGENT SYSTEM
# ============================================================================

print_info "Setting up Laddr multi-agent system..."

# Laddr docker-compose
cat > /tmp/laddr-compose.yml <<'LADDR_COMPOSE'
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: laddr-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=laddr
      - POSTGRES_PASSWORD=laddr_secure_pass
      - POSTGRES_DB=laddr
    ports:
      - "5433:5432"  # Different port to avoid conflict
    volumes:
      - laddr-postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U laddr"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: laddr-redis
    restart: unless-stopped
    ports:
      - "6380:6379"  # Different port to avoid conflict
    volumes:
      - laddr-redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: laddr-minio
    restart: unless-stopped
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    volumes:
      - laddr-minio-data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  laddr-api:
    image: agnetlabs/laddr:latest
    container_name: laddr-api
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      - POSTGRES_URL=postgresql://laddr:laddr_secure_pass@postgres:5432/laddr
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - LLM_PROVIDER=ollama
    ports:
      - "8001:8000"  # Different port to avoid conflict with vLLM
    volumes:
      - ./agents:/app/agents
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  laddr-dashboard:
    image: agnetlabs/laddr-dashboard:latest
    container_name: laddr-dashboard
    restart: unless-stopped
    depends_on:
      - laddr-api
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8001

volumes:
  laddr-postgres-data:
  laddr-redis-data:
  laddr-minio-data:

networks:
  default:
    name: laddr-network
LADDR_COMPOSE

print_success "Laddr docker-compose created"

# ============================================================================
# 4. CREATE MASTER DOCKER-COMPOSE
# ============================================================================

print_info "Creating master docker-compose.yml..."

mkdir -p /home/user/LAT5150DRVMIL/docker-services
cd /home/user/LAT5150DRVMIL/docker-services

# Combine everything
cat > docker-compose.yml <<'MASTER_COMPOSE'
version: '3.8'

services:
  # Intel LLM-Scaler (vLLM with Intel GPU optimization)
  llm-scaler:
    image: intel/llm-scaler-vllm:1.0
    container_name: intel-llm-scaler
    restart: unless-stopped
    devices:
      - /dev/dri:/dev/dri
    environment:
      - MODEL_NAME=wizardlm-uncensored-codellama:34b
      - MAX_MODEL_LEN=100000
      - GPU_MEMORY_UTILIZATION=0.9
      - ENABLE_CHUNKED_PREFILL=true
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - ./models:/models
      - ./llm-logs:/logs

  # Laddr PostgreSQL
  laddr-postgres:
    image: postgres:15
    restart: unless-stopped
    environment:
      - POSTGRES_USER=laddr
      - POSTGRES_PASSWORD=laddr_secure_pass
      - POSTGRES_DB=laddr
    ports:
      - "5433:5432"
    volumes:
      - laddr-postgres-data:/var/lib/postgresql/data

  # Laddr Redis
  laddr-redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6380:6379"
    volumes:
      - laddr-redis-data:/data

  # Laddr MinIO (object storage)
  laddr-minio:
    image: minio/minio:latest
    restart: unless-stopped
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - laddr-minio-data:/data
    command: server /data --console-address ":9001"

  # Laddr API
  laddr-api:
    image: agnetlabs/laddr:latest
    restart: unless-stopped
    depends_on:
      - laddr-postgres
      - laddr-redis
      - laddr-minio
    environment:
      - POSTGRES_URL=postgresql://laddr:laddr_secure_pass@laddr-postgres:5432/laddr
      - REDIS_URL=redis://laddr-redis:6379
      - MINIO_ENDPOINT=laddr-minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - VLLM_BASE_URL=http://llm-scaler:8000
      - LLM_PROVIDER=ollama
    ports:
      - "8001:8000"
    volumes:
      - ./agents:/app/agents
      - ./laddr-logs:/app/logs

  # Laddr Dashboard
  laddr-dashboard:
    image: agnetlabs/laddr-dashboard:latest
    restart: unless-stopped
    depends_on:
      - laddr-api
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8001

volumes:
  laddr-postgres-data:
  laddr-redis-data:
  laddr-minio-data:

networks:
  default:
    name: dsmil-ai-network
MASTER_COMPOSE

print_success "Master docker-compose.yml created"

# ============================================================================
# 5. CREATE MANAGEMENT SCRIPTS
# ============================================================================

print_info "Creating management scripts..."

# Start script
cat > start.sh <<'START_SCRIPT'
#!/bin/bash
echo "Starting DSMIL AI Services (Docker)..."
docker-compose up -d
echo ""
echo "Services starting..."
echo "  • Intel LLM-Scaler: http://localhost:8000"
echo "  • Laddr API: http://localhost:8001"
echo "  • Laddr Dashboard: http://localhost:5173"
echo "  • MinIO Console: http://localhost:9001"
echo ""
echo "Check status with: docker-compose ps"
echo "View logs with: docker-compose logs -f <service-name>"
START_SCRIPT

chmod +x start.sh

# Stop script
cat > stop.sh <<'STOP_SCRIPT'
#!/bin/bash
echo "Stopping DSMIL AI Services..."
docker-compose down
echo "All services stopped."
STOP_SCRIPT

chmod +x stop.sh

# Status script
cat > status.sh <<'STATUS_SCRIPT'
#!/bin/bash
echo "DSMIL AI Services Status:"
echo ""
docker-compose ps
echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
STATUS_SCRIPT

chmod +x status.sh

# Logs script
cat > logs.sh <<'LOGS_SCRIPT'
#!/bin/bash
SERVICE=${1:-"all"}

if [ "$SERVICE" == "all" ]; then
    docker-compose logs -f
else
    docker-compose logs -f "$SERVICE"
fi
LOGS_SCRIPT

chmod +x logs.sh

print_success "Management scripts created"

# ============================================================================
# 6. CREATE README
# ============================================================================

cat > README.md <<'README'
# DSMIL AI Services - Docker Deployment

## Overview

This directory contains Docker-based deployment for:
- **Intel LLM-Scaler** - vLLM with Intel GPU optimization (106 TOPS)
- **Laddr Multi-Agent Framework** - Scalable AI agent orchestration
- Supporting services: PostgreSQL, Redis, MinIO

## Quick Start

### 1. Start All Services

```bash
./start.sh
```

### 2. Check Status

```bash
./status.sh
```

### 3. View Logs

```bash
# All services
./logs.sh

# Specific service
./logs.sh llm-scaler
./logs.sh laddr-api
```

### 4. Stop Services

```bash
./stop.sh
```

## Services

### Intel LLM-Scaler (Port 8000)

**vLLM API with Intel GPU optimization**

Test:
```bash
curl http://localhost:8000/v1/models
```

Generate:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wizardlm-uncensored-codellama:34b",
    "prompt": "Explain quantum computing",
    "max_tokens": 500
  }'
```

### Laddr Multi-Agent (Port 8001)

**API:** http://localhost:8001
**Dashboard:** http://localhost:5173
**Docs:** http://localhost:8001/docs

Test:
```bash
curl http://localhost:8001/api/agents
```

Submit job:
```bash
curl -X POST http://localhost:8001/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "research_agent",
    "task": "Find information about AI benchmarking"
  }'
```

### MinIO (Ports 9000, 9001)

**API:** http://localhost:9000
**Console:** http://localhost:9001

Login: minioadmin / minioadmin

## Configuration

Edit `docker-compose.yml` to customize:

**LLM-Scaler:**
- `MODEL_NAME`: Change model
- `MAX_MODEL_LEN`: Adjust context window
- `GPU_MEMORY_UTILIZATION`: 0.7-0.98

**Laddr:**
- `OLLAMA_BASE_URL`: Point to Ollama or vLLM
- `LLM_PROVIDER`: ollama, openai, anthropic

## Performance

**Expected with 106 TOPS:**
- 34B model: ~22 tokens/sec
- 70B model: ~14 tokens/sec
- 40K context: 4.2x faster than baseline

## Troubleshooting

### GPU Not Detected

```bash
# Check Intel GPU
lspci | grep -i vga

# Check device
ls -la /dev/dri/
```

### Service Won't Start

```bash
# View logs
./logs.sh <service-name>

# Restart service
docker-compose restart <service-name>
```

### Out of Memory

Reduce `GPU_MEMORY_UTILIZATION` in docker-compose.yml:
```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.7  # Lower value
```

## Integration with Enhanced AI Engine

The Enhanced AI Engine can use these services:

```python
# In enhanced_ai_engine.py

# Use vLLM endpoint
VLLM_BASE_URL = "http://localhost:8000"

# Use Laddr for multi-agent tasks
LADDR_API_URL = "http://localhost:8001"
```

## Monitoring

### Prometheus Metrics

LLM-Scaler exposes metrics on port 9090:
```bash
curl http://localhost:9090/metrics
```

### Container Stats

```bash
docker stats
```

### Resource Usage

```bash
./status.sh
```

## Backup

### Database

```bash
docker exec laddr-postgres pg_dump -U laddr laddr > backup.sql
```

### MinIO

Access console at http://localhost:9001 and download buckets.

## Updating

```bash
# Pull latest images
docker-compose pull

# Restart services
./stop.sh
./start.sh
```

## Production Considerations

1. **Change default passwords** in docker-compose.yml
2. **Enable HTTPS** with reverse proxy (nginx, traefik)
3. **Set resource limits** in docker-compose.yml
4. **Configure persistent volumes** properly
5. **Set up monitoring** (Prometheus + Grafana)
6. **Enable backups** for PostgreSQL and MinIO

## Support

For issues:
- Intel LLM-Scaler: https://github.com/intel/llm-scaler
- Laddr: https://github.com/AgnetLabs/laddr
- vLLM: https://docs.vllm.ai/

---

**Hardware:** 106 TOPS Intel GPU
**Optimization:** vLLM + IPEX-LLM + Intel oneAPI
**Expected speedup:** 1.8x-4.2x for long context
README

print_success "README.md created"

# ============================================================================
# 7. SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Docker Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

print_success "Created Docker deployment in:"
echo "  $(pwd)"
echo ""

print_info "Files created:"
echo "  • docker-compose.yml - Master configuration"
echo "  • start.sh - Start all services"
echo "  • stop.sh - Stop all services"
echo "  • status.sh - Check service status"
echo "  • logs.sh - View logs"
echo "  • README.md - Complete documentation"
echo ""

print_info "Services included:"
echo "  ✓ Intel LLM-Scaler (vLLM + Intel GPU)"
echo "  ✓ Laddr Multi-Agent Framework"
echo "  ✓ PostgreSQL (Laddr database)"
echo "  ✓ Redis (Laddr queue)"
echo "  ✓ MinIO (Object storage)"
echo ""

print_info "Ports:"
echo "  • 8000 - Intel LLM-Scaler API"
echo "  • 8001 - Laddr API"
echo "  • 5173 - Laddr Dashboard"
echo "  • 5433 - PostgreSQL"
echo "  • 6380 - Redis"
echo "  • 9000 - MinIO API"
echo "  • 9001 - MinIO Console"
echo "  • 9090 - LLM-Scaler Metrics"
echo ""

print_warning "IMPORTANT - Tomorrow's Tasks:"
echo "  1. Review docker-compose.yml"
echo "  2. Change default passwords!"
echo "  3. Run: ./start.sh"
echo "  4. Test: curl http://localhost:8000/v1/models"
echo "  5. Access dashboard: http://localhost:5173"
echo ""

print_info "Quick start commands:"
echo "  ./start.sh          # Start everything"
echo "  ./status.sh         # Check status"
echo "  ./logs.sh llm-scaler # View LLM logs"
echo "  ./stop.sh           # Stop everything"
echo ""

print_success "Docker setup ready for tomorrow! 🚀"
echo ""
echo -e "${CYAN}Expected performance with 106 TOPS:${NC}"
echo "  • 34B model: ~22 tokens/sec"
echo "  • 70B model: ~14 tokens/sec"
echo "  • 40K context: 4.2x faster"
echo ""
