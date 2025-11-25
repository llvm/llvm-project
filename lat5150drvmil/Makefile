.PHONY: help install install-dev clean test test-unit test-integration test-performance \
        lint format check security profile build deploy docker-build docker-run \
        docker-stop mcp-start mcp-stop mcp-status docs serve-docs update-deps \
        validate-config benchmark ci pre-commit all

.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Project directories
AI_ENGINE := 02-ai-engine
MCP_SERVERS := 03-mcp-servers
TESTS_DIR := $(AI_ENGINE)/tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist

# Python settings
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
PYLINT := $(PYTHON) -m pylint
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit

# Version and build info
VERSION := $(shell grep -oP 'version\s*=\s*"\K[^"]+' setup.py 2>/dev/null || echo "dev")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

##@ General

help: ## Display this help message
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  LAT5150DRVMIL - AI Engine Build System$(NC)"
	@echo "$(BLUE)  Version: $(VERSION)$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Installation

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ Development environment ready$(NC)"

update-deps: ## Update all dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) freeze > requirements-lock.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

##@ Testing

test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v --cov=$(AI_ENGINE) --cov-report=html --cov-report=term \
		-m "not integration and not performance" \
		--junit-xml=$(BUILD_DIR)/junit.xml
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v -m integration \
		--junit-xml=$(BUILD_DIR)/junit-integration.xml
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

test-performance: ## Run performance/benchmark tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v -m performance \
		--benchmark-only \
		--benchmark-json=$(BUILD_DIR)/benchmark.json
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Starting test watcher...$(NC)"
	$(PYTHON) -m pytest_watch $(TESTS_DIR) -- -v

benchmark: ## Run comprehensive benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) $(AI_ENGINE)/performance/benchmark_suite.py
	@echo "$(GREEN)✓ Benchmarks completed$(NC)"

##@ Code Quality

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PYLINT) $(AI_ENGINE) --output-format=colorized --reports=y
	$(MYPY) $(AI_ENGINE) --ignore-missing-imports
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) $(AI_ENGINE) $(MCP_SERVERS)
	$(ISORT) $(AI_ENGINE) $(MCP_SERVERS)
	@echo "$(GREEN)✓ Code formatted$(NC)"

check: lint ## Run all code quality checks
	@echo "$(BLUE)Running code quality checks...$(NC)"
	$(BLACK) --check $(AI_ENGINE) $(MCP_SERVERS)
	$(ISORT) --check-only $(AI_ENGINE) $(MCP_SERVERS)
	@echo "$(GREEN)✓ Code quality checks passed$(NC)"

security: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scans...$(NC)"
	$(BANDIT) -r $(AI_ENGINE) -f json -o $(BUILD_DIR)/security-report.json
	$(PIP) check
	@echo "$(GREEN)✓ Security scans completed$(NC)"

##@ Performance

profile: ## Run profiler on AI engine
	@echo "$(BLUE)Running profiler...$(NC)"
	$(PYTHON) -m cProfile -o $(BUILD_DIR)/profile.stats \
		$(AI_ENGINE)/unified_orchestrator.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('$(BUILD_DIR)/profile.stats'); p.sort_stats('cumulative').print_stats(50)"
	@echo "$(GREEN)✓ Profiling completed$(NC)"

profile-memory: ## Profile memory usage
	@echo "$(BLUE)Running memory profiler...$(NC)"
	$(PYTHON) -m memory_profiler $(AI_ENGINE)/unified_orchestrator.py
	@echo "$(GREEN)✓ Memory profiling completed$(NC)"

profile-gpu: ## Check GPU availability and performance
	@echo "$(BLUE)Checking GPU status...$(NC)"
	$(PYTHON) $(AI_ENGINE)/performance/gpu_check.py
	@echo "$(GREEN)✓ GPU check completed$(NC)"

##@ Build & Package

build: clean ## Build distribution packages
	@echo "$(BLUE)Building packages...$(NC)"
	mkdir -p $(BUILD_DIR) $(DIST_DIR)
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Build completed$(NC)"

validate-config: ## Validate configuration files
	@echo "$(BLUE)Validating configurations...$(NC)"
	$(PYTHON) $(AI_ENGINE)/validate_configs.py
	@echo "$(GREEN)✓ Configuration validated$(NC)"

##@ Data Management

validate-data: ## Validate all sample data files
	@echo "$(BLUE)Validating data files...$(NC)"
	@for file in sample_data/*.csv sample_data/*.json; do \
		if [ -f "$$file" ]; then \
			echo "  Validating $$file..."; \
			$(PYTHON) data_management/validators/data_validator.py --file "$$file" || exit 1; \
		fi \
	done
	@echo "$(GREEN)✓ Data validation completed$(NC)"

validate-data-report: ## Generate validation reports for all data
	@echo "$(BLUE)Generating validation reports...$(NC)"
	mkdir -p validation_reports
	@for file in sample_data/*.csv sample_data/*.json; do \
		if [ -f "$$file" ]; then \
			report_name=$$(basename "$$file" | sed 's/\.[^.]*$$/_report.json/'); \
			echo "  Validating $$file -> validation_reports/$$report_name"; \
			$(PYTHON) -c "from data_management.validators.data_validator import DataValidator; \
				v = DataValidator(); \
				r = v.validate_file('$$file'); \
				v.export_report(r, 'validation_reports/$$report_name')"; \
		fi \
	done
	@echo "$(GREEN)✓ Validation reports generated in validation_reports/$(NC)"

dvc-status: ## Check DVC data version control status
	@echo "$(BLUE)Checking DVC status...$(NC)"
	@command -v dvc >/dev/null 2>&1 && dvc status || echo "$(YELLOW)DVC not installed. Run: pip install dvc$(NC)"

dvc-pull: ## Pull data from DVC remote
	@echo "$(BLUE)Pulling data from DVC remote...$(NC)"
	@command -v dvc >/dev/null 2>&1 && dvc pull || echo "$(YELLOW)DVC not installed$(NC)"
	@echo "$(GREEN)✓ DVC pull completed$(NC)"

dvc-push: ## Push data to DVC remote
	@echo "$(BLUE)Pushing data to DVC remote...$(NC)"
	@command -v dvc >/dev/null 2>&1 && dvc push || echo "$(YELLOW)DVC not installed$(NC)"
	@echo "$(GREEN)✓ DVC push completed$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t lat5150drvmil:$(VERSION) -t lat5150drvmil:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Starting Docker container...$(NC)"
	docker run -d --name lat5150drvmil \
		-v $(PWD)/config:/app/config \
		-v $(PWD)/data:/app/data \
		-p 8000:8000 \
		lat5150drvmil:latest
	@echo "$(GREEN)✓ Container started$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop lat5150drvmil || true
	docker rm lat5150drvmil || true
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-logs: ## View Docker container logs
	docker logs -f lat5150drvmil

docker-shell: ## Open shell in Docker container
	docker exec -it lat5150drvmil /bin/bash

##@ MCP Servers

mcp-start: ## Start all MCP servers
	@echo "$(BLUE)Starting MCP servers...$(NC)"
	$(PYTHON) $(AI_ENGINE)/mcp_manager.py start
	@echo "$(GREEN)✓ MCP servers started$(NC)"

mcp-stop: ## Stop all MCP servers
	@echo "$(BLUE)Stopping MCP servers...$(NC)"
	$(PYTHON) $(AI_ENGINE)/mcp_manager.py stop
	@echo "$(GREEN)✓ MCP servers stopped$(NC)"

mcp-restart: mcp-stop mcp-start ## Restart all MCP servers
	@echo "$(GREEN)✓ MCP servers restarted$(NC)"

mcp-status: ## Check MCP server status
	@echo "$(BLUE)Checking MCP server status...$(NC)"
	$(PYTHON) $(AI_ENGINE)/mcp_manager.py status

mcp-logs: ## View MCP server logs
	tail -f $(AI_ENGINE)/logs/mcp-*.log

##@ Deployment

deploy-local: build ## Deploy to local environment
	@echo "$(BLUE)Deploying to local environment...$(NC)"
	./scripts/deploy-local.sh
	@echo "$(GREEN)✓ Local deployment completed$(NC)"

deploy-staging: build ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging environment...$(NC)"
	./scripts/deploy-staging.sh
	@echo "$(GREEN)✓ Staging deployment completed$(NC)"

deploy-prod: build security test ## Deploy to production (requires all checks to pass)
	@echo "$(YELLOW)⚠ Deploying to PRODUCTION...$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./scripts/deploy-prod.sh; \
		echo "$(GREEN)✓ Production deployment completed$(NC)"; \
	else \
		echo "$(RED)✗ Deployment cancelled$(NC)"; \
	fi

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdir -p $(DOCS_DIR)
	$(PYTHON) -m pdoc --html --output-dir $(DOCS_DIR) $(AI_ENGINE)
	@echo "$(GREEN)✓ Documentation generated$(NC)"

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd $(DOCS_DIR) && $(PYTHON) -m http.server 8080

##@ Maintenance

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean docker-stop ## Clean everything including Docker
	@echo "$(BLUE)Deep cleaning...$(NC)"
	docker rmi lat5150drvmil:latest lat5150drvmil:$(VERSION) 2>/dev/null || true
	rm -rf venv .venv
	@echo "$(GREEN)✓ Deep clean completed$(NC)"

logs-clean: ## Clean old log files (keep last 7 days)
	@echo "$(BLUE)Cleaning old logs...$(NC)"
	find $(AI_ENGINE)/logs -name "*.log" -mtime +7 -delete
	@echo "$(GREEN)✓ Old logs cleaned$(NC)"

##@ CI/CD

ci: install-dev check test security ## Run full CI pipeline
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  ✓ CI Pipeline Completed Successfully$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"

pre-commit: format check test-unit ## Run pre-commit checks
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

all: clean install-dev check test build docs ## Run complete build workflow
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  ✓ Complete Build Workflow Finished$(NC)"
	@echo "$(GREEN)  Version: $(VERSION)$(NC)"
	@echo "$(GREEN)  Build Date: $(BUILD_DATE)$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"

##@ Utilities

version: ## Display version information
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"

env-info: ## Display environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "  OS: $(shell uname -s)"
	@echo "  Arch: $(shell uname -m)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo "  Working Dir: $(PWD)"
	@echo "  AI Engine: $(AI_ENGINE)"
	@echo "  MCP Servers: $(MCP_SERVERS)"

quick-start: install mcp-start ## Quick start for new users
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  ✓ Quick Start Complete!$(NC)"
	@echo "$(GREEN)$(NC)"
	@echo "$(GREEN)  Try these commands:$(NC)"
	@echo "$(GREEN)    make test          - Run tests$(NC)"
	@echo "$(GREEN)    make profile       - Profile performance$(NC)"
	@echo "$(GREEN)    make mcp-status    - Check MCP servers$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════$(NC)"
