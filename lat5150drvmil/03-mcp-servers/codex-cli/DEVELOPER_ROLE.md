# Developer Role Support in Codex CLI

## Overview

The codex CLI now supports **developer role messages** as described in [Simon Willison's blog post](https://simonwillison.net/2025/Nov/9/gpt-5-codex-mini/). This allows you to send system-level context or instructions **before** your user prompt to provide additional context that shapes how Codex responds.

## What is the Developer Role?

The `role="developer"` message type allows you to:

1. **Set system context** - Define project-specific requirements, conventions, or constraints
2. **Provide architectural guidance** - Specify design patterns, frameworks, or approaches to use
3. **Define code style** - Set formatting preferences, naming conventions, or documentation standards
4. **Add domain knowledge** - Include domain-specific terminology or business logic context
5. **Specify constraints** - Define what libraries, features, or approaches to avoid

## Usage Examples

### Example 1: Project Context

```rust
use codex_cli::client::{CodexClient, Message};

let mut messages = vec![
    Message::developer(
        "This is a Django REST API project. \
         Use Django REST Framework conventions. \
         All endpoints must include authentication and rate limiting."
    ),
    Message::user("Create an endpoint to list user profiles")
];

let response = client.execute_with_context(&messages).await?;
```

### Example 2: Architecture Guidance

```python
from codex_subagent import CodexAgent

agent = CodexAgent()

result = agent.execute({
    "action": "generate",
    "prompt": "Create a user authentication service",
    "language": "python",
    "developer_context": """
        Architecture: Microservices with FastAPI
        Auth: JWT tokens with refresh mechanism
        Storage: PostgreSQL with SQLAlchemy ORM
        Security: bcrypt for passwords, 2FA support required
        Testing: pytest with 80%+ coverage
    """
})
```

### Example 3: Code Style Requirements

```bash
./target/release/codex-cli exec \
  --developer-context "Follow Google Python Style Guide. Use type hints for all functions. Max line length 88 characters." \
  "Write a function to parse JSON configuration files"
```

### Example 4: Domain-Specific Context

```python
result = agent.execute({
    "action": "generate",
    "prompt": "Calculate order total with discounts",
    "language": "python",
    "developer_context": """
        E-commerce domain context:
        - Orders can have multiple line items
        - Discounts can be: percentage, fixed amount, or BOGO
        - Tax calculation happens after discounts
        - Shipping is calculated separately
        - All prices in cents (integer) to avoid float issues
    """
})
```

## Implementation in Codex CLI

### Rust Client

The `Message` struct now supports three roles:

```rust
pub struct Message {
    pub role: String,  // "user", "assistant", "developer"
    pub content: String,
}

impl Message {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self { ... }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self { ... }

    /// Create a developer message (system-level context)
    pub fn developer(content: impl Into<String>) -> Self { ... }
}
```

### Python CodexAgent

Tasks can now include `developer_context`:

```python
task = {
    "action": "generate",
    "prompt": "Main task description",
    "language": "python",
    "developer_context": "System-level context and constraints"
}
```

### MCP Server

MCP tools accept a `developer_context` parameter:

```json
{
  "name": "codex_generate",
  "arguments": {
    "description": "Create authentication middleware",
    "language": "python",
    "developer_context": "Use Flask-Login for session management"
  }
}
```

## Best Practices

### 1. Be Specific and Concise

**Good**:
```
Framework: React 18 with TypeScript
State: Redux Toolkit
Styling: Tailwind CSS
Testing: Jest + React Testing Library
```

**Avoid**:
```
Use modern React with good patterns and whatever testing framework you think is best
```

### 2. Separate Concerns

Use developer context for **how** to build, not **what** to build:

- **Developer Context**: "Use async/await, handle errors with try/catch, log to Winston"
- **User Prompt**: "Create a function to fetch user data from API"

### 3. Include Security Requirements

Always specify security constraints:

```
Security requirements:
- Validate and sanitize all user input
- Use parameterized queries (never string concatenation)
- Implement rate limiting on all endpoints
- Log security events for audit
```

### 4. Define Quality Standards

```
Code quality:
- 90%+ test coverage required
- Type hints for all functions
- Docstrings in Google style
- Max cyclomatic complexity: 10
```

### 5. Specify Dependencies

```
Allowed libraries:
- requests (HTTP client)
- pydantic (validation)
- python-jose (JWT)

Avoid:
- No deprecated libraries
- No GPL-licensed code
```

## Advanced Patterns

### Pattern 1: Layered Context

Build context incrementally for complex tasks:

```python
# Base architectural context
base_context = """
Architecture: Microservices
Language: Python 3.11
Framework: FastAPI
Database: PostgreSQL with asyncpg
"""

# Service-specific context
service_context = base_context + """
Service: User Management
Responsibilities: CRUD operations, authentication, authorization
Dependencies: Redis (sessions), RabbitMQ (events)
"""

# Task with full context
result = agent.execute({
    "action": "generate",
    "prompt": "Create user registration endpoint",
    "developer_context": service_context
})
```

### Pattern 2: Template-Based Context

Create reusable templates:

```python
SECURITY_CONTEXT = """
Security: OWASP Top 10 compliance required
- Input validation with strong typing
- Output encoding for XSS prevention
- SQL injection prevention (parameterized queries)
- CSRF protection on state-changing operations
- Rate limiting on authentication endpoints
"""

TESTING_CONTEXT = """
Testing: TDD approach
- Unit tests with pytest
- Integration tests for API endpoints
- Mock external services
- 80%+ code coverage
"""

# Combine templates
full_context = f"{SECURITY_CONTEXT}\n\n{TESTING_CONTEXT}"
```

### Pattern 3: Project Configuration File

Store developer context in a config file:

```toml
# .codex/project.toml
[project]
name = "E-commerce Platform"
language = "Python"
framework = "Django"

[architecture]
pattern = "Layered Architecture"
layers = ["API", "Business Logic", "Data Access"]

[security]
authentication = "JWT with OAuth2"
authorization = "Role-Based Access Control"
encryption = "AES-256 for sensitive data"

[code_style]
formatter = "black"
linter = "ruff"
type_checker = "mypy"
max_line_length = 88

[testing]
framework = "pytest"
coverage_minimum = 80
mock_library = "pytest-mock"
```

Then load and use:

```python
import tomli

with open(".codex/project.toml", "rb") as f:
    config = tomli.load(f)

developer_context = f"""
Project: {config['project']['name']}
Framework: {config['project']['framework']}
Architecture: {config['architecture']['pattern']}
Security: {config['security']['authentication']}
Testing: {config['testing']['framework']} with {config['testing']['coverage_minimum']}% coverage
Code Style: {config['code_style']['formatter']}
"""
```

## When to Use Developer Context

### Use It For:

- ✅ Defining technical constraints and requirements
- ✅ Specifying frameworks, libraries, or tools to use
- ✅ Setting code style and formatting preferences
- ✅ Providing architectural guidance
- ✅ Including domain-specific terminology
- ✅ Security and compliance requirements
- ✅ Testing and quality standards

### Don't Use It For:

- ❌ The actual task description (use user prompt)
- ❌ Example code or data (include in user prompt)
- ❌ Questions to Codex (use user prompt)
- ❌ Debug information or error messages (use user prompt)

## Performance Considerations

### Token Usage

Developer context adds to token count:
- Keep context focused and relevant
- Typical range: 50-200 tokens
- Update only when requirements change

### Caching

Developer context is ideal for caching:
- Reuse same context across multiple requests
- Update only when project requirements change
- Reduces cost and latency

### Context Compression

For large projects, compress context:

```python
# Instead of listing all 50 dependencies
VERBOSE = """
Dependencies: requests, pydantic, fastapi, uvicorn, sqlalchemy,
alembic, redis, celery, pytest, pytest-cov, black, ruff, mypy...
"""

# Summarize key constraints
COMPRESSED = """
Stack: FastAPI + SQLAlchemy + Redis + Celery
Testing: pytest with 80%+ coverage
Style: black + ruff + mypy strict mode
"""
```

## Integration with LAT5150DRVMIL

The developer role integrates seamlessly with the platform:

### Agent Orchestrator

```python
from codex_subagent import CodexAgent

# Define project-wide context
PROJECT_CONTEXT = """
Platform: LAT5150DRVMIL AI Tactical Platform
Language: Python 3.10+
Architecture: Agent-based with MCP servers
Security: CSNA 2.0 compliant, TPM 2.0 attestation
Hardware: Dell Latitude 5450 (Meteor Lake)
Optimization: AVX2/AVX-512 where applicable
"""

agent = CodexAgent()

# All tasks inherit project context
task = {
    "action": "generate",
    "prompt": "Create new MCP server for log analysis",
    "developer_context": PROJECT_CONTEXT
}

result = orchestrator.execute_task(task)
```

### ACE-FCA Integration

Developer context is compressed along with responses:

```python
class CodexAgent(BaseSubagent):
    def execute(self, task: Dict) -> SubagentResult:
        # Developer context is prepended to prompt
        developer_ctx = task.get("developer_context", "")

        # Full prompt includes both
        full_prompt = self._build_prompt_with_context(
            developer_context=developer_ctx,
            user_prompt=task["prompt"]
        )

        # Response is compressed as usual
        compressed = self._compress_output(raw_output, max_tokens=500)
```

## Examples by Use Case

### 1. Microservice Development

```python
MICROSERVICE_CONTEXT = """
Service Architecture:
- FastAPI with async/await
- PostgreSQL via asyncpg
- Redis for caching
- RabbitMQ for events
- OpenAPI spec generation
- Health checks on /health
- Metrics on /metrics
"""
```

### 2. Security-Critical Code

```python
SECURITY_CONTEXT = """
Security Requirements:
- OWASP Top 10 compliance
- Input validation (Pydantic models)
- Output sanitization
- Rate limiting (slowapi)
- Audit logging (all actions)
- Secrets in environment variables
- TLS 1.3 for all connections
"""
```

### 3. Data Science Pipeline

```python
DS_CONTEXT = """
Data Science Stack:
- NumPy/Pandas for data
- Scikit-learn for ML
- Jupyter for notebooks
- Type hints required
- Docstrings in NumPy style
- Unit tests with pytest
- Data validation with pandera
"""
```

### 4. Frontend Development

```python
FRONTEND_CONTEXT = """
Frontend Stack:
- React 18 with TypeScript
- Vite for bundling
- Tailwind CSS for styling
- React Query for data fetching
- Zustand for state management
- Vitest for testing
- ESLint + Prettier
"""
```

## Troubleshooting

### Issue: Context Ignored

**Problem**: Codex doesn't follow developer context

**Solutions**:
1. Make context more specific and directive
2. Repeat critical constraints in user prompt
3. Use stronger language ("must", "required", "always")

### Issue: Context Too Verbose

**Problem**: Token limit exceeded or high costs

**Solutions**:
1. Remove redundant information
2. Use abbreviated formats
3. Reference documentation URLs instead of copying

### Issue: Conflicting Instructions

**Problem**: Developer context contradicts user prompt

**Solutions**:
1. Keep context for "how", prompt for "what"
2. Prioritize user prompt for task-specific overrides
3. Structure context hierarchically

## References

- [Simon Willison's Blog Post](https://simonwillison.net/2025/Nov/9/gpt-5-codex-mini/)
- [OpenAI Codex Documentation](https://github.com/openai/codex)
- [LAT5150DRVMIL Platform](../../README.md)
- [CodexAgent Implementation](../../02-ai-engine/codex_subagent.py)

---

**Added to Codex CLI v0.1.0** | **LAT5150DRVMIL v8.3.2**
