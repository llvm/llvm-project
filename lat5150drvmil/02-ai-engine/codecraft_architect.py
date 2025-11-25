#!/usr/bin/env python3
"""
CodeCraft-Architect: Production-Grade Architecture Enforcement

Based on: https://github.com/xPOURY4/CodeCraft-Architect
Approach: Systematic architectural patterns and production-ready practices

This module provides architectural guidance and enforcement for maintaining
production-grade code quality across the entire codebase.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ArchitecturalLayer(Enum):
    """Three primary architectural layers"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    SHARED = "shared"


class ComponentType(Enum):
    """Types of code components"""
    CONTROLLER = "controller"  # API endpoints
    SERVICE = "service"  # Business logic
    MODEL = "model"  # Data models
    COMPONENT = "component"  # UI components
    UTILITY = "utility"  # Helper functions
    TYPE = "type"  # Type definitions
    TEST = "test"  # Test files


@dataclass
class ArchitecturalPattern:
    """Pattern for organizing code"""
    name: str
    layer: ArchitecturalLayer
    component_type: ComponentType
    directory_pattern: str
    file_naming: str
    description: str
    examples: List[str]


class CodeCraftArchitect:
    """
    Enforces CodeCraft-Architect principles for production-grade code

    Seven Key Responsibilities:
    1. Code Generation - Proper file placement and structure
    2. Context Awareness - Architectural requirements
    3. Documentation - Auto-generated docs and comments
    4. Testing & Quality - Test coverage requirements
    5. Security - Authentication, validation, protection
    6. Infrastructure - Docker, CI/CD configurations
    7. Technical Debt Tracking - Optimization opportunities
    """

    # Architectural patterns following CodeCraft-Architect
    PATTERNS = [
        ArchitecturalPattern(
            name="Backend Controller",
            layer=ArchitecturalLayer.BACKEND,
            component_type=ComponentType.CONTROLLER,
            directory_pattern="02-ai-engine/api/",
            file_naming="{feature}_controller.py",
            description="API endpoint handlers with route definitions",
            examples=["user_controller.py", "auth_controller.py"]
        ),
        ArchitecturalPattern(
            name="Backend Service",
            layer=ArchitecturalLayer.BACKEND,
            component_type=ComponentType.SERVICE,
            directory_pattern="02-ai-engine/services/",
            file_naming="{feature}_service.py",
            description="Business logic separated from controllers",
            examples=["user_service.py", "auth_service.py"]
        ),
        ArchitecturalPattern(
            name="Data Model",
            layer=ArchitecturalLayer.BACKEND,
            component_type=ComponentType.MODEL,
            directory_pattern="02-ai-engine/models/",
            file_naming="{entity}_model.py",
            description="Database models and schemas",
            examples=["user_model.py", "pattern_model.py"]
        ),
        ArchitecturalPattern(
            name="Frontend Component",
            layer=ArchitecturalLayer.FRONTEND,
            component_type=ComponentType.COMPONENT,
            directory_pattern="03-web-interface/components/",
            file_naming="{Component}.tsx",
            description="React/UI components",
            examples=["UserCard.tsx", "SearchBar.tsx"]
        ),
        ArchitecturalPattern(
            name="Frontend Service",
            layer=ArchitecturalLayer.FRONTEND,
            component_type=ComponentType.SERVICE,
            directory_pattern="03-web-interface/services/",
            file_naming="{feature}_service.ts",
            description="Client-side API calls and state management",
            examples=["api_service.ts", "auth_service.ts"]
        ),
        ArchitecturalPattern(
            name="Shared Types",
            layer=ArchitecturalLayer.SHARED,
            component_type=ComponentType.TYPE,
            directory_pattern="common/types/",
            file_naming="{domain}_types.py",
            description="Shared type definitions and interfaces",
            examples=["user_types.py", "api_types.py"]
        ),
    ]

    # Production-grade requirements
    REQUIREMENTS = {
        "documentation": {
            "module_docstring": True,
            "function_docstrings": True,
            "inline_comments": True,
            "architecture_updates": True
        },
        "testing": {
            "unit_tests": True,
            "integration_tests": True,
            "test_coverage_min": 80,
            "test_file_pattern": "test_{module}.py"
        },
        "security": {
            "input_validation": True,
            "authentication": True,
            "authorization": True,
            "data_encryption": True,
            "sql_injection_prevention": True,
            "xss_prevention": True
        },
        "code_quality": {
            "type_hints": True,
            "error_handling": True,
            "logging": True,
            "code_formatting": True,
            "linting": True
        }
    }

    @classmethod
    def get_pattern_for_feature(cls, feature: str, component_type: ComponentType) -> Optional[ArchitecturalPattern]:
        """Get the appropriate architectural pattern for a feature"""
        for pattern in cls.PATTERNS:
            if pattern.component_type == component_type:
                return pattern
        return None

    @classmethod
    def generate_file_path(cls, feature: str, component_type: ComponentType) -> str:
        """Generate proper file path following architectural patterns"""
        pattern = cls.get_pattern_for_feature(feature, component_type)
        if pattern:
            filename = pattern.file_naming.format(feature=feature, Component=feature.capitalize())
            return pattern.directory_pattern + filename
        return f"{feature}.py"  # fallback

    @classmethod
    def get_code_template(cls, component_type: ComponentType, name: str) -> str:
        """Get production-ready code template"""
        templates = {
            ComponentType.CONTROLLER: cls._controller_template(name),
            ComponentType.SERVICE: cls._service_template(name),
            ComponentType.MODEL: cls._model_template(name),
            ComponentType.UTILITY: cls._utility_template(name),
        }
        return templates.get(component_type, cls._default_template(name))

    @staticmethod
    def _controller_template(name: str) -> str:
        """Production-ready controller template"""
        return f'''#!/usr/bin/env python3
"""
{name.capitalize()} Controller - API Endpoints

Handles HTTP requests for {name} operations.
Follows CodeCraft-Architect patterns for backend controllers.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from services.{name}_service import {name.capitalize()}Service
from common.types.{name}_types import {name.capitalize()}Response
from middleware.auth import require_authentication

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{name}", tags=["{name}"])


class {name.capitalize()}Request(BaseModel):
    """Request model for {name} operations"""
    # Add fields here
    pass


@router.get("/")
async def list_{name}(
    limit: int = 100,
    offset: int = 0,
    user = Depends(require_authentication)
) -> List[{name.capitalize()}Response]:
    """
    List all {name} items

    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
        user: Authenticated user

    Returns:
        List of {name} items
    """
    try:
        service = {name.capitalize()}Service()
        items = await service.list_items(limit=limit, offset=offset, user_id=user.id)
        return items
    except Exception as e:
        logger.error(f"Error listing {name}: {{e}}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/")
async def create_{name}(
    data: {name.capitalize()}Request,
    user = Depends(require_authentication)
) -> {name.capitalize()}Response:
    """
    Create a new {name} item

    Args:
        data: Request data
        user: Authenticated user

    Returns:
        Created {name} item
    """
    try:
        service = {name.capitalize()}Service()
        item = await service.create_item(data, user_id=user.id)
        return item
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating {name}: {{e}}")
        raise HTTPException(status_code=500, detail="Internal server error")


# TODO: Add update, delete, and other CRUD endpoints
# TODO: Add input validation
# TODO: Add rate limiting
'''

    @staticmethod
    def _service_template(name: str) -> str:
        """Production-ready service template"""
        return f'''#!/usr/bin/env python3
"""
{name.capitalize()} Service - Business Logic

Contains business logic for {name} operations.
Follows CodeCraft-Architect patterns for backend services.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from models.{name}_model import {name.capitalize()}Model
from common.types.{name}_types import {name.capitalize()}Response

logger = logging.getLogger(__name__)


class {name.capitalize()}Service:
    """
    Service for {name} business logic

    Responsibilities:
    - Data validation
    - Business rules enforcement
    - Database operations
    - External API calls
    """

    def __init__(self):
        """Initialize service"""
        self.model = {name.capitalize()}Model()

    async def list_items(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None
    ) -> List[{name.capitalize()}Response]:
        """
        List {name} items with pagination

        Args:
            limit: Maximum items to return
            offset: Number of items to skip
            user_id: Optional user filter

        Returns:
            List of {name} items
        """
        try:
            items = await self.model.find_many(
                limit=limit,
                offset=offset,
                user_id=user_id
            )
            return [self._to_response(item) for item in items]
        except Exception as e:
            logger.error(f"Error listing {name} items: {{e}}")
            raise

    async def create_item(self, data: Any, user_id: str) -> {name.capitalize()}Response:
        """
        Create a new {name} item

        Args:
            data: Item data
            user_id: User ID

        Returns:
            Created item

        Raises:
            ValueError: If validation fails
        """
        # Validate input
        self._validate_data(data)

        # Apply business rules
        processed_data = self._apply_business_rules(data, user_id)

        # Create in database
        item = await self.model.create(processed_data)

        logger.info(f"Created {name} item: {{item.id}}")
        return self._to_response(item)

    def _validate_data(self, data: Any) -> None:
        """Validate input data"""
        # Add validation logic
        pass

    def _apply_business_rules(self, data: Any, user_id: str) -> Dict:
        """Apply business rules"""
        # Add business logic
        return {{**data.__dict__, "user_id": user_id}}

    def _to_response(self, item: Any) -> {name.capitalize()}Response:
        """Convert model to response"""
        return {name.capitalize()}Response(**item.__dict__)


# TODO: Add caching
# TODO: Add error recovery
# TODO: Add metrics/monitoring
'''

    @staticmethod
    def _model_template(name: str) -> str:
        """Production-ready model template"""
        return f'''#!/usr/bin/env python3
"""
{name.capitalize()} Model - Data Layer

Database model for {name} entities.
Follows CodeCraft-Architect patterns for data models.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class {name.capitalize()}Model:
    """
    Database model for {name}

    Schema:
        id: Unique identifier
        created_at: Creation timestamp
        updated_at: Update timestamp
        # Add more fields
    """
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    async def find_many(
        self,
        limit: int = 100,
        offset: int = 0,
        **filters
    ) -> List['{name.capitalize()}Model']:
        """Find multiple records"""
        # Implement database query
        pass

    async def create(self, data: Dict) -> '{name.capitalize()}Model':
        """Create new record"""
        # Implement database insert
        pass

    async def update(self, id: str, data: Dict) -> '{name.capitalize()}Model':
        """Update existing record"""
        # Implement database update
        pass

    async def delete(self, id: str) -> bool:
        """Delete record"""
        # Implement database delete
        pass


# TODO: Add indexes
# TODO: Add migrations
# TODO: Add validation
'''

    @staticmethod
    def _utility_template(name: str) -> str:
        """Production-ready utility template"""
        return f'''#!/usr/bin/env python3
"""
{name.capitalize()} Utility Functions

Helper functions for {name}.
Follows CodeCraft-Architect patterns for utilities.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def {name}_helper(data: Any) -> Any:
    """
    Helper function for {name}

    Args:
        data: Input data

    Returns:
        Processed data
    """
    try:
        # Implement helper logic
        return data
    except Exception as e:
        logger.error(f"Error in {name}_helper: {{e}}")
        raise


# Add more utility functions as needed
'''

    @staticmethod
    def _default_template(name: str) -> str:
        """Default production-ready template"""
        return f'''#!/usr/bin/env python3
"""
{name.capitalize()} Module

{name.capitalize()} implementation.
Follows CodeCraft-Architect production standards.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class {name.capitalize()}:
    """Main class for {name}"""

    def __init__(self):
        """Initialize {name}"""
        logger.info("Initializing {name}")

    def process(self, data: Any) -> Any:
        """
        Process data

        Args:
            data: Input data

        Returns:
            Processed result
        """
        try:
            # Implement processing logic
            return data
        except Exception as e:
            logger.error(f"Error processing: {{e}}")
            raise


# TODO: Add comprehensive error handling
# TODO: Add input validation
# TODO: Add monitoring/metrics
'''

    @classmethod
    def get_architectural_guidance(cls) -> str:
        """Get complete architectural guidance prompt"""
        return """
# CodeCraft-Architect: Production-Grade Development Standards

You are a lead software architect and full-stack engineer. Follow these principles:

## 1. ARCHITECTURAL ORGANIZATION

**Backend Structure:**
- Controllers: `/02-ai-engine/api/` - API endpoints only
- Services: `/02-ai-engine/services/` - Business logic
- Models: `/02-ai-engine/models/` - Data layer

**Frontend Structure:**
- Components: `/03-web-interface/components/` - UI components
- Services: `/03-web-interface/services/` - Client-side logic

**Shared:**
- Types: `/common/types/` - Shared type definitions

## 2. CODE GENERATION STANDARDS

- Place files in architecturally appropriate directories
- Follow established naming conventions
- Maintain clear separation of concerns
- Keep controllers thin, services thick

## 3. DOCUMENTATION REQUIREMENTS

- Module docstrings explaining purpose
- Function docstrings with Args/Returns
- Inline comments for complex logic
- Update architecture docs when adding modules

## 4. TESTING & QUALITY

- Generate matching test files for all modules
- Test file pattern: `test_{module}.py`
- Minimum 80% code coverage
- Include unit and integration tests

## 5. SECURITY

- Input validation on all user data
- Authentication/authorization checks
- SQL injection prevention
- XSS prevention
- Data encryption for sensitive info

## 6. INFRASTRUCTURE

- Docker configurations when needed
- CI/CD pipeline definitions
- Environment-specific configs

## 7. TECHNICAL DEBT TRACKING

- Mark optimization opportunities with TODO
- Document future refactoring needs
- Track performance bottlenecks

## IMPLEMENTATION CHECKLIST

For every feature:
- [ ] Place in correct directory
- [ ] Follow naming conventions
- [ ] Add comprehensive docstrings
- [ ] Include type hints
- [ ] Add error handling
- [ ] Add logging
- [ ] Create test file
- [ ] Add security checks
- [ ] Document in architecture

Follow these principles religiously to maintain production-grade quality.
"""


# Export main class
__all__ = ['CodeCraftArchitect', 'ComponentType', 'ArchitecturalLayer']
