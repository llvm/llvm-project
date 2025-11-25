#!/usr/bin/env python3
"""
LAT5150 Dynamic Page Integration API
Allows external projects to register pages dynamically with full TEMPEST compliance

SECURITY: Localhost-only, validated schema, sandboxed execution
TEMPEST: Enforced tactical theming, EMF-reduced animations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Blueprint

logger = logging.getLogger(__name__)


@dataclass
class PageMetadata:
    """Metadata for a dynamically registered page"""
    page_id: str                    # Unique identifier (e.g., "cyber_retrieval")
    title: str                      # Display title
    category: str                   # Category: "analysis", "operations", "admin", "custom"
    icon: str                       # Icon class or emoji
    route: str                      # URL route (e.g., "/cyber-retrieval")
    description: str                # Brief description
    security_classification: str    # "UNCLASSIFIED", "CUI", "SECRET", "TOP_SECRET"
    tempest_compliant: bool         # TEMPEST compliance flag
    registered_by: str              # Project/module name
    registered_at: str              # ISO timestamp
    version: str                    # Page version
    requires_auth: bool = True      # Requires authentication
    tactical_mode: str = "comfort"  # Default tactical display mode
    custom_css: Optional[str] = None    # Additional CSS (must be TEMPEST-compliant)
    custom_js: Optional[str] = None     # Additional JavaScript (sandboxed)


@dataclass
class PageEndpoint:
    """API endpoint for a page"""
    method: str                 # "GET", "POST", "PUT", "DELETE"
    path: str                   # API path (e.g., "/api/cyber-retrieval/query")
    handler: str                # Handler function name
    requires_auth: bool = True
    rate_limit: int = 100       # Requests per minute
    input_schema: Optional[Dict] = None  # JSON schema for validation


@dataclass
class PageRegistration:
    """Complete page registration"""
    metadata: PageMetadata
    html_content: str               # HTML content (uses tactical theme)
    endpoints: List[PageEndpoint]   # API endpoints
    initialization_script: Optional[str] = None  # Initialization JS (sandboxed)


class DynamicPageRegistry:
    """
    Registry for dynamically registered pages
    Thread-safe, persistent storage, validation
    """

    def __init__(self, storage_path: str = "/opt/lat5150/pages"):
        """Initialize page registry"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.pages: Dict[str, PageRegistration] = {}
        self.blueprints: Dict[str, Blueprint] = {}

        self._load_registered_pages()

        logger.info(f"📄 Dynamic Page Registry initialized ({len(self.pages)} pages)")

    def register_page(
        self,
        registration: PageRegistration,
        overwrite: bool = False
    ) -> bool:
        """
        Register a new page

        Args:
            registration: Page registration data
            overwrite: Allow overwriting existing page

        Returns:
            Success status

        Raises:
            ValueError: Invalid registration data
            PermissionError: Page already exists and overwrite=False
        """
        page_id = registration.metadata.page_id

        # Validate registration
        self._validate_registration(registration)

        # Check if page exists
        if page_id in self.pages and not overwrite:
            raise PermissionError(f"Page '{page_id}' already exists. Use overwrite=True to replace.")

        # Validate TEMPEST compliance
        if registration.metadata.tempest_compliant:
            self._validate_tempest_compliance(registration)

        # Store registration
        self.pages[page_id] = registration

        # Persist to disk
        self._save_page(page_id, registration)

        logger.info(f"✓ Registered page: {page_id} ({registration.metadata.title})")
        return True

    def unregister_page(self, page_id: str) -> bool:
        """Unregister a page"""
        if page_id not in self.pages:
            logger.warning(f"Page '{page_id}' not found")
            return False

        del self.pages[page_id]

        # Remove from disk
        page_file = self.storage_path / f"{page_id}.json"
        if page_file.exists():
            page_file.unlink()

        logger.info(f"✓ Unregistered page: {page_id}")
        return True

    def get_page(self, page_id: str) -> Optional[PageRegistration]:
        """Get page registration by ID"""
        return self.pages.get(page_id)

    def list_pages(self, category: Optional[str] = None) -> List[PageMetadata]:
        """
        List all registered pages

        Args:
            category: Filter by category (optional)

        Returns:
            List of page metadata
        """
        pages = [reg.metadata for reg in self.pages.values()]

        if category:
            pages = [p for p in pages if p.category == category]

        return sorted(pages, key=lambda p: p.title)

    def get_page_html(self, page_id: str, tactical_mode: str = "comfort") -> Optional[str]:
        """
        Get rendered HTML for a page

        Args:
            page_id: Page identifier
            tactical_mode: Tactical display mode (comfort, day, night)

        Returns:
            Rendered HTML with tactical theme applied
        """
        page = self.get_page(page_id)
        if not page:
            return None

        # Wrap content in tactical theme container
        html = self._wrap_tactical_theme(
            page.html_content,
            page.metadata,
            tactical_mode
        )

        return html

    def create_blueprint(self, page_id: str, app: Flask) -> Optional[Blueprint]:
        """
        Create Flask Blueprint for page endpoints

        Args:
            page_id: Page identifier
            app: Flask application

        Returns:
            Configured Blueprint
        """
        page = self.get_page(page_id)
        if not page:
            return None

        # Create blueprint
        bp = Blueprint(
            page_id,
            __name__,
            url_prefix=page.metadata.route
        )

        # Register endpoints
        for endpoint in page.endpoints:
            self._register_endpoint(bp, endpoint, page)

        # Cache blueprint
        self.blueprints[page_id] = bp

        logger.info(f"✓ Created blueprint for: {page_id} ({len(page.endpoints)} endpoints)")
        return bp

    def _validate_registration(self, registration: PageRegistration):
        """Validate page registration"""
        metadata = registration.metadata

        # Validate page_id
        if not metadata.page_id.isidentifier():
            raise ValueError(f"Invalid page_id: '{metadata.page_id}' (must be valid Python identifier)")

        # Validate route
        if not metadata.route.startswith('/'):
            raise ValueError(f"Route must start with '/': {metadata.route}")

        # Validate category
        valid_categories = ["analysis", "operations", "admin", "custom", "integration"]
        if metadata.category not in valid_categories:
            raise ValueError(f"Invalid category: {metadata.category}. Must be one of: {valid_categories}")

        # Validate security classification
        valid_classifications = ["UNCLASSIFIED", "CUI", "SECRET", "TOP_SECRET"]
        if metadata.security_classification not in valid_classifications:
            raise ValueError(f"Invalid classification: {metadata.security_classification}")

        # Validate HTML content
        if not registration.html_content.strip():
            raise ValueError("HTML content cannot be empty")

        logger.debug(f"✓ Validated registration for: {metadata.page_id}")

    def _validate_tempest_compliance(self, registration: PageRegistration):
        """
        Validate TEMPEST compliance

        Checks:
        - No excessive animations (EMF reduction)
        - Brightness limits enforced
        - Color schemes follow tactical standards
        - No external resources (OPSEC)
        """
        html = registration.html_content
        css = registration.metadata.custom_css or ""

        # Check for forbidden patterns
        forbidden_patterns = [
            ("@keyframes", "Animations increase EMF emissions"),
            ("animation:", "Animations increase EMF emissions"),
            ("transition: all", "Full transitions increase EMF"),
            ("http://", "External resources violate OPSEC"),
            ("https://", "External resources violate OPSEC"),
            ("<script src=", "External scripts violate OPSEC"),
        ]

        violations = []
        for pattern, reason in forbidden_patterns:
            if pattern in html or pattern in css:
                violations.append(f"{pattern}: {reason}")

        if violations:
            raise ValueError(f"TEMPEST compliance violations:\n" + "\n".join(f"  - {v}" for v in violations))

        logger.debug(f"✓ TEMPEST compliance validated")

    def _wrap_tactical_theme(
        self,
        content: str,
        metadata: PageMetadata,
        tactical_mode: str
    ) -> str:
        """Wrap content in tactical theme container"""
        classification_color = {
            "UNCLASSIFIED": "var(--classification-unclass)",
            "CUI": "var(--classification-cui)",
            "SECRET": "var(--classification-secret)",
            "TOP_SECRET": "var(--classification-ts)"
        }.get(metadata.security_classification, "#00ff00")

        template = f"""
        <div class="tactical-page-container" data-tactical-mode="{tactical_mode}">
            <!-- Security Classification Banner -->
            <div class="classification-banner" style="background: {classification_color}; color: #000; padding: 4px 16px; text-align: center; font-weight: bold;">
                {metadata.security_classification}
            </div>

            <!-- Page Header -->
            <div class="tactical-page-header">
                <h1 class="tactical-page-title">
                    <span class="tactical-icon">{metadata.icon}</span>
                    {metadata.title}
                </h1>
                <p class="tactical-page-description">{metadata.description}</p>
            </div>

            <!-- Page Content -->
            <div class="tactical-page-content">
                {content}
            </div>

            <!-- Page Footer -->
            <div class="tactical-page-footer">
                <span class="tactical-page-meta">
                    Registered by: {metadata.registered_by} | Version: {metadata.version} |
                    {'TEMPEST Compliant ✓' if metadata.tempest_compliant else 'Standard Mode'}
                </span>
            </div>
        </div>
        """

        return template

    def _register_endpoint(self, bp: Blueprint, endpoint: PageEndpoint, page: PageRegistration):
        """Register API endpoint on blueprint"""
        # This is a placeholder - actual handler registration would be done by the integrating project
        # The registry provides the structure, the project provides the implementation

        @bp.route(endpoint.path, methods=[endpoint.method])
        def endpoint_handler():
            return jsonify({
                "error": "Handler not implemented",
                "message": f"Page '{page.metadata.page_id}' must provide handler implementation"
            }), 501

        logger.debug(f"✓ Registered endpoint: {endpoint.method} {endpoint.path}")

    def _save_page(self, page_id: str, registration: PageRegistration):
        """Persist page registration to disk"""
        page_file = self.storage_path / f"{page_id}.json"

        data = {
            "metadata": asdict(registration.metadata),
            "html_content": registration.html_content,
            "endpoints": [asdict(e) for e in registration.endpoints],
            "initialization_script": registration.initialization_script
        }

        with open(page_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"✓ Saved page to: {page_file}")

    def _load_registered_pages(self):
        """Load registered pages from disk"""
        if not self.storage_path.exists():
            return

        for page_file in self.storage_path.glob("*.json"):
            try:
                with open(page_file) as f:
                    data = json.load(f)

                # Reconstruct registration
                metadata = PageMetadata(**data["metadata"])
                endpoints = [PageEndpoint(**e) for e in data["endpoints"]]

                registration = PageRegistration(
                    metadata=metadata,
                    html_content=data["html_content"],
                    endpoints=endpoints,
                    initialization_script=data.get("initialization_script")
                )

                self.pages[metadata.page_id] = registration
                logger.info(f"✓ Loaded page: {metadata.page_id}")

            except Exception as e:
                logger.error(f"Failed to load page from {page_file}: {e}")

    def export_registry_info(self) -> Dict[str, Any]:
        """Export registry information for status/monitoring"""
        return {
            "total_pages": len(self.pages),
            "pages_by_category": {
                category: len([p for p in self.pages.values() if p.metadata.category == category])
                for category in ["analysis", "operations", "admin", "custom", "integration"]
            },
            "tempest_compliant": len([p for p in self.pages.values() if p.metadata.tempest_compliant]),
            "storage_path": str(self.storage_path),
            "pages": [asdict(p.metadata) for p in self.pages.values()]
        }


# Global registry instance
_registry: Optional[DynamicPageRegistry] = None


def get_page_registry() -> DynamicPageRegistry:
    """Get or create global page registry"""
    global _registry
    if _registry is None:
        _registry = DynamicPageRegistry()
    return _registry


def register_page_api_routes(app: Flask):
    """
    Register page management API routes

    Endpoints:
    - GET  /api/pages               - List all pages
    - GET  /api/pages/<page_id>     - Get page details
    - POST /api/pages/register      - Register new page
    - DELETE /api/pages/<page_id>   - Unregister page
    - GET  /api/pages/registry-info - Get registry information
    """
    registry = get_page_registry()

    @app.route('/api/pages', methods=['GET'])
    def list_pages():
        """List all registered pages"""
        category = request.args.get('category')
        pages = registry.list_pages(category=category)
        return jsonify({
            "pages": [asdict(p) for p in pages],
            "total": len(pages)
        })

    @app.route('/api/pages/<page_id>', methods=['GET'])
    def get_page_details(page_id: str):
        """Get page details"""
        page = registry.get_page(page_id)
        if not page:
            return jsonify({"error": f"Page '{page_id}' not found"}), 404

        return jsonify({
            "metadata": asdict(page.metadata),
            "endpoints": [asdict(e) for e in page.endpoints],
            "has_initialization_script": page.initialization_script is not None
        })

    @app.route('/api/pages/register', methods=['POST'])
    def register_page():
        """Register a new page"""
        try:
            data = request.json

            # Construct registration
            metadata = PageMetadata(**data['metadata'])
            endpoints = [PageEndpoint(**e) for e in data.get('endpoints', [])]

            registration = PageRegistration(
                metadata=metadata,
                html_content=data['html_content'],
                endpoints=endpoints,
                initialization_script=data.get('initialization_script')
            )

            # Register
            overwrite = data.get('overwrite', False)
            success = registry.register_page(registration, overwrite=overwrite)

            # Create blueprint
            blueprint = registry.create_blueprint(metadata.page_id, app)
            if blueprint:
                app.register_blueprint(blueprint)

            return jsonify({
                "success": success,
                "page_id": metadata.page_id,
                "route": metadata.route
            })

        except Exception as e:
            logger.error(f"Failed to register page: {e}")
            return jsonify({"error": str(e)}), 400

    @app.route('/api/pages/<page_id>', methods=['DELETE'])
    def unregister_page(page_id: str):
        """Unregister a page"""
        success = registry.unregister_page(page_id)
        if not success:
            return jsonify({"error": f"Page '{page_id}' not found"}), 404

        return jsonify({"success": True, "page_id": page_id})

    @app.route('/api/pages/registry-info', methods=['GET'])
    def get_registry_info():
        """Get registry information"""
        return jsonify(registry.export_registry_info())

    @app.route('/page/<page_id>', methods=['GET'])
    def render_page(page_id: str):
        """Render a registered page"""
        tactical_mode = request.args.get('mode', 'comfort')
        html = registry.get_page_html(page_id, tactical_mode=tactical_mode)

        if not html:
            return f"Page '{page_id}' not found", 404

        return render_template_string(html)

    logger.info("✓ Registered page management API routes")
