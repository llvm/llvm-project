#!/usr/bin/env python3
"""
Example: Dynamic Page Integration for External Projects
Demonstrates how to register a custom page with the LAT5150 tactical UI

This example shows integration for a hypothetical "Jina Cyber Retrieval" project
"""

from datetime import datetime
from dynamic_page_api import (
    get_page_registry,
    PageRegistration,
    PageMetadata,
    PageEndpoint
)


def register_example_page():
    """
    Example: Register a cyber threat retrieval page

    This demonstrates a complete page registration with:
    - TEMPEST-compliant HTML
    - Tactical theming
    - Multiple API endpoints
    - Security classification
    """

    # Get the global registry
    registry = get_page_registry()

    # =========================================================================
    # STEP 1: Define page metadata
    # =========================================================================

    metadata = PageMetadata(
        # Unique identifier (valid Python identifier)
        page_id="cyber_threat_retrieval",

        # Display title (shown in page header)
        title="Cyber Threat Retrieval",

        # Category for navigation grouping
        # Options: "analysis", "operations", "admin", "custom", "integration"
        category="analysis",

        # Icon (emoji or font awesome class)
        icon="🛡️",

        # URL route (must start with /)
        route="/cyber-retrieval",

        # Brief description
        description="Advanced persistent threat intelligence retrieval using Jina AI embeddings",

        # Security classification
        # Options: "UNCLASSIFIED", "CUI", "SECRET", "TOP_SECRET"
        security_classification="CUI",

        # TEMPEST compliance (enforces EMF reduction)
        tempest_compliant=True,

        # Registered by (your project name)
        registered_by="jina_cyber_retrieval_module",

        # Registration timestamp (ISO 8601)
        registered_at=datetime.utcnow().isoformat(),

        # Version (semantic versioning)
        version="1.0.0",

        # Requires authentication (default: True)
        requires_auth=True,

        # Default tactical display mode
        # Options: "comfort", "day", "night"
        tactical_mode="comfort"
    )

    # =========================================================================
    # STEP 2: Create TEMPEST-compliant HTML content
    # =========================================================================

    html_content = """
    <div class="tactical-container">
        <!-- Search Section -->
        <div class="tactical-section">
            <h2 class="tactical-section-title">Threat Intelligence Search</h2>

            <div class="tactical-flex tactical-flex-col tactical-gap-md">
                <!-- Search Input -->
                <div>
                    <label class="tactical-label">Query</label>
                    <input
                        type="text"
                        id="threat-query"
                        class="tactical-input"
                        placeholder="Enter threat indicators, CVE IDs, or keywords..."
                    />
                </div>

                <!-- Search Depth -->
                <div style="width: 200px;">
                    <label class="tactical-label">Search Depth</label>
                    <select id="search-depth" class="tactical-select">
                        <option value="1">Surface (Fast)</option>
                        <option value="3" selected>Standard</option>
                        <option value="5">Deep</option>
                        <option value="10">Exhaustive</option>
                    </select>
                </div>

                <!-- Search Button -->
                <div>
                    <button
                        id="search-btn"
                        class="tactical-btn tactical-btn-primary"
                        onclick="performSearch()"
                    >
                        Execute Search
                    </button>
                </div>
            </div>
        </div>

        <!-- Statistics Grid -->
        <div class="tactical-grid">
            <div class="tactical-card">
                <div class="tactical-card-header">Query Status</div>
                <div class="tactical-card-value" id="query-status">READY</div>
                <span class="tactical-status tactical-status-info" id="status-badge">STANDBY</span>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">Results Found</div>
                <div class="tactical-card-value" id="results-count">--</div>
                <div class="tactical-card-footer">
                    <span id="results-time">--</span>
                </div>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">Threat Level</div>
                <div class="tactical-card-value" id="threat-level">--</div>
                <span class="tactical-status tactical-status-success" id="threat-badge">NOMINAL</span>
            </div>
        </div>

        <!-- Results Section -->
        <div class="tactical-section">
            <h2 class="tactical-section-title">Retrieved Intelligence</h2>

            <div id="results-container">
                <div class="tactical-alert tactical-alert-info">
                    Enter a query to retrieve threat intelligence from the Jina embeddings database.
                </div>
            </div>
        </div>

        <!-- Recent Queries -->
        <div class="tactical-section">
            <h3 class="tactical-section-subtitle">Recent Queries</h3>
            <table class="tactical-table" id="recent-queries">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Query</th>
                        <th>Results</th>
                        <th>Threat Level</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="4" style="text-align: center; color: var(--text-muted);">
                            No recent queries
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
    /**
     * Client-side JavaScript (sandboxed)
     * TEMPEST-compliant: No animations, minimal DOM updates
     */

    // Search function
    async function performSearch() {
        const query = document.getElementById('threat-query').value;
        const depth = parseInt(document.getElementById('search-depth').value);

        if (!query.trim()) {
            showAlert('Please enter a query', 'warning');
            return;
        }

        // Update status (instant - TEMPEST compliant)
        updateStatus('SEARCHING', 'processing');

        try {
            // Call API endpoint
            const response = await fetch('/api/cyber-retrieval/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query, depth })
            });

            const data = await response.json();

            // Update UI (instant updates only)
            document.getElementById('results-count').textContent = data.results.length;
            document.getElementById('results-time').textContent = `Query time: ${data.query_time_ms}ms`;
            document.getElementById('threat-level').textContent = data.threat_level;

            // Update status
            updateStatus('COMPLETE', 'success');

            // Display results
            displayResults(data.results);

            // Update recent queries table
            addRecentQuery(query, data.results.length, data.threat_level);

        } catch (error) {
            console.error('Search failed:', error);
            updateStatus('ERROR', 'error');
            showAlert('Search failed: ' + error.message, 'error');
        }
    }

    // Update status indicators (instant - no animation)
    function updateStatus(text, type) {
        document.getElementById('query-status').textContent = text;

        const badge = document.getElementById('status-badge');
        badge.textContent = text;
        badge.className = 'tactical-status tactical-status-' + type;
    }

    // Display search results
    function displayResults(results) {
        const container = document.getElementById('results-container');
        container.innerHTML = '';

        if (results.length === 0) {
            container.innerHTML = `
                <div class="tactical-alert tactical-alert-warning">
                    No threat intelligence found for this query.
                </div>
            `;
            return;
        }

        // Create result cards
        results.forEach((result, index) => {
            const card = document.createElement('div');
            card.className = 'tactical-card tactical-mb-md';
            card.innerHTML = `
                <div class="tactical-card-header">Result #${index + 1} (Score: ${result.score.toFixed(3)})</div>
                <div style="margin: var(--spacing-sm) 0;">
                    <strong style="color: var(--text-primary);">${escapeHtml(result.title)}</strong>
                </div>
                <div style="color: var(--text-secondary); font-size: 13px;">
                    ${escapeHtml(result.description)}
                </div>
                <div class="tactical-card-footer">
                    Source: ${result.source} | Type: ${result.type}
                </div>
            `;
            container.appendChild(card);
        });
    }

    // Add to recent queries table
    function addRecentQuery(query, count, level) {
        const tbody = document.getElementById('recent-queries').querySelector('tbody');

        // Remove "no queries" message if present
        if (tbody.rows[0].cells.length === 1) {
            tbody.innerHTML = '';
        }

        // Add new row at top
        const row = tbody.insertRow(0);
        const timestamp = new Date().toLocaleTimeString();

        row.innerHTML = `
            <td>${timestamp}</td>
            <td>${escapeHtml(query.substring(0, 50))}${query.length > 50 ? '...' : ''}</td>
            <td>${count}</td>
            <td><span class="tactical-text-${getThreatColor(level)}">${level}</span></td>
        `;

        // Keep only last 10 queries
        while (tbody.rows.length > 10) {
            tbody.deleteRow(tbody.rows.length - 1);
        }
    }

    // Show alert message
    function showAlert(message, type) {
        const container = document.getElementById('results-container');
        container.innerHTML = `
            <div class="tactical-alert tactical-alert-${type}">
                ${escapeHtml(message)}
            </div>
        `;
    }

    // Get threat level color
    function getThreatColor(level) {
        const colors = {
            'LOW': 'success',
            'MODERATE': 'info',
            'HIGH': 'warning',
            'CRITICAL': 'error'
        };
        return colors[level] || 'info';
    }

    // HTML escape utility
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Allow Enter key to trigger search
    document.getElementById('threat-query').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    </script>
    """

    # =========================================================================
    # STEP 3: Define API endpoints
    # =========================================================================

    endpoints = [
        # Search endpoint
        PageEndpoint(
            method="POST",
            path="/api/cyber-retrieval/search",
            handler="handle_search",
            requires_auth=True,
            rate_limit=30,  # 30 requests per minute
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query", "depth"]
            }
        ),

        # Status endpoint
        PageEndpoint(
            method="GET",
            path="/api/cyber-retrieval/status",
            handler="handle_status",
            requires_auth=True,
            rate_limit=100
        ),

        # History endpoint
        PageEndpoint(
            method="GET",
            path="/api/cyber-retrieval/history",
            handler="handle_history",
            requires_auth=True,
            rate_limit=60
        )
    ]

    # =========================================================================
    # STEP 4: Create registration and register page
    # =========================================================================

    registration = PageRegistration(
        metadata=metadata,
        html_content=html_content,
        endpoints=endpoints
    )

    # Register the page
    try:
        success = registry.register_page(registration, overwrite=True)
        if success:
            print(f"✓ Successfully registered: {metadata.title}")
            print(f"  Page ID: {metadata.page_id}")
            print(f"  Route: {metadata.route}")
            print(f"  Endpoints: {len(endpoints)}")
            print(f"  TEMPEST: {metadata.tempest_compliant}")
            return True
        else:
            print("✗ Failed to register page")
            return False

    except Exception as e:
        print(f"✗ Registration error: {e}")
        return False


def register_simple_dashboard():
    """
    Example: Simple status dashboard
    Minimal example showing basic structure
    """

    registry = get_page_registry()

    metadata = PageMetadata(
        page_id="simple_dashboard",
        title="System Dashboard",
        category="operations",
        icon="📊",
        route="/dashboard",
        description="Simple system status dashboard",
        security_classification="UNCLASSIFIED",
        tempest_compliant=True,
        registered_by="example_module",
        registered_at=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

    html_content = """
    <div class="tactical-container">
        <div class="tactical-grid">
            <div class="tactical-card">
                <div class="tactical-card-header">System Status</div>
                <div class="tactical-card-value">OPERATIONAL</div>
                <span class="tactical-status tactical-status-success">ONLINE</span>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">Active Users</div>
                <div class="tactical-card-value" id="user-count">1</div>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">Uptime</div>
                <div class="tactical-card-value" id="uptime">--</div>
            </div>
        </div>
    </div>

    <script>
    // Update uptime every second
    function updateUptime() {
        fetch('/api/dashboard/uptime')
            .then(r => r.json())
            .then(data => {
                document.getElementById('uptime').textContent = data.uptime;
            });
    }

    updateUptime();
    setInterval(updateUptime, 1000);
    </script>
    """

    endpoints = [
        PageEndpoint(
            method="GET",
            path="/api/dashboard/uptime",
            handler="get_uptime"
        )
    ]

    registration = PageRegistration(
        metadata=metadata,
        html_content=html_content,
        endpoints=endpoints
    )

    registry.register_page(registration, overwrite=True)
    print(f"✓ Registered: {metadata.title}")


if __name__ == "__main__":
    """
    Run this script to register example pages
    """
    print("=" * 70)
    print("LAT5150 Dynamic Page Integration - Examples")
    print("=" * 70)
    print()

    # Register example pages
    register_example_page()
    print()
    register_simple_dashboard()

    print()
    print("=" * 70)
    print("Pages registered successfully!")
    print()
    print("Access pages at:")
    print("  - http://localhost:5001/page/cyber_threat_retrieval")
    print("  - http://localhost:5001/page/simple_dashboard")
    print()
    print("View all pages:")
    print("  - GET http://localhost:5001/api/pages")
    print("=" * 70)
