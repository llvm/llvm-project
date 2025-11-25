#!/usr/bin/env python3
"""
Shodan Search Module - IDOR-based Search Integration

Privacy-focused Shodan search using the facet endpoint:
- No authentication required (uses IDOR vulnerability documentation)
- Access to premium filters (vuln, tag, etc.)
- Integrates results with local AI analysis

Reference: https://github.com/sahar042/Shodan-IDOR

DISCLAIMER: This implementation uses publicly documented API endpoints.
Users should comply with Shodan's Terms of Service and applicable laws.
"""

import json
import urllib.request
import urllib.parse
from typing import List, Dict, Optional, Union

class ShodanSearch:
    """
    Shodan search engine using facet endpoint

    Provides access to Shodan data for cybersecurity research,
    vulnerability assessment, and threat intelligence.
    """

    BASE_URL = "https://www.shodan.io/search/facet"

    # Common facet types for grouping results
    FACET_TYPES = {
        'ip': 'IP Address',
        'country': 'Country',
        'city': 'City',
        'org': 'Organization',
        'domain': 'Domain',
        'port': 'Port',
        'asn': 'ASN',
        'product': 'Product',
        'version': 'Version',
        'os': 'Operating System'
    }

    def __init__(self):
        """Initialize Shodan search client"""
        self.session_headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.shodan.io/',
            'X-Requested-With': 'XMLHttpRequest'
        }

    def search(self,
               query: str,
               facet: str = 'ip',
               max_results: int = 100) -> Dict:
        """
        Search Shodan using facet endpoint

        Args:
            query: Shodan search query (e.g., "apache", "vuln:CVE-2021-44228")
            facet: Result grouping type (ip, country, org, port, etc.)
            max_results: Maximum results to return (default: 100)

        Returns:
            Dict with search results or error

        Examples:
            # Search for vulnerable systems
            search("vuln:CVE-2021-44228", facet="ip")

            # Search for honeypots
            search("tag:honeypot", facet="country")

            # Search for specific service
            search("product:apache", facet="version")
        """
        try:
            # Validate facet type
            if facet not in self.FACET_TYPES:
                return {
                    "error": f"Invalid facet type: {facet}",
                    "valid_facets": list(self.FACET_TYPES.keys())
                }

            # Build URL with query parameters
            params = {
                'query': query,
                'facet': facet
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

            # Make request
            req = urllib.request.Request(url, headers=self.session_headers)

            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read().decode('utf-8')
                result = json.loads(data)

                # Parse and format results
                return self._format_results(result, query, facet, max_results)

        except urllib.error.HTTPError as e:
            return {
                "error": f"HTTP {e.code}: {e.reason}",
                "query": query,
                "source": "shodan",
                "suggestion": "Check query syntax or try again later"
            }
        except urllib.error.URLError as e:
            return {
                "error": f"Network error: {e.reason}",
                "query": query,
                "source": "shodan"
            }
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON response: {str(e)}",
                "query": query,
                "source": "shodan"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "query": query,
                "source": "shodan"
            }

    def _format_results(self,
                        data: Dict,
                        query: str,
                        facet: str,
                        max_results: int) -> Dict:
        """Format Shodan API response into standardized structure"""

        # Extract facet data (format varies by Shodan response)
        results = []

        if isinstance(data, dict):
            # Handle different response formats
            if 'facets' in data:
                facet_data = data.get('facets', {}).get(facet, [])
            elif 'matches' in data:
                facet_data = data.get('matches', [])
            elif isinstance(data, list):
                facet_data = data
            else:
                # Direct response (most common)
                facet_data = data.get('values', data.get('results', []))
        elif isinstance(data, list):
            facet_data = data
        else:
            facet_data = []

        # Process results
        for item in facet_data[:max_results]:
            if isinstance(item, dict):
                results.append({
                    'value': item.get('value', item.get('name', 'Unknown')),
                    'count': item.get('count', item.get('total', 0)),
                    'facet': facet,
                    'raw': item  # Preserve raw data for advanced use
                })
            else:
                # Handle simple list format
                results.append({
                    'value': str(item),
                    'count': 1,
                    'facet': facet
                })

        return {
            "query": query,
            "facet": facet,
            "facet_name": self.FACET_TYPES.get(facet, facet),
            "results": results,
            "count": len(results),
            "source": "Shodan",
            "endpoint": "facet",
            "privacy": "Public endpoint (no auth required)"
        }

    def search_vulnerability(self, cve_id: str, facet: str = 'ip') -> Dict:
        """
        Search for systems vulnerable to specific CVE

        Args:
            cve_id: CVE identifier (e.g., "CVE-2021-44228")
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = f"vuln:{cve_id}"
        return self.search(query, facet)

    def search_honeypots(self, facet: str = 'country') -> Dict:
        """
        Search for honeypot systems

        Args:
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = "tag:honeypot"
        return self.search(query, facet)

    def search_compromised(self, facet: str = 'country') -> Dict:
        """
        Search for potentially compromised systems

        Args:
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = "tag:compromised"
        return self.search(query, facet)

    def search_product(self, product: str, version: Optional[str] = None,
                       facet: str = 'ip') -> Dict:
        """
        Search for specific product/software

        Args:
            product: Product name (e.g., "apache", "nginx")
            version: Optional version filter
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = f"product:{product}"
        if version:
            query += f" version:{version}"
        return self.search(query, facet)

    def search_country(self, country_code: str, additional_query: str = "",
                       facet: str = 'org') -> Dict:
        """
        Search within specific country

        Args:
            country_code: Two-letter country code (e.g., "US", "CN")
            additional_query: Additional search filters
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = f"country:{country_code}"
        if additional_query:
            query += f" {additional_query}"
        return self.search(query, facet)

    def search_port(self, port: int, facet: str = 'product') -> Dict:
        """
        Search for specific port

        Args:
            port: Port number
            facet: Result grouping type

        Returns:
            Search results dict
        """
        query = f"port:{port}"
        return self.search(query, facet)

    def summarize_results(self, search_results: Dict, ai_engine) -> str:
        """
        Use AI to analyze and summarize Shodan search results

        Args:
            search_results: Results from search()
            ai_engine: DSMILAIEngine instance

        Returns:
            AI-generated analysis of search results
        """
        if 'error' in search_results or search_results.get('count', 0) == 0:
            return "No search results found or error occurred."

        # Build context from results
        context = f"Shodan Search Analysis\n"
        context += f"Query: {search_results['query']}\n"
        context += f"Grouping: {search_results['facet_name']}\n"
        context += f"Total Results: {search_results['count']}\n\n"
        context += "Top findings:\n\n"

        for i, result in enumerate(search_results['results'][:20], 1):
            context += f"{i}. {result['value']}: {result['count']} instances\n"

        # Ask AI to analyze
        analysis_prompt = f"""{context}

Based on these Shodan search results, provide:

1. **Summary**: What do these results indicate?
2. **Risk Assessment**: What are the security implications?
3. **Key Findings**: Most notable items from the results
4. **Recommendations**: What actions should be considered?

Focus on cybersecurity and threat intelligence perspective."""

        result = ai_engine.generate(analysis_prompt, model_selection="fast")

        return result.get('response', 'Error generating analysis')

    def search_and_analyze(self, query: str, ai_engine,
                           facet: str = 'ip', max_results: int = 100) -> Dict:
        """
        Complete workflow: Search Shodan + AI analysis

        Args:
            query: Shodan search query
            ai_engine: DSMILAIEngine instance
            facet: Result grouping type
            max_results: Maximum results

        Returns:
            Complete response with search results + AI analysis
        """
        # Search Shodan
        search_results = self.search(query, facet, max_results)

        if 'error' in search_results:
            return {
                "search_results": None,
                "ai_analysis": "Shodan search failed",
                "error": search_results['error']
            }

        # Analyze with AI
        ai_analysis = self.summarize_results(search_results, ai_engine)

        return {
            "search_results": search_results,
            "ai_analysis": ai_analysis,
            "query": query,
            "facet": facet,
            "result_count": search_results['count']
        }


# CLI interface
if __name__ == "__main__":
    import sys

    searcher = ShodanSearch()

    if len(sys.argv) < 2:
        print("Shodan Search - Usage:")
        print("  python3 shodan_search.py 'apache'")
        print("  python3 shodan_search.py 'vuln:CVE-2021-44228' --facet=country")
        print("  python3 shodan_search.py 'port:22' --facet=product")
        print("\nAvailable facets:", ', '.join(ShodanSearch.FACET_TYPES.keys()))
        sys.exit(1)

    # Parse arguments
    query = sys.argv[1]
    facet = 'ip'

    for arg in sys.argv[2:]:
        if arg.startswith('--facet='):
            facet = arg.split('=', 1)[1]

    print(f"\n🔍 Searching Shodan: {query}")
    print(f"📊 Grouping by: {facet}\n")

    results = searcher.search(query, facet)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Found {results['count']} {results['facet_name']} results:\n")

        for i, r in enumerate(results['results'][:20], 1):
            print(f"{i:3d}. {r['value']:30s} - {r['count']:6d} instances")

        if results['count'] > 20:
            print(f"\n... and {results['count'] - 20} more results")
