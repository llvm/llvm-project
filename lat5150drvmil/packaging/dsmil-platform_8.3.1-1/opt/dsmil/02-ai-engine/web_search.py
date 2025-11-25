#!/usr/bin/env python3
"""
Web Search Module - DuckDuckGo + Google Integration

Privacy-first web search for current information:
- DuckDuckGo (primary - no tracking)
- Google Custom Search (backup - if API key provided)
- Integrates results with local AI analysis
"""

import json
from typing import List, Dict, Optional

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️  duckduckgo-search not installed. Run: pip3 install duckduckgo-search")

class WebSearch:
    def __init__(self):
        self.ddgs_available = DDGS_AVAILABLE
        self.google_api_key = None  # Set if available

    def search(self, query: str, max_results: int = 5, prefer_google: bool = False) -> Dict:
        """
        Search the web (privacy-first with DuckDuckGo)

        Args:
            query: Search query
            max_results: Number of results to return
            prefer_google: Use Google if available (default: DuckDuckGo)

        Returns:
            Dict with results or error
        """
        # Try DuckDuckGo first (privacy-first)
        if self.ddgs_available and not prefer_google:
            return self._search_duckduckgo(query, max_results)

        # Fallback to Google if API key available
        elif self.google_api_key:
            return self._search_google(query, max_results)

        # No search available
        else:
            return {
                "error": "No search backend available",
                "suggestion": "Install: pip3 install duckduckgo-search"
            }

    def _search_duckduckgo(self, query: str, max_results: int) -> Dict:
        """Search using DuckDuckGo (privacy-first)"""
        try:
            with DDGS() as ddgs:
                results = []

                # Get search results
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'source': 'duckduckgo'
                    })

                return {
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "source": "DuckDuckGo",
                    "privacy": "high"
                }

        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "source": "duckduckgo"
            }

    def _search_google(self, query: str, max_results: int) -> Dict:
        """Search using Google Custom Search API (if key available)"""
        # TODO: Implement Google Custom Search
        # Requires API key setup
        return {
            "error": "Google search not yet implemented",
            "suggestion": "Using DuckDuckGo instead"
        }

    def summarize_results(self, search_results: Dict, ai_engine) -> str:
        """
        Use AI to summarize search results

        Args:
            search_results: Results from search()
            ai_engine: DSMILAIEngine instance

        Returns:
            AI-generated summary of search results
        """
        if 'error' in search_results or search_results.get('count', 0) == 0:
            return "No search results found."

        # Build context from results
        context = f"Search query: {search_results['query']}\n\nTop results:\n\n"

        for i, result in enumerate(search_results['results'], 1):
            context += f"{i}. {result['title']}\n"
            context += f"   {result['snippet']}\n"
            context += f"   Source: {result['url']}\n\n"

        # Ask AI to synthesize
        summary_prompt = f"""{context}

Based on these search results, provide a comprehensive answer to: {search_results['query']}

Include:
1. Summary of key information
2. Mention sources where appropriate
3. Note any contradictions or uncertainties"""

        result = ai_engine.generate(summary_prompt, model_selection="fast")

        return result.get('response', 'Error generating summary')

    def search_and_synthesize(self, query: str, ai_engine, max_results: int = 5) -> Dict:
        """
        Complete workflow: Search web + AI analysis

        Returns complete response with search results + AI synthesis
        """
        # Search web
        search_results = self.search(query, max_results)

        if 'error' in search_results:
            return {
                "search_results": None,
                "ai_summary": "Web search unavailable",
                "error": search_results['error']
            }

        # Synthesize with AI
        ai_summary = self.summarize_results(search_results, ai_engine)

        return {
            "search_results": search_results,
            "ai_summary": ai_summary,
            "sources": [r['url'] for r in search_results['results']],
            "source_count": search_results['count']
        }

# CLI
if __name__ == "__main__":
    import sys

    searcher = WebSearch()

    if len(sys.argv) < 2:
        print("Web Search - Usage:")
        print("  python3 web_search.py 'latest AI breakthroughs'")
        print("  python3 web_search.py 'news about quantum computing'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    print(f"\n🌐 Searching: {query}\n")

    results = searcher.search(query)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"Found {results['count']} results from {results['source']}:\n")
        for i, r in enumerate(results['results'], 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['snippet'][:100]}...")
            print(f"   {r['url']}\n")
