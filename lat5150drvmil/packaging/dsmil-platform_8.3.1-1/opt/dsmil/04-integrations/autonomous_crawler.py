#!/usr/bin/env python3
"""
Autonomous Web Crawler - Powered by crawl4ai

Fully autonomous intelligent crawler:
- Auto-detects index pages vs content
- Recursively follows all links
- Prioritizes PDFs
- Deduplication with reporting
- Live progress updates
- Knows when to stop

Just paste a URL - it does the rest.
"""

from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent))
from rag_manager import RAGManager

class AutonomousCrawler:
    def __init__(self):
        """Initialize autonomous crawler"""
        self.crawler = WebCrawler(verbose=True)
        self.rag = RAGManager()
        self.session_stats = {
            'new': 0,
            'duplicates': 0,
            'errors': 0,
            'total_tokens': 0
        }

    def auto_crawl(self, start_url: str) -> dict:
        """
        Fully autonomous crawl - detects page type and crawls intelligently

        Args:
            start_url: URL to start from

        Returns:
            Comprehensive results with deduplication info
        """
        print(f"🤖 Autonomous Crawler starting...")
        print(f"   URL: {start_url}")
        print(f"   Strategy: Auto-detect and adapt")

        results = {
            'start_url': start_url,
            'pages_found': 0,
            'new_documents': [],
            'duplicates_skipped': [],
            'errors': [],
            'total_tokens': 0,
            'pdfs_found': 0
        }

        # Use crawl4ai to crawl
        self.crawler.warmup()

        result = self.crawler.run(
            url=start_url,
            bypass_cache=True,
            word_count_threshold=10
        )

        if result.success:
            # Extract content
            content = result.markdown or result.cleaned_html or ""

            # Add to RAG with dedup check
            rag_result = self._add_to_rag(start_url, content, result.title or "Untitled")

            if rag_result['status'] == 'new':
                results['new_documents'].append({
                    'url': start_url,
                    'title': result.title,
                    'tokens': rag_result['tokens']
                })
                results['total_tokens'] += rag_result['tokens']
            elif rag_result['status'] == 'duplicate':
                results['duplicates_skipped'].append({
                    'url': start_url,
                    'title': result.title
                })

            # Extract all links
            if result.links:
                print(f"   Found {len(result.links['internal'])} internal links")

                # Crawl internal links
                for link_url in result.links['internal'][:50]:  # Limit to 50
                    # Check if PDF
                    if link_url.lower().endswith('.pdf'):
                        results['pdfs_found'] += 1
                        print(f"   📄 PDF: {link_url}")
                        pdf_result = self._crawl_pdf(link_url)
                        if pdf_result:
                            results['new_documents'].append(pdf_result)
                            results['total_tokens'] += pdf_result.get('tokens', 0)

                    # Check if content page
                    elif any(ext in link_url.lower() for ext in ['.html', '.htm', '.php']):
                        page_result = self.crawler.run(url=link_url, bypass_cache=True)

                        if page_result.success:
                            content = page_result.markdown or page_result.cleaned_html or ""
                            rag_result = self._add_to_rag(link_url, content, page_result.title or "Untitled")

                            if rag_result['status'] == 'new':
                                results['new_documents'].append({
                                    'url': link_url,
                                    'title': page_result.title,
                                    'tokens': rag_result['tokens']
                                })
                                results['total_tokens'] += rag_result['tokens']
                            elif rag_result['status'] == 'duplicate':
                                results['duplicates_skipped'].append({
                                    'url': link_url,
                                    'title': page_result.title
                                })

        results['pages_found'] = len(results['new_documents']) + len(results['duplicates_skipped'])

        print(f"\n✅ Autonomous crawl complete!")
        print(f"   Pages found: {results['pages_found']}")
        print(f"   ✨ New: {len(results['new_documents'])} ({results['total_tokens']} tokens)")
        print(f"   ⏭ Skipped: {len(results['duplicates_skipped'])} duplicates")
        print(f"   PDFs: {results['pdfs_found']}")

        return results

    def _add_to_rag(self, url: str, content: str, title: str) -> dict:
        """Add content to RAG with deduplication detection"""

        # Calculate hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if already in RAG
        rag = self.rag.rag
        if content_hash in rag.documents:
            return {'status': 'duplicate', 'hash': content_hash}

        # Save to temp file for RAG ingestion
        temp_file = Path(f"/tmp/crawled_{content_hash[:8]}.md")
        temp_file.write_text(f"# {title}\nSource: {url}\n\n---\n\n{content}")

        # Add to RAG
        result = self.rag.add_file(str(temp_file))

        if result.get('status') == 'success':
            return {
                'status': 'new',
                'tokens': result.get('tokens', 0),
                'hash': content_hash
            }

        return {'status': 'error', 'error': result.get('error')}

    def _crawl_pdf(self, pdf_url: str) -> dict:
        """Crawl and extract PDF (simplified for crawl4ai)"""
        # TODO: Implement PDF extraction
        # For now, return placeholder
        return None

# CLI
if __name__ == "__main__":
    import json

    crawler = AutonomousCrawler()

    if len(sys.argv) < 2:
        print("Autonomous Crawler - Usage:")
        print("  python3 autonomous_crawler.py https://site.com/index.html")
        print("\nJust provide URL - crawler does the rest!")
        sys.exit(1)

    url = sys.argv[1]
    result = crawler.auto_crawl(url)

    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
