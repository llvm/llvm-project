#!/usr/bin/env python3
"""
Crawl4AI Integration - Industrial-strength web crawling

Uses crawl4ai (https://github.com/unclecode/crawl4ai) for:
- JavaScript rendering
- Smart content extraction
- Better link detection
- Async performance
- Auto-tokenization and RAG indexing
"""

import sys
import asyncio
from pathlib import Path

# Add crawl4ai to path
sys.path.insert(0, '/home/john/crawl4ai')
sys.path.insert(0, str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler
from rag_manager import RAGManager
import hashlib

class Crawl4AIWrapper:
    def __init__(self):
        """Initialize crawl4ai wrapper"""
        self.rag = RAGManager()

    async def crawl_and_index(self, url: str, max_pages: int = 50) -> dict:
        """
        Crawl URL and all linked pages, auto-index to RAG

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl

        Returns:
            Results with deduplication info
        """
        results = {
            'url': url,
            'pages_crawled': 0,
            'new_documents': [],
            'duplicates_skipped': [],
            'total_tokens': 0,
            'errors': []
        }

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Crawl starting page
            result = await crawler.arun(url=url)

            if result.success:
                # Add to RAG
                rag_result = self._add_to_rag(url, result.markdown, result.metadata.get('title', 'Untitled'))

                if rag_result['status'] == 'new':
                    results['new_documents'].append({
                        'url': url,
                        'title': result.metadata.get('title'),
                        'tokens': rag_result['tokens']
                    })
                    results['total_tokens'] += rag_result['tokens']
                else:
                    results['duplicates_skipped'].append(url)

                # Extract and crawl links
                links = result.links.get('internal', [])
                print(f"Found {len(links)} internal links")

                for link in links[:max_pages]:
                    try:
                        link_result = await crawler.arun(url=link)

                        if link_result.success:
                            rag_result = self._add_to_rag(
                                link,
                                link_result.markdown,
                                link_result.metadata.get('title', 'Untitled')
                            )

                            if rag_result['status'] == 'new':
                                results['new_documents'].append({
                                    'url': link,
                                    'title': link_result.metadata.get('title'),
                                    'tokens': rag_result['tokens']
                                })
                                results['total_tokens'] += rag_result['tokens']
                            else:
                                results['duplicates_skipped'].append(link)

                        results['pages_crawled'] += 1

                    except Exception as e:
                        results['errors'].append({'url': link, 'error': str(e)})

        return results

    def _add_to_rag(self, url: str, content: str, title: str) -> dict:
        """Add content to RAG with deduplication"""
        # Hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if already indexed
        if content_hash in self.rag.rag.documents:
            return {'status': 'duplicate', 'hash': content_hash}

        # Save to temp file
        temp_file = Path(f"/tmp/crawl4ai_{content_hash[:8]}.md")
        temp_file.write_text(f"# {title}\nSource: {url}\n\n{content}")

        # Add to RAG
        result = self.rag.add_file(str(temp_file))

        if result.get('status') == 'success':
            return {'status': 'new', 'tokens': result.get('tokens', 0)}

        return {'status': 'error'}

# CLI
async def main():
    if len(sys.argv) < 2:
        print("Crawl4AI Wrapper - Usage:")
        print("  python3 crawl4ai_wrapper.py https://site.com")
        sys.exit(1)

    url = sys.argv[1]
    crawler = Crawl4AIWrapper()

    print(f"🤖 Crawl4AI: {url}\n")
    results = await crawler.crawl_and_index(url)

    print(f"\n✅ Complete!")
    print(f"   Pages: {results['pages_crawled']}")
    print(f"   ✨ New: {len(results['new_documents'])} ({results['total_tokens']} tokens)")
    print(f"   ⏭ Duplicates: {len(results['duplicates_skipped'])}")
    print(f"   ❌ Errors: {len(results['errors'])}")

if __name__ == "__main__":
    asyncio.run(main())
