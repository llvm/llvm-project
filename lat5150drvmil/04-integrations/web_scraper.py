#!/usr/bin/env python3
"""
Web Scraper - URL Content Extraction & Auto-RAG Integration

Fetch webpage content, extract main text, auto-add to RAG knowledge base.
Perfect for research: paste arxiv/paper URLs → instant knowledge base addition.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent))
from rag_manager import RAGManager

class WebScraper:
    def __init__(self, verify_ssl: bool = False):
        """
        Initialize web scraper

        Args:
            verify_ssl: Verify SSL certificates (False for research sites with self-signed certs)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) DSMIL-Research-Bot/1.0'
        })
        self.verify_ssl = verify_ssl
        self.rag = RAGManager()

        # Disable SSL warnings if not verifying
        if not verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def scrape_url(self, url: str, auto_add_to_rag: bool = True) -> dict:
        """
        Scrape URL and extract main content (handles PDFs and HTML)

        Args:
            url: Web URL to scrape
            auto_add_to_rag: Automatically add to RAG database

        Returns:
            Dict with content, metadata, and RAG status
        """
        try:
            # Check if PDF
            if url.lower().endswith('.pdf'):
                return self._scrape_pdf(url, auto_add_to_rag)

            # Fetch URL (with SSL verification setting)
            response = self.session.get(url, timeout=30, verify=self.verify_ssl)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)

            # Extract title
            title = soup.find('title').get_text() if soup.find('title') else url

            # Create metadata
            metadata = {
                "url": url,
                "title": title,
                "content_length": len(clean_text),
                "word_count": len(clean_text.split()),
                "scraped_successfully": True
            }

            # Auto-add to RAG if enabled
            if auto_add_to_rag:
                # Save to temp file for RAG ingestion
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                temp_file = Path(f"/tmp/scraped_{url_hash}.md")

                # Write with metadata
                content_with_meta = f"""# {title}
Source: {url}
Scraped: {metadata['content_length']} chars, {metadata['word_count']} words

---

{clean_text}
"""
                temp_file.write_text(content_with_meta)

                # Add to RAG
                rag_result = self.rag.add_file(str(temp_file))

                metadata['rag_added'] = rag_result.get('status') == 'success'
                metadata['rag_tokens'] = rag_result.get('tokens', 0)
                metadata['temp_file'] = str(temp_file)

            return {
                "status": "success",
                "url": url,
                "title": title,
                "content": clean_text,
                "metadata": metadata
            }

        except requests.RequestException as e:
            return {
                "error": f"Failed to fetch URL: {str(e)}",
                "url": url
            }
        except Exception as e:
            return {
                "error": f"Scraping error: {str(e)}",
                "url": url
            }

    def scrape_multiple(self, urls: list, auto_add_to_rag: bool = True) -> dict:
        """
        Scrape multiple URLs

        Args:
            urls: List of URLs
            auto_add_to_rag: Add all to RAG

        Returns:
            Dict with results for each URL
        """
        results = {
            "total": len(urls),
            "success": 0,
            "failed": 0,
            "urls": []
        }

        for url in urls:
            print(f"Scraping: {url}")
            result = self.scrape_url(url, auto_add_to_rag=auto_add_to_rag)

            if result.get('status') == 'success':
                results['success'] += 1
                results['urls'].append({
                    "url": url,
                    "title": result['title'],
                    "tokens": result['metadata'].get('rag_tokens', 0)
                })
            else:
                results['failed'] += 1
                results['urls'].append({
                    "url": url,
                    "error": result.get('error')
                })

        return results

    def extract_links(self, url: str, filter_pattern: str = None) -> list:
        """Extract all links from a webpage"""
        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')

            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']

                # Make absolute URL
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)

                # Filter if pattern provided
                if filter_pattern and filter_pattern not in href:
                    continue

                links.append({
                    'url': href,
                    'text': a.get_text(strip=True)
                })

            return links

        except Exception as e:
            return []

    def scrape_arxiv(self, arxiv_id: str) -> dict:
        """
        Scrape arXiv paper (specialized)

        Args:
            arxiv_id: arXiv ID (e.g., "2401.12345")

        Returns:
            Paper content and metadata
        """
        url = f"https://arxiv.org/abs/{arxiv_id}"
        return self.scrape_url(url, auto_add_to_rag=True)

    def _scrape_pdf(self, pdf_url: str, auto_add_to_rag: bool = True) -> dict:
        """
        Scrape PDF content (using pdftotext or strings)

        Args:
            pdf_url: URL to PDF file
            auto_add_to_rag: Add to RAG after extraction

        Returns:
            Dict with PDF content and metadata
        """
        import tempfile
        import subprocess

        try:
            # Download PDF (with SSL verification setting)
            response = self.session.get(pdf_url, timeout=60, verify=self.verify_ssl)
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Extract text with pdftotext (fallback to strings)
            try:
                result = subprocess.run(['pdftotext', tmp_path, '-'],
                                      capture_output=True, text=True, timeout=60)
                text = result.stdout if result.returncode == 0 else ""
            except:
                # Fallback to strings
                result = subprocess.run(['strings', tmp_path],
                                      capture_output=True, text=True, timeout=60)
                text = result.stdout

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            if not text or len(text) < 100:
                return {"error": "No text extracted from PDF", "url": pdf_url}

            # Get title from filename
            title = pdf_url.split('/')[-1].replace('.pdf', '')

            # Auto-add to RAG
            if auto_add_to_rag:
                url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
                temp_file = Path(f"/tmp/pdf_{url_hash}.md")

                content_with_meta = f"""# {title} (PDF)
Source: {pdf_url}
Extracted: {len(text)} chars

---

{text}
"""
                temp_file.write_text(content_with_meta)

                # Add to RAG
                rag_result = self.rag.add_file(str(temp_file))

                return {
                    "status": "success",
                    "url": pdf_url,
                    "title": title,
                    "content": text,
                    "metadata": {
                        "content_length": len(text),
                        "word_count": len(text.split()),
                        "rag_added": rag_result.get('status') == 'success',
                        "rag_tokens": rag_result.get('tokens', 0),
                        "temp_file": str(temp_file),
                        "type": "pdf"
                    }
                }

            return {
                "status": "success",
                "url": pdf_url,
                "title": title,
                "content": text,
                "metadata": {"type": "pdf"}
            }

        except Exception as e:
            return {
                "error": f"PDF scraping failed: {str(e)}",
                "url": pdf_url
            }

    def browse_and_index(self, start_url: str, max_pages: int = 50, same_domain: bool = True, depth_limit: int = 3, smart_filter: bool = True) -> dict:
        """
        Intelligently browse website areas and index content

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain: Only crawl pages from same domain
            depth_limit: Maximum link depth to follow
            smart_filter: Filter out navigation/ads/boilerplate

        Returns:
            Dict with crawl results
        """
        # Alias for backward compatibility
        return self.crawl_and_index(start_url, max_pages, same_domain, depth_limit, smart_filter)

    def crawl_and_index(self, start_url: str, max_pages: int = 50, same_domain: bool = True, depth_limit: int = 3, smart_filter: bool = True) -> dict:
        """
        Crawl website starting from URL and index all pages to RAG

        Intelligent browsing:
        - Detects content vs navigation areas
        - Prioritizes PDFs and documents
        - Filters out ads, menus, footers
        - Identifies article/content sections

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain: Only crawl pages from same domain
            depth_limit: Maximum link depth to follow
            smart_filter: Use intelligent filtering

        Returns:
            Dict with crawl results
        """
        from urllib.parse import urlparse, urljoin

        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        results = {
            "start_url": start_url,
            "pages_crawled": 0,
            "pages_indexed": 0,
            "total_tokens": 0,
            "errors": 0,
            "urls": [],
            "pdfs_found": 0,
            "content_pages": 0
        }

        start_domain = urlparse(start_url).netloc

        print(f"🕷️ Starting intelligent crawl from: {start_url}")
        print(f"   Max pages: {max_pages}, Max depth: {depth_limit}, Same domain: {same_domain}")
        print(f"   Smart filtering: {'ON' if smart_filter else 'OFF'}")

        while to_visit and results['pages_crawled'] < max_pages:
            current_url, depth = to_visit.pop(0)

            if current_url in visited:
                continue

            if depth > depth_limit:
                continue

            visited.add(current_url)
            results['pages_crawled'] += 1

            print(f"   [{results['pages_crawled']}/{max_pages}] Scraping: {current_url} (depth {depth})")

            # Scrape page
            page_result = self.scrape_url(current_url, auto_add_to_rag=True)

            if page_result.get('status') == 'success':
                results['pages_indexed'] += 1
                tokens = page_result['metadata'].get('rag_tokens', 0)
                results['total_tokens'] += tokens

                results['urls'].append({
                    'url': current_url,
                    'title': page_result['title'],
                    'tokens': tokens,
                    'depth': depth
                })

                # Extract links for next level
                if depth < depth_limit:
                    try:
                        response = self.session.get(current_url, timeout=30, verify=self.verify_ssl)
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Collect PDFs and content pages separately
                        pdf_links = []
                        content_links = []
                        nav_links = []  # Navigation/index pages

                        # Look for content areas (main, article, content divs)
                        content_areas = soup.find_all(['main', 'article']) or [soup]
                        for area in content_areas[:1]:  # Use first content area found
                            for a in area.find_all('a', href=True):
                                href = a['href']

                                # Skip anchors, javascript, mailto
                                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                                    continue

                                # Make absolute URL
                                if href.startswith('/'):
                                    href = urljoin(current_url, href)
                                elif not href.startswith('http'):
                                    href = urljoin(current_url, href)

                                # Filter by domain if required
                                if same_domain:
                                    link_domain = urlparse(href).netloc
                                    if link_domain != start_domain:
                                        continue

                                # Categorize link by type and purpose
                                link_lower = href.lower()
                                link_text = a.get_text(strip=True).lower()

                                # PDFs (highest priority)
                                if link_lower.endswith('.pdf'):
                                    pdf_links.append((href, 'pdf'))
                                    results['pdfs_found'] += 1

                                # Content indicators (high priority)
                                elif any(word in link_text for word in ['synthesis', 'procedure', 'method', 'preparation', 'reaction', 'chemistry']):
                                    content_links.append((href, 'content'))

                                # Index/navigation pages (lower priority)
                                elif any(word in link_text for word in ['index', 'list', 'archive', 'table of contents', 'menu']):
                                    nav_links.append((href, 'nav'))

                                # Regular pages
                                elif link_lower.endswith(('.html', '.htm', '.php')):
                                    content_links.append((href, 'page'))

                        # Priority order: PDFs > Content > Navigation
                        # PDFs first (always a sucker for pdf's)
                        for href, link_type in pdf_links:
                            if href not in visited and (href, depth + 1) not in to_visit:
                                to_visit.insert(0, (href, depth + 1))  # Front of queue

                        # Content pages next
                        for href, link_type in content_links:
                            if href not in visited and (href, depth + 1) not in to_visit:
                                to_visit.insert(len(pdf_links), (href, depth + 1))  # After PDFs

                        # Navigation/index pages last (but still important)
                        for href, link_type in nav_links:
                            if href not in visited and (href, depth + 1) not in to_visit:
                                to_visit.append((href, depth + 1))  # Back of queue

                    except Exception as e:
                        print(f"     ⚠ Failed to extract links: {e}")
                        pass  # Failed to extract links, continue

            else:
                results['errors'] += 1

        print(f"\n✅ Intelligent crawl complete!")
        print(f"   Pages crawled: {results['pages_crawled']}")
        print(f"   Pages indexed: {results['pages_indexed']}")
        print(f"   PDFs found: {results['pdfs_found']}")
        print(f"   Content pages: {results['content_pages']}")
        print(f"   Total tokens: {results['total_tokens']}")
        print(f"   Errors: {results['errors']}")

        return results

# CLI
if __name__ == "__main__":
    import json

    scraper = WebScraper()

    if len(sys.argv) < 2:
        print("Web Scraper - Usage:")
        print("  python3 web_scraper.py https://example.com")
        print("  python3 web_scraper.py arxiv 2401.12345")
        print("  python3 web_scraper.py links https://example.com")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "arxiv" and len(sys.argv) > 2:
        result = scraper.scrape_arxiv(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "links" and len(sys.argv) > 2:
        links = scraper.extract_links(sys.argv[2])
        for link in links[:20]:  # Show first 20
            print(f"{link['text']}: {link['url']}")

    elif cmd == "crawl" and len(sys.argv) > 2:
        # Crawl website
        url = sys.argv[2]
        max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        depth = int(sys.argv[4]) if len(sys.argv) > 4 else 3

        result = scraper.crawl_and_index(url, max_pages=max_pages, depth_limit=depth)
        print(json.dumps(result, indent=2))

    elif cmd.startswith('http'):
        # URL provided
        url = cmd
        print(f"\n🌐 Scraping: {url}\n")

        result = scraper.scrape_url(url, auto_add_to_rag=True)

        if result.get('status') == 'success':
            print(f"✓ Title: {result['title']}")
            print(f"✓ Content: {result['metadata']['content_length']} chars, {result['metadata']['word_count']} words")

            if result['metadata'].get('rag_added'):
                print(f"✓ Added to RAG: {result['metadata']['rag_tokens']} tokens")
                print(f"  Searchable in knowledge base!")
            else:
                print(f"  (Not added to RAG)")

            print(f"\nPreview:")
            print(result['content'][:500] + "...")
        else:
            print(f"❌ Error: {result.get('error')}")
    else:
        print("Usage:")
        print("  python3 web_scraper.py <url>")
        print("  python3 web_scraper.py crawl <url> [max_pages] [depth]")
        print("  python3 web_scraper.py arxiv 2401.12345")
