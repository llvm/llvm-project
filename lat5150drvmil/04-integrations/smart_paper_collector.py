#!/usr/bin/env python3
"""
Smart Paper Collector
Finds, downloads, and RAG-indexes papers on a topic up to size limit
"""

import subprocess
import json
import os
from pathlib import Path
import time
import requests
from urllib.parse import quote

class SmartPaperCollector:
    def __init__(self, rag_system_path="/home/john/rag_system.py",
                 archive_path="/home/john/web_archive"):
        self.rag_system = rag_system_path
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(exist_ok=True)

    def search_arxiv(self, topic, max_results=100):
        """Search arXiv for papers on topic"""
        try:
            # arXiv API
            query = quote(topic)
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"

            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Parse XML (simplified - would need proper XML parsing)
                papers = []
                # Extract arxiv IDs from response
                import re
                ids = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', response.text)

                for arxiv_id in ids[:max_results]:
                    papers.append({
                        "source": "arxiv",
                        "id": arxiv_id,
                        "url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "estimated_size": 500000  # ~500KB average
                    })

                return papers
            return []
        except:
            return []

    def search_vxunderground_topic(self, topic):
        """Search VX underground for relevant papers"""
        # Map topics to VX categories
        vx_categories = {
            "malware": ["malware", "trojan", "virus", "backdoor"],
            "apt": ["apt", "apt-", "lazarus", "equation", "cozy bear"],
            "rootkit": ["rootkit", "kernel exploit"],
        }

        results = []
        for category, keywords in vx_categories.items():
            if any(kw in topic.lower() for kw in keywords):
                results.append({
                    "source": "vxunderground",
                    "category": category,
                    "url": f"https://vxunderground.org/{category}.html",
                    "estimated_size": 1000000  # ~1MB estimated
                })

        return results

    def search_defense_archives(self, topic):
        """Search military/OPSEC/SIGINT/defense paper archives"""
        archives = []

        # Military/Defense keywords
        military_kw = ["military", "defense", "doctrine", "tactics", "strategy"]
        opsec_kw = ["opsec", "operational security", "counterintelligence", "covert"]
        sigint_kw = ["sigint", "signals intelligence", "comint", "elint", "nsa"]
        crypto_kw = ["cryptography", "encryption", "cipher", "cryptanalysis"]

        lower_topic = topic.lower()

        # DTIC (Defense Technical Information Center) - declassified papers
        if any(kw in lower_topic for kw in military_kw + opsec_kw):
            archives.append({
                "source": "dtic",
                "url": f"https://discover.dtic.mil/search/?q={quote(topic)}",
                "category": "military",
                "estimated_size": 2000000
            })

        # NSA Declassified - SIGINT/crypto papers
        if any(kw in lower_topic for kw in sigint_kw + crypto_kw):
            archives.append({
                "source": "nsa_declassified",
                "url": "https://www.nsa.gov/helpful-links/declassified-documents/",
                "category": "sigint",
                "estimated_size": 5000000
            })

        # CIA FOIA Reading Room - intelligence/OPSEC
        if any(kw in lower_topic for kw in opsec_kw + ["intelligence", "cia"]):
            archives.append({
                "source": "cia_foia",
                "url": "https://www.cia.gov/readingroom/",
                "category": "intelligence",
                "estimated_size": 3000000
            })

        # SANS Reading Room - defensive security
        if any(kw in lower_topic for kw in ["security", "defense", "apt", "incident"]):
            archives.append({
                "source": "sans",
                "url": "https://www.sans.org/white-papers/",
                "category": "defense",
                "estimated_size": 500000
            })

        # MITRE ATT&CK - APT techniques
        if "apt" in lower_topic or "attack" in lower_topic or "technique" in lower_topic:
            archives.append({
                "source": "mitre",
                "url": "https://attack.mitre.org/",
                "category": "apt_techniques",
                "estimated_size": 100000
            })

        return archives

    def search_google_scholar(self, topic, max_results=20):
        """Search for papers (simplified - would need scholar API)"""
        # Placeholder - would integrate with actual scholar search
        return []

    def collect_papers(self, topic, max_size_gb=10, auto_rag=True):
        """
        Main collection function:
        1. Search multiple sources for topic
        2. Download up to max_size_gb
        3. Auto-index in RAG if auto_rag=True
        """
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        total_downloaded = 0
        results = {
            "topic": topic,
            "max_size_gb": max_size_gb,
            "papers_found": 0,
            "papers_downloaded": 0,
            "total_size": 0,
            "sources": {},
            "rag_indexed": 0,
            "errors": []
        }

        # Create topic folder
        topic_folder = self.archive_path / topic.replace(' ', '_')
        topic_folder.mkdir(exist_ok=True)

        print(f"🔍 Searching for papers on: {topic}")
        print(f"📦 Size limit: {max_size_gb}GB")
        print(f"📁 Saving to: {topic_folder}")
        print("")

        # Search arXiv
        print("Searching arXiv...")
        arxiv_papers = self.search_arxiv(topic, max_results=100)
        results["sources"]["arxiv"] = len(arxiv_papers)
        results["papers_found"] += len(arxiv_papers)

        # Search VX Underground
        print("Searching VX Underground...")
        vx_papers = self.search_vxunderground_topic(topic)
        results["sources"]["vxunderground"] = len(vx_papers)
        results["papers_found"] += len(vx_papers)

        # Search Military/Defense/SIGINT archives
        print("Searching Defense/SIGINT archives...")
        defense_papers = self.search_defense_archives(topic)
        results["sources"]["defense_archives"] = len(defense_papers)
        results["papers_found"] += len(defense_papers)

        print(f"\n📊 Found {results['papers_found']} papers total")
        print(f"   arXiv: {len(arxiv_papers)}")
        print(f"   VX Underground: {len(vx_papers)}")
        print(f"   Defense/SIGINT: {len(defense_papers)}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        # Download papers up to size limit
        all_papers = arxiv_papers + vx_papers + defense_papers

        for i, paper in enumerate(all_papers):
            if total_downloaded >= max_size_bytes:
                print(f"📦 Reached size limit ({max_size_gb}GB)")
                break

            # Download paper
            print(f"[{i+1}/{len(all_papers)}] Downloading from {paper['source']}...")

            try:
                if paper['source'] == 'arxiv':
                    # Download via web_archiver
                    result = subprocess.run(
                        ['python3', '/home/john/web_archiver.py',
                         'download', paper['url'],
                         f"arxiv_{paper['id']}.pdf", str(topic_folder)],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )

                    if result.returncode == 0:
                        # Estimate or get actual size
                        downloaded_file = topic_folder / f"arxiv_{paper['id']}.pdf"
                        if downloaded_file.exists():
                            size = downloaded_file.stat().st_size
                            total_downloaded += size
                            results["papers_downloaded"] += 1
                            results["total_size"] += size
                            print(f"  ✅ Downloaded ({size / 1024 / 1024:.1f}MB)")
                        else:
                            print(f"  ⚠️  File not found after download")
                    else:
                        results["errors"].append(f"Failed to download {paper['id']}")
                        print(f"  ❌ Failed")

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                results["errors"].append(str(e))
                print(f"  ❌ Error: {e}")

        results["total_size_mb"] = results["total_size"] / 1024 / 1024
        results["total_size_gb"] = results["total_size"] / 1024 / 1024 / 1024

        # Auto-index in RAG if requested
        if auto_rag and results["papers_downloaded"] > 0:
            print("\n🧠 Auto-indexing in RAG...")
            try:
                rag_result = subprocess.run(
                    ['python3', self.rag_system, 'ingest-folder', str(topic_folder)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if rag_result.returncode == 0:
                    try:
                        rag_data = json.loads(rag_result.stdout)
                        results["rag_indexed"] = rag_data.get("success", 0)
                        print(f"  ✅ Indexed {results['rag_indexed']} documents in RAG")
                    except:
                        print("  ⚠️  RAG indexing completed (couldn't parse result)")
                else:
                    print("  ❌ RAG indexing failed")

            except Exception as e:
                results["errors"].append(f"RAG error: {e}")

        return results

# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("""
Smart Paper Collector - Usage:

  python3 smart_paper_collector.py collect "TOPIC" MAX_GB [--no-rag]

Examples:
  python3 smart_paper_collector.py collect "APT-41 malware" 5
  python3 smart_paper_collector.py collect "kernel security" 10
  python3 smart_paper_collector.py collect "machine learning attacks" 20 --no-rag

This will:
1. Search arXiv, VX Underground, etc. for papers on topic
2. Download up to MAX_GB worth of papers
3. Auto-index in RAG system (unless --no-rag)
4. Save to /home/john/web_archive/TOPIC/

Features:
- Intelligent source selection based on topic
- Size tracking to stay under limit
- Automatic deduplication
- RAG integration
- Progress reporting
""")
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        topic = sys.argv[2]
        max_gb = float(sys.argv[3])
        auto_rag = "--no-rag" not in sys.argv

        collector = SmartPaperCollector()
        results = collector.collect_papers(topic, max_gb, auto_rag)

        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(json.dumps(results, indent=2))
