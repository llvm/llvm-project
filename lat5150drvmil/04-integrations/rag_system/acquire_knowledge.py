#!/usr/bin/env python3
"""
Knowledge Acquisition Script for LAT5150DRVMIL RAG System
Downloads public domain knowledge from legitimate sources

Topics: Security, Kernel Dev, SIGINT, Geopolitics, Malware Analysis, Intelligence
Sources: Academic papers, declassified docs, public reports, OSINT
"""

import os
import subprocess
from pathlib import Path
from typing import List

KNOWLEDGE_BASE = Path('00-documentation/General_Knowledge')


def setup_directories():
    """Create knowledge base directory structure"""
    dirs = [
        'Security',
        'Kernel_Development',
        'Malware_Analysis',
        'Geopolitics',
        'SIGINT_OSINT',
        'Threat_Intelligence',
        'Hardware_Security',
        'AI_ML',
    ]

    for d in dirs:
        (KNOWLEDGE_BASE / d).mkdir(parents=True, exist_ok=True)

    print(f"✓ Created knowledge base structure in {KNOWLEDGE_BASE}")


def download_security_awesome_lists():
    """Download curated security knowledge from GitHub Awesome lists"""
    repos = [
        ('https://github.com/sbilly/awesome-security', 'Security'),
        ('https://github.com/carpedm20/awesome-hacking', 'Security'),
        ('https://github.com/onlurking/awesome-infosec', 'Security'),
        ('https://github.com/hslatman/awesome-threat-intelligence', 'Threat_Intelligence'),
        ('https://github.com/rshipp/awesome-malware-analysis', 'Malware_Analysis'),
        ('https://github.com/The-Art-of-Hacking/h4cker', 'Security'),
        ('https://github.com/qazbnm456/awesome-cve-poc', 'Security'),
    ]

    print("\n📚 Downloading security knowledge repositories...")

    for repo_url, category in repos:
        repo_name = repo_url.split('/')[-1]
        target_dir = KNOWLEDGE_BASE / category / repo_name

        if target_dir.exists():
            print(f"  ✓ Already exists: {repo_name}")
            continue

        print(f"  Downloading: {repo_name}...")
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, str(target_dir)],
                capture_output=True,
                check=True
            )
            print(f"  ✓ Downloaded: {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {repo_name} - {e}")


def download_kernel_docs():
    """Download Linux kernel development documentation"""
    print("\n📚 Downloading kernel development docs...")

    # Kernel newbies wiki
    urls = [
        ('https://kernelnewbies.org/Documents', 'Kernel_Development/KernelNewbies.md'),
    ]

    for url, filename in urls:
        target = KNOWLEDGE_BASE / filename
        if target.exists():
            print(f"  ✓ Already exists: {filename}")
            continue

        print(f"  Downloading: {url}...")
        try:
            subprocess.run(['wget', '-q', '-O', str(target), url], check=True)
            print(f"  ✓ Downloaded: {filename}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def download_arxiv_papers():
    """Download relevant arXiv research papers"""
    print("\n📚 Downloading research papers from arXiv...")

    try:
        import arxiv
    except ImportError:
        print("  ⚠️  arxiv package not installed. Install with: pip3 install arxiv")
        return

    topics = [
        ('malware detection machine learning', 'Malware_Analysis', 10),
        ('neural processing unit NPU', 'AI_ML', 10),
        ('hardware security vulnerabilities', 'Hardware_Security', 10),
        ('kernel security linux', 'Kernel_Development', 10),
        ('side channel attacks', 'Hardware_Security', 10),
        ('adversarial machine learning', 'AI_ML', 10),
    ]

    for query, category, max_results in topics:
        print(f"  Searching: {query}...")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for paper in search.results():
            filename = f"{paper.title[:60].replace('/', '_')}.pdf"
            target = KNOWLEDGE_BASE / category / 'Papers' / filename

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                continue

            try:
                paper.download_pdf(filename=str(target))
                print(f"  ✓ Downloaded: {paper.title[:60]}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")


def download_public_threat_intel():
    """Download public threat intelligence reports"""
    print("\n📚 Downloading public threat intelligence...")

    # Create markdown with links to major public reports
    content = """# Public Threat Intelligence Sources

## APT Reports (Public)
- FireEye APT Reports: https://www.mandiant.com/resources/insights/apt-groups
- CrowdStrike Intelligence: https://www.crowdstrike.com/adversaries/
- Kaspersky APT Intelligence: https://securelist.com/category/apt-reports/

## Chinese APT Groups (Public Research)
- APT41 (Winnti): https://attack.mitre.org/groups/G0096/
- APT10 (MenuPass): https://attack.mitre.org/groups/G0045/
- APT1 (Comment Crew): https://attack.mitre.org/groups/G0006/

## Russian APT Groups (Public Research)
- APT28 (Fancy Bear): https://attack.mitre.org/groups/G0007/
- APT29 (Cozy Bear): https://attack.mitre.org/groups/G0016/
- Sandworm: https://attack.mitre.org/groups/G0034/

## MITRE ATT&CK Framework
- Enterprise: https://attack.mitre.org/
- Mobile: https://attack.mitre.org/matrices/mobile/
- ICS: https://attack.mitre.org/matrices/ics/

## OSINT / SIGINT Resources
- Bellingcat Toolkit: https://docs.google.com/spreadsheets/d/18rtqh8EG2q1xBo2cLNyhIDuK9jrPGwYr9DI2UncoqJQ/
- OSINT Framework: https://osintframework.com/
- Shodan: https://www.shodan.io/
- Censys: https://search.censys.io/

## Geopolitical Analysis (Public)
- CSIS Cyber Policy: https://www.csis.org/programs/strategic-technologies-program/significant-cyber-incidents
- Council on Foreign Relations: https://www.cfr.org/cyber-operations/
- Atlantic Council: https://www.atlanticcouncil.org/programs/scowcroft-center-for-strategy-and-security/cyber-statecraft-initiative/

## Malware Databases
- VirusTotal: https://www.virustotal.com/
- MalwareBazaar: https://bazaar.abuse.ch/
- theZoo: https://github.com/ytisf/theZoo (for research only)

## Declassified Intelligence
- CIA FOIA Reading Room: https://www.cia.gov/readingroom/
- NSA Declassified Documents: https://www.nsa.gov/news-features/declassified-documents/
- GCHQ: https://www.gchq.gov.uk/

**Note:** All sources are public domain. For actual intelligence work,
use official classified channels with proper clearances.
"""

    target = KNOWLEDGE_BASE / 'Threat_Intelligence' / 'PUBLIC_SOURCES.md'
    target.write_text(content)
    print(f"  ✓ Created public sources reference: {target}")


def clone_public_intel_repos():
    """Clone public intelligence/OSINT repositories"""
    print("\n📚 Downloading OSINT/Intelligence tools and knowledge...")

    repos = [
        ('https://github.com/jivoi/awesome-osint', 'SIGINT_OSINT'),
        ('https://github.com/hslatman/awesome-threat-intelligence', 'Threat_Intelligence'),
        ('https://github.com/rmusser01/Infosec_Reference', 'Security'),
        ('https://github.com/The-Art-of-Hacking/h4cker', 'Security'),
    ]

    for repo_url, category in repos:
        repo_name = repo_url.split('/')[-1]
        target_dir = KNOWLEDGE_BASE / category / repo_name

        if target_dir.exists():
            print(f"  ✓ Already exists: {repo_name}")
            continue

        print(f"  Downloading: {repo_name}...")
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, str(target_dir)],
                capture_output=True,
                check=True
            )
            print(f"  ✓ Downloaded: {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {repo_name}")


def download_geopolitics_resources():
    """Download geopolitical analysis resources"""
    print("\n📚 Downloading geopolitical resources...")

    content = """# Geopolitical Intelligence Resources (Public)

## China
### Cyber Capabilities
- Chinese APT landscape: https://attack.mitre.org/groups/?country=China
- MSS operations (public reporting)
- PLA cyber units (Unit 61398, Unit 78020)

### Strategic Analysis
- CSIS China Power: https://chinapower.csis.org/
- Council on Foreign Relations - China: https://www.cfr.org/china/

## Russia
### Cyber Capabilities
- GRU cyber operations (public reporting)
- FSB cyber activities
- Russian APT groups

### Strategic Analysis
- Atlantic Council - Russia: https://www.atlanticcouncil.org/programs/eurasia-center/
- Carnegie Moscow Center: https://carnegiemoscow.org/

## SIGINT/ELINT (Public Knowledge)
- Five Eyes alliance (public information)
- ECHELON system (declassified)
- NSA capabilities (Snowden leaks - public domain)
- GCHQ Tempora (public reporting)

## Intelligence Community Structure (Public)
- US IC organizational chart
- UK intelligence services
- Five Eyes partnership
- Information sharing agreements

**Disclaimer:** All information here is from public sources.
Classified intelligence requires proper clearances and need-to-know.
"""

    target = KNOWLEDGE_BASE / 'Geopolitics' / 'INTEL_OVERVIEW.md'
    target.write_text(content)
    print(f"  ✓ Created geopolitics overview: {target}")


def main():
    """Main acquisition workflow"""
    print("="*70)
    print("Knowledge Acquisition for LAT5150DRVMIL RAG System")
    print("="*70)
    print("\nTopics:")
    print("  - Security & Malware Analysis")
    print("  - Kernel Development")
    print("  - SIGINT/OSINT")
    print("  - Threat Intelligence")
    print("  - Geopolitics (China/Russia)")
    print("  - Hardware Security")
    print("  - AI/ML Security")
    print("\nAll sources are public domain and legally accessible.")
    print("="*70)

    setup_directories()
    download_security_awesome_lists()
    clone_public_intel_repos()
    download_public_threat_intel()
    download_geopolitics_resources()
    download_kernel_docs()

    # Optional: arXiv papers (requires package)
    try:
        download_arxiv_papers()
    except Exception as e:
        print(f"\n⚠️  Skipping arXiv download: {e}")

    print("\n" + "="*70)
    print("✓ Knowledge acquisition complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review downloaded content in 00-documentation/General_Knowledge/")
    print("  2. Re-index with: python3 rag_system/document_processor.py")
    print("  3. Update embeddings: python3 rag_system/transformer_upgrade.py")
    print("  4. Use code assistant: python3 rag_system/code_assistant.py -i")
    print()


if __name__ == '__main__':
    main()
