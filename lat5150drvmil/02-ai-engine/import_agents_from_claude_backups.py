#!/usr/bin/env python3
"""
Import all agents from claude-backups repository
Fetches markdown agent definitions and converts them to local format

This script:
1. Fetches all agent .md files from claude-backups GitHub
2. Parses them using LocalAgentLoader
3. Creates a local agent database
4. Exports to JSON for fast loading
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_agent_loader import LocalAgentLoader, StandardizedAgent


class ClaudeBackupsImporter:
    """Import agents from claude-backups GitHub repository"""

    def __init__(self, github_user: str = "SWORDIntel", repo: str = "claude-backups"):
        """
        Initialize importer

        Args:
            github_user: GitHub username
            repo: Repository name
        """
        self.github_user = github_user
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{github_user}/{repo}"
        self.raw_base_url = f"https://raw.githubusercontent.com/{github_user}/{repo}/main"
        self.loader = LocalAgentLoader()

        print(f"✓ Importer initialized for {github_user}/{repo}")

    def fetch_agent_list(self) -> List[str]:
        """
        Fetch list of all agent markdown files

        Returns:
            List of agent filenames
        """
        print("Fetching agent list from GitHub...")

        url = f"{self.base_url}/contents/agents"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            files = response.json()

            # Filter for .md files (exclude README, etc.)
            agent_files = [
                f['name'] for f in files
                if f['name'].endswith('.md')
                and not f['name'].lower().startswith('readme')
                and f['type'] == 'file'
            ]

            print(f"✓ Found {len(agent_files)} agent files")
            return agent_files

        except Exception as e:
            print(f"❌ Error fetching agent list: {e}")
            return []

    def fetch_agent_content(self, filename: str) -> str:
        """
        Fetch agent markdown content from GitHub

        Args:
            filename: Agent filename (e.g., "PYTHON-INTERNAL.md")

        Returns:
            Markdown content
        """
        url = f"{self.raw_base_url}/agents/{filename}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text

        except Exception as e:
            print(f"  ⚠️  Error fetching {filename}: {e}")
            return ""

    def import_all_agents(self, limit: int = None) -> int:
        """
        Import all agents from claude-backups

        Args:
            limit: Optional limit on number of agents to import

        Returns:
            Number of agents imported
        """
        agent_files = self.fetch_agent_list()

        if limit:
            agent_files = agent_files[:limit]

        print(f"\nImporting {len(agent_files)} agents...")
        print("=" * 70)

        imported_count = 0
        failed_count = 0

        for i, filename in enumerate(agent_files, 1):
            try:
                print(f"[{i}/{len(agent_files)}] {filename}...", end=" ")

                # Fetch content
                content = self.fetch_agent_content(filename)

                if not content:
                    print("❌ Failed to fetch")
                    failed_count += 1
                    continue

                # Parse and load
                agent = self.loader.load_agent_definition(
                    agent_id=filename.replace(".md", "").lower().replace("-", "_"),
                    markdown_content=content
                )

                print(f"✓ {agent.name} ({agent.category.value}, {agent.preferred_hardware})")
                imported_count += 1

                # Rate limiting (be nice to GitHub)
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Error: {e}")
                failed_count += 1

        print("=" * 70)
        print(f"\n✓ Imported {imported_count} agents")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} agents")

        return imported_count

    def export_agent_database(self, output_file: str = "agent_database.json"):
        """
        Export loaded agents to JSON database

        Args:
            output_file: Output filename
        """
        output_path = Path(__file__).parent / output_file

        self.loader.export_to_json(str(output_path))

        # Also create a detailed export with full agent data
        detailed_output = output_path.with_stem(f"{output_path.stem}_detailed")

        agents_detailed = {}
        for agent_id, agent in self.loader.agents.items():
            agents_detailed[agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "category": agent.category.value,
                "role": agent.role,
                "capabilities": agent.capabilities,
                "execution_mode": agent.execution_mode.value,
                "preferred_hardware": agent.preferred_hardware,
                "requires_avx512": agent.requires_avx512,
                "requires_npu": agent.requires_npu,
                "requires_gpu": agent.requires_gpu,
                "local_model": agent.local_model,
                "model_params": agent.model_params,
                "required_tools": agent.required_tools,
                "optional_tools": agent.optional_tools,
                "can_delegate_to": agent.can_delegate_to,
                "avg_response_time_ms": agent.avg_response_time_ms,
                "parallel_capable": agent.parallel_capable,
                "original_version": agent.original_version,
                "source_file": agent.source_file,
            }

        with open(detailed_output, 'w') as f:
            json.dump(agents_detailed, f, indent=2)

        print(f"✓ Detailed database: {detailed_output}")

    def print_summary(self):
        """Print import summary"""
        stats = self.loader.get_stats()

        print("\n" + "=" * 70)
        print(" Agent Import Summary")
        print("=" * 70)
        print()

        print(f"Total Agents: {stats['total']}")
        print()

        print("By Category:")
        for category, count in sorted(stats['by_category'].items()):
            print(f"  {category:20s}: {count:3d} agents")
        print()

        print("By Hardware Preference:")
        for hardware, count in sorted(stats['by_hardware'].items()):
            print(f"  {hardware:20s}: {count:3d} agents")
        print()

        print("By Execution Mode:")
        for mode, count in sorted(stats['by_execution_mode'].items()):
            print(f"  {mode:20s}: {count:3d} agents")
        print()

        print(f"Requires NPU: {stats['requires_npu']} agents")
        print(f"Requires AVX512: {stats['requires_avx512']} agents")
        print()

        print("=" * 70)


def main():
    """Main import process"""
    print("=" * 70)
    print(" Claude-Backups Agent Importer")
    print("=" * 70)
    print()

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Import agents from claude-backups")
    parser.add_argument('--limit', type=int, help="Limit number of agents to import")
    parser.add_argument('--output', default="agent_database.json", help="Output database file")
    parser.add_argument('--user', default="SWORDIntel", help="GitHub user")
    parser.add_argument('--repo', default="claude-backups", help="GitHub repository")

    args = parser.parse_args()

    # Create importer
    importer = ClaudeBackupsImporter(github_user=args.user, repo=args.repo)

    # Import agents
    count = importer.import_all_agents(limit=args.limit)

    if count > 0:
        # Export database
        importer.export_agent_database(output_file=args.output)

        # Print summary
        importer.print_summary()

        print("\n✅ Import complete!")
        print(f"\nAgent database created: {args.output}")
        print("\nTo use the agents:")
        print("  from local_agent_loader import LocalAgentLoader")
        print("  loader = LocalAgentLoader()")
        print("  # Load from JSON database")
        print(f"  # Or load from directory: loader.load_from_directory('/path/to/agents')")
    else:
        print("\n❌ No agents imported")
        sys.exit(1)


if __name__ == "__main__":
    main()
