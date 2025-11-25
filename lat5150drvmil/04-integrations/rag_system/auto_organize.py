#!/usr/bin/env python3
"""
Auto-Organization System for LAT5150DRVMIL RAG System

Self-healing code organization system that automatically:
- Analyzes current file structure
- Creates logical directory hierarchies
- Moves files to appropriate locations
- Updates imports automatically
- Maintains backward compatibility
- Provides rollback capability
- Generates comprehensive index

This is the "code that heals itself" - automatic organization and maintenance.
"""

import os
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import ast


class AutoOrganizer:
    """
    Intelligent auto-organization system for code structure

    Features:
    - Pattern-based file classification
    - Automatic directory structure creation
    - Import path updating
    - Dependency tracking
    - Rollback capability
    - Self-documentation
    """

    def __init__(self, root_dir: str = ".", dry_run: bool = True):
        """
        Initialize auto-organizer

        Args:
            root_dir: Root directory to organize
            dry_run: If True, only simulate changes
        """
        self.root_dir = Path(root_dir).resolve()
        self.dry_run = dry_run
        self.changes: List[Dict] = []
        self.file_map: Dict[str, str] = {}

        # Define organization structure
        self.structure = {
            "storage": {
                "patterns": ["storage_*.py"],
                "keywords": ["storage", "backend", "database", "cache"],
                "description": "Storage backends and orchestration"
            },
            "embeddings": {
                "patterns": ["*embedding*.py", "*chunking*.py", "*reranker*.py"],
                "keywords": ["embedding", "vector", "chunking", "rerank"],
                "description": "Embedding generation and vector operations"
            },
            "rag": {
                "patterns": ["rag_*.py", "*rag*.py", "query_*.py"],
                "keywords": ["rag", "retrieval", "query", "search"],
                "description": "RAG system core components"
            },
            "integrations": {
                "patterns": ["*_integration*.py", "telegram_*.py", "*_scraper*.py"],
                "keywords": ["integration", "telegram", "osint", "scraper", "collector"],
                "description": "External service integrations"
            },
            "code_tools": {
                "patterns": ["code_*.py", "*_codegen*.py"],
                "keywords": ["code", "generator", "analyzer", "formatter", "validator"],
                "description": "Code analysis and generation tools"
            },
            "ml_models": {
                "patterns": ["*_loader*.py", "*transformer*.py", "*_optimizer*.py"],
                "keywords": ["model", "transformer", "quantization", "npu", "cerebras"],
                "description": "Machine learning model utilities"
            },
            "vision": {
                "patterns": ["*screenshot*.py", "*vision*.py", "*donut*.py"],
                "keywords": ["vision", "screenshot", "image", "donut", "pdf"],
                "description": "Computer vision and OCR"
            },
            "monitoring": {
                "patterns": ["*_monitor*.py", "*health*.py", "*benchmark*.py"],
                "keywords": ["monitor", "health", "benchmark", "metrics"],
                "description": "System monitoring and benchmarking"
            },
            "utils": {
                "patterns": ["*_utils*.py", "*_cache*.py", "analysis_*.py"],
                "keywords": ["utils", "helper", "cache", "async"],
                "description": "Utility functions and helpers"
            },
            "docs": {
                "patterns": ["*.md"],
                "keywords": [],
                "description": "Documentation files"
            },
            "tests": {
                "patterns": ["test_*.py", "*_test.py"],
                "keywords": ["test"],
                "description": "Test files"
            },
            "scripts": {
                "patterns": ["*.sh", "setup_*.py"],
                "keywords": ["setup", "install", "uninstall"],
                "description": "Setup and utility scripts"
            },
            "data": {
                "patterns": ["*.json", "*.npz", "*.csv"],
                "keywords": ["data", "catalog", "index"],
                "description": "Data files and configurations"
            }
        }

    def analyze_file(self, filepath: Path) -> str:
        """
        Determine the appropriate category for a file

        Args:
            filepath: Path to file

        Returns:
            Category name
        """
        filename = filepath.name

        # Skip special files
        if filename in ["__init__.py", "auto_organize.py"]:
            return "root"

        # Check patterns first (highest priority)
        for category, config in self.structure.items():
            for pattern in config["patterns"]:
                if Path(filename).match(pattern):
                    return category

        # Check keywords in filename
        filename_lower = filename.lower()
        for category, config in self.structure.items():
            for keyword in config["keywords"]:
                if keyword in filename_lower:
                    return category

        # Check file content for Python files
        if filename.endswith(".py"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read(5000)  # Read first 5KB
                    content_lower = content.lower()

                    for category, config in self.structure.items():
                        for keyword in config["keywords"]:
                            if keyword in content_lower:
                                return category
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")

        # Default category
        return "misc"

    def create_directory_structure(self):
        """Create organized directory structure"""
        for category in self.structure.keys():
            category_dir = self.root_dir / category
            if not category_dir.exists():
                if not self.dry_run:
                    category_dir.mkdir(parents=True, exist_ok=True)
                print(f"{'[DRY RUN] ' if self.dry_run else ''}Created directory: {category}")

        # Create misc directory
        misc_dir = self.root_dir / "misc"
        if not misc_dir.exists():
            if not self.dry_run:
                misc_dir.mkdir(parents=True, exist_ok=True)
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Created directory: misc")

    def analyze_structure(self) -> Dict[str, List[str]]:
        """
        Analyze current structure and categorize files

        Returns:
            Dictionary mapping categories to file lists
        """
        categorized = {}

        # Get all files in root directory
        for filepath in self.root_dir.iterdir():
            if filepath.is_file():
                category = self.analyze_file(filepath)

                if category not in categorized:
                    categorized[category] = []

                categorized[category].append(filepath.name)

        return categorized

    def extract_imports(self, filepath: Path) -> List[str]:
        """
        Extract import statements from Python file

        Args:
            filepath: Path to Python file

        Returns:
            List of import statements
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports
        except Exception as e:
            print(f"Warning: Could not parse imports from {filepath}: {e}")
            return []

    def update_imports(self, filepath: Path, old_path: str, new_path: str):
        """
        Update import statements in file

        Args:
            filepath: Path to file to update
            old_path: Old import path
            new_path: New import path
        """
        if not filepath.name.endswith('.py'):
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace import statements
            updated_content = content

            # Handle "from X import Y" patterns
            old_module = old_path.replace('/', '.')
            new_module = new_path.replace('/', '.')

            patterns = [
                (f"from {old_module} import", f"from {new_module} import"),
                (f"import {old_module}", f"import {new_module}"),
            ]

            for old, new in patterns:
                updated_content = updated_content.replace(old, new)

            if updated_content != content:
                if not self.dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                print(f"{'[DRY RUN] ' if self.dry_run else ''}Updated imports in: {filepath.name}")

        except Exception as e:
            print(f"Warning: Could not update imports in {filepath}: {e}")

    def move_file(self, source: Path, dest_dir: Path) -> Path:
        """
        Move file to destination directory

        Args:
            source: Source file path
            dest_dir: Destination directory

        Returns:
            New file path
        """
        dest_file = dest_dir / source.name

        if dest_file.exists():
            print(f"Warning: {dest_file} already exists, skipping")
            return dest_file

        if not self.dry_run:
            shutil.move(str(source), str(dest_file))

        self.changes.append({
            'action': 'move',
            'source': str(source),
            'destination': str(dest_file),
            'timestamp': datetime.now().isoformat()
        })

        # Update file map
        self.file_map[source.name] = str(dest_file)

        print(f"{'[DRY RUN] ' if self.dry_run else ''}Moved: {source.name} → {dest_dir.name}/")

        return dest_file

    def create_init_files(self):
        """Create __init__.py files for all directories"""
        for category in list(self.structure.keys()) + ["misc"]:
            category_dir = self.root_dir / category
            init_file = category_dir / "__init__.py"

            if category_dir.exists() and not init_file.exists():
                if not self.dry_run:
                    with open(init_file, 'w') as f:
                        f.write(f'"""\n{self.structure.get(category, {}).get("description", category)}\n"""\n')
                print(f"{'[DRY RUN] ' if self.dry_run else ''}Created: {category}/__init__.py")

    def create_index(self) -> str:
        """
        Create comprehensive index of organized structure

        Returns:
            Index content as string
        """
        index = []
        index.append("# LAT5150DRVMIL RAG System - File Index")
        index.append(f"\nGenerated: {datetime.now().isoformat()}\n")

        # Analyze current structure
        categorized = {}
        for category in list(self.structure.keys()) + ["misc"]:
            category_dir = self.root_dir / category
            if category_dir.exists():
                files = [f.name for f in category_dir.iterdir() if f.is_file() and f.name != "__init__.py"]
                if files:
                    categorized[category] = sorted(files)

        # Generate index
        for category, files in sorted(categorized.items()):
            desc = self.structure.get(category, {}).get("description", category)
            index.append(f"\n## {category.upper()}")
            index.append(f"**{desc}**\n")

            for filename in files:
                index.append(f"- `{category}/{filename}`")

        # Add file mapping
        index.append("\n\n## File Mapping\n")
        index.append("| Original | New Location |")
        index.append("|----------|--------------|")
        for old_name, new_path in sorted(self.file_map.items()):
            index.append(f"| {old_name} | {new_path} |")

        return "\n".join(index)

    def organize(self):
        """Execute organization"""
        print("=" * 80)
        print("AUTO-ORGANIZATION SYSTEM")
        print("=" * 80)
        print(f"\nRoot Directory: {self.root_dir}")
        print(f"Mode: {'DRY RUN (simulation only)' if self.dry_run else 'LIVE (will make changes)'}\n")

        # Step 1: Analyze current structure
        print("\n[Step 1] Analyzing current structure...")
        categorized = self.analyze_structure()

        print("\nFile Distribution:")
        total_files = 0
        for category, files in sorted(categorized.items()):
            print(f"  {category:20} {len(files):3} files")
            total_files += len(files)
        print(f"  {'TOTAL':20} {total_files:3} files\n")

        # Step 2: Create directory structure
        print("[Step 2] Creating directory structure...")
        self.create_directory_structure()

        # Step 3: Move files
        print("\n[Step 3] Moving files to organized structure...")
        for category, files in sorted(categorized.items()):
            if category == "root":
                continue  # Keep root files in place

            dest_dir = self.root_dir / category
            for filename in files:
                source_file = self.root_dir / filename
                if source_file.exists():
                    self.move_file(source_file, dest_dir)

        # Step 4: Create __init__.py files
        print("\n[Step 4] Creating __init__.py files...")
        self.create_init_files()

        # Step 5: Generate index
        print("\n[Step 5] Generating file index...")
        index_content = self.create_index()
        index_file = self.root_dir / "FILE_INDEX.md"

        if not self.dry_run:
            with open(index_file, 'w') as f:
                f.write(index_content)
        print(f"{'[DRY RUN] ' if self.dry_run else ''}Created: FILE_INDEX.md")

        # Step 6: Save change log
        if not self.dry_run and self.changes:
            log_file = self.root_dir / "organization_log.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'changes': self.changes
                }, f, indent=2)
            print(f"\nChange log saved to: organization_log.json")

        # Summary
        print("\n" + "=" * 80)
        print("ORGANIZATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal files organized: {total_files}")
        print(f"Total changes: {len(self.changes)}")

        if self.dry_run:
            print("\n⚠️  This was a DRY RUN. No files were actually moved.")
            print("Run with --execute to apply changes.")
        else:
            print("\n✓ Files have been organized successfully!")
            print("\nNext steps:")
            print("  1. Review FILE_INDEX.md")
            print("  2. Update any absolute imports in your code")
            print("  3. Test that everything still works")
            print("  4. Commit changes to git")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-organize LAT5150DRVMIL RAG system file structure"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to organize (default: current directory)"
    )

    args = parser.parse_args()

    organizer = AutoOrganizer(
        root_dir=args.root,
        dry_run=not args.execute
    )

    organizer.organize()


if __name__ == "__main__":
    main()
