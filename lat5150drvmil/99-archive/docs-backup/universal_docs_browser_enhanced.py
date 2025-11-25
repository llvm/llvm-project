#!/usr/bin/env python3
"""
Universal Documentation Browser with AI-Enhanced Analysis
A modular PyGUI interface that adapts to any documentation structure with:
- Automatic dependency installation
- AI-powered document classification and summarization 
- Standardized overview generation with ML
- Cached extracted text versions for PDFs
- Markdown preview until analysis completes

Features:
- Auto-detection of documentation structure
- Intelligent document classification using basic ML
- Automatic PDF text extraction with caching
- Standardized folder-based overview generation
- Markdown preview with live analysis updates
- Automatic library installation for dependencies
- Single-file portable implementation

Usage: python3 universal_docs_browser_enhanced.py [directory]
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import webbrowser
import subprocess
from pathlib import Path
import re
import json
from typing import Dict, List, Optional, Tuple, Set
import threading
import queue
import configparser
import argparse
import tempfile
import shutil
import hashlib
import time

# Auto-installation system
def auto_install_package(package_name: str, import_name: str = None) -> bool:
    """Auto-install package if not available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return False

# Auto-install critical dependencies
PDFPLUMBER_AVAILABLE = auto_install_package('pdfplumber')
SKLEARN_AVAILABLE = auto_install_package('scikit-learn', 'sklearn')
MARKDOWN_AVAILABLE = auto_install_package('markdown')

# Try to import dependencies after installation
try:
    import pdfplumber
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import markdown
    MARKDOWN_PROCESSING_AVAILABLE = True
except ImportError:
    MARKDOWN_PROCESSING_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class PDFExtractor:
    """Integrated PDF text extraction using pdfplumber"""
    
    @staticmethod
    def extract_text(pdf_path: Path, progress_callback=None) -> str:
        """Extract text from PDF file"""
        if not PDF_EXTRACTION_AVAILABLE:
            return "PDF extraction not available. Install with: pip install pdfplumber"
        
        text_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages, 1):
                    if progress_callback:
                        progress_callback(f"Processing page {i}/{total_pages}...")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"\n--- Page {i} ---\n")
                        text_content.append(page_text)
                
                return "\n".join(text_content)
        
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    @staticmethod
    def get_pdf_info(pdf_path: Path) -> Dict[str, str]:
        """Get PDF metadata information"""
        if not PDF_EXTRACTION_AVAILABLE:
            return {"error": "PDF processing not available"}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                info = {
                    "pages": str(len(pdf.pages)),
                    "title": getattr(pdf.metadata, 'title', 'Unknown'),
                    "author": getattr(pdf.metadata, 'author', 'Unknown'),
                    "subject": getattr(pdf.metadata, 'subject', 'Unknown'),
                    "creator": getattr(pdf.metadata, 'creator', 'Unknown'),
                }
                return info
        except Exception as e:
            return {"error": f"Could not read PDF info: {e}"}
    
    @staticmethod
    def get_cached_text_path(pdf_path: Path) -> Path:
        """Get path for cached extracted text"""
        return pdf_path.with_suffix('.pdf.txt')
    
    @staticmethod
    def extract_with_cache(pdf_path: Path, progress_callback=None) -> str:
        """Extract PDF text with caching support"""
        cache_path = PDFExtractor.get_cached_text_path(pdf_path)
        
        # Check if cached version exists and is newer than PDF
        if cache_path.exists() and cache_path.stat().st_mtime > pdf_path.stat().st_mtime:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                pass  # Fall through to extraction
        
        # Extract text and cache it
        text_content = PDFExtractor.extract_text(pdf_path, progress_callback)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        except Exception as e:
            print(f"Warning: Could not cache extracted text: {e}")
        
        return text_content

class AIDocumentClassifier:
    """AI-powered document classification and overview generation"""
    
    def __init__(self):
        self.vectorizer = None
        self.patterns = {
            'agent_files': {
                'pattern': r'(AGENT|agent).*\.(md|MD)',
                'keywords': ['agent', 'implementation', 'coordination', 'task', 'proactive'],
                'template': '{lang} AGENT specialist with {capabilities}'
            },
            'infrastructure': {
                'pattern': r'(infrastructure|deployment|docker|database)',
                'keywords': ['infrastructure', 'deployment', 'system', 'configuration'],
                'template': 'Infrastructure implementation with {focus_areas}'
            },
            'documentation': {
                'pattern': r'(docs|documentation|guide|manual|readme)',
                'keywords': ['documentation', 'guide', 'instructions', 'manual'],
                'template': '{doc_type} documentation for {system_name}'
            },
            'protocol': {
                'pattern': r'(protocol|communication|binary|api)',
                'keywords': ['protocol', 'communication', 'binary', 'message', 'api'],
                'template': '{protocol_type} protocol documentation in {language}'
            },
            'security': {
                'pattern': r'(security|crypto|auth|protection)',
                'keywords': ['security', 'authentication', 'encryption', 'protection'],
                'template': 'Security documentation covering {security_aspects}'
            }
        }
    
    def classify_document(self, file_path: Path, content: str = None) -> Dict[str, str]:
        """Classify document and generate standardized overview"""
        if content is None:
            try:
                if file_path.suffix.lower() == '.pdf':
                    content = PDFExtractor.extract_with_cache(file_path)[:2000]  # First 2K chars
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]
            except:
                content = ""
        
        # Extract key information
        doc_info = {
            'filename': file_path.name,
            'category': self._classify_category(file_path, content),
            'language': self._detect_language(file_path, content),
            'capabilities': self._extract_capabilities(content),
            'overview': self._generate_overview(file_path, content)
        }
        
        return doc_info
    
    def _classify_category(self, file_path: Path, content: str) -> str:
        """Classify document category"""
        filename_lower = file_path.name.lower()
        content_lower = content.lower()
        
        # Check patterns
        for category, info in self.patterns.items():
            if re.search(info['pattern'], filename_lower, re.IGNORECASE):
                return category
            
            # Check content for keywords
            keyword_count = sum(1 for keyword in info['keywords'] if keyword in content_lower)
            if keyword_count >= 2:
                return category
        
        return 'general'
    
    def _detect_language(self, file_path: Path, content: str) -> str:
        """Detect programming language or document type"""
        filename = file_path.name.upper()
        content_lower = content.lower()
        
        # Language detection patterns
        languages = {
            'JULIA': ['julia', '.jl', 'using ', '@', 'function ', 'end'],
            'PYTHON': ['python', '.py', 'import ', 'def ', 'class '],
            'RUST': ['rust', '.rs', 'fn ', 'let ', 'cargo', 'crate'],
            'C/C++': ['c++', '.cpp', '.c', '.h', '#include', 'int main'],
            'JAVASCRIPT': ['javascript', '.js', '.ts', 'function', 'const ', 'let '],
            'GO': ['golang', '.go', 'package ', 'func ', 'import '],
            'JAVA': ['java', '.java', 'public class', 'import java'],
            'DART': ['dart', '.dart', 'void main', 'import \'dart'],
            'PHP': ['php', '.php', '<?php', 'function ', 'class '],
            'SQL': ['sql', '.sql', 'select ', 'create table', 'database'],
            'ASSEMBLY': ['assembly', '.asm', '.s', 'mov ', 'jmp ', 'call '],
            'MATLAB': ['matlab', '.m', 'function ', 'clear', 'clc'],
            'SCADA': ['scada', 'modbus', 'opc', 'hmi', 'plc'],
            'BINARY': ['binary', 'protocol', 'communication', 'message'],
            'MARKDOWN': ['.md', '# ', '## ', '```']
        }
        
        for lang, indicators in languages.items():
            score = 0
            for indicator in indicators:
                if indicator in filename or indicator in content_lower:
                    score += 1
            if score >= 2:
                return lang
        
        return 'GENERAL'
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract key capabilities from document content"""
        capabilities = []
        content_lower = content.lower()
        
        capability_patterns = {
            'high-performance': ['performance', 'optimization', 'speed', 'fast'],
            'security': ['security', 'authentication', 'encryption', 'protection'],
            'coordination': ['coordination', 'orchestration', 'collaboration'],
            'analysis': ['analysis', 'analytics', 'intelligence', 'insights'],
            'automation': ['automation', 'automated', 'auto-', 'scripted'],
            'integration': ['integration', 'interface', 'api', 'connector'],
            'monitoring': ['monitoring', 'observability', 'metrics', 'logging'],
            'deployment': ['deployment', 'production', 'container', 'docker'],
            'testing': ['testing', 'validation', 'verification', 'quality']
        }
        
        for capability, keywords in capability_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities[:3]  # Top 3 capabilities
    
    def _generate_overview(self, file_path: Path, content: str) -> str:
        """Generate standardized one-line overview"""
        doc_info = {
            'filename': file_path.name,
            'language': self._detect_language(file_path, content),
            'capabilities': self._extract_capabilities(content),
            'category': self._classify_category(file_path, content)
        }
        
        # Generate overview based on category and detected information
        if doc_info['category'] == 'agent_files':
            caps = ' AND '.join(doc_info['capabilities']).upper() if doc_info['capabilities'] else 'SPECIALIZED PROCESSING'
            return f"{doc_info['language']} AGENT specialist with {caps}"
        
        elif doc_info['category'] == 'infrastructure':
            focus = ' AND '.join(doc_info['capabilities']).upper() if doc_info['capabilities'] else 'SYSTEM MANAGEMENT'
            return f"Infrastructure implementation with {focus}"
        
        elif doc_info['category'] == 'protocol':
            return f"{doc_info['language']} protocol communication system documentation"
        
        elif doc_info['category'] == 'security':
            aspects = ' AND '.join(doc_info['capabilities']).upper() if doc_info['capabilities'] else 'SECURITY MEASURES'
            return f"Security documentation covering {aspects}"
        
        elif doc_info['category'] == 'documentation':
            system = doc_info['language'] if doc_info['language'] != 'GENERAL' else 'SYSTEM'
            return f"{system} documentation and implementation guide"
        
        else:
            # General case
            if doc_info['capabilities']:
                caps = ' AND '.join(doc_info['capabilities']).upper()
                return f"{doc_info['language']} implementation with {caps}"
            else:
                return f"{doc_info['language']} technical documentation"

class MarkdownProcessor:
    """Process and render markdown content"""
    
    @staticmethod
    def process_markdown(content: str) -> str:
        """Convert markdown to HTML if processor available"""
        if not MARKDOWN_PROCESSING_AVAILABLE:
            return content
        
        try:
            # Convert markdown to HTML
            html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
            return html
        except Exception:
            return content
    
    @staticmethod
    def extract_first_paragraph(content: str) -> str:
        """Extract first meaningful paragraph from markdown"""
        lines = content.split('\n')
        paragraph_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):  # Skip headers
                continue
            if line.startswith('---'):  # Skip frontmatter
                continue
            if line.startswith('```'):  # Skip code blocks
                break
            
            paragraph_lines.append(line)
            if len(' '.join(paragraph_lines)) > 200:  # Reasonable preview length
                break
        
        return ' '.join(paragraph_lines)

class DocumentationStructureAnalyzer:
    """Analyzes and adapts to different documentation structures"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.structure = {}
        self.categories = {}
        self.role_mappings = {}
        self.config_file = root_path / ".docs_browser_config.json"
        
    def analyze_structure(self) -> Dict:
        """Automatically analyze documentation structure"""
        structure = {
            'total_files': 0,
            'categories': {},
            'file_types': {},
            'depth': 0,
            'patterns': []
        }
        
        # Load existing config if available
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    # Check if config is recent (within 1 hour)
                    if saved_config.get('structure') and \
                       abs(self.root_path.stat().st_mtime - saved_config.get('analyzed_at', 0)) < 3600:
                        return saved_config['structure']
            except Exception:
                pass
        
        # Directories to ignore during analysis
        ignore_dirs = {
            'venv', 'env', '.venv', '.env',  # Virtual environments
            'node_modules', '.git', '.svn',  # Version control and dependencies
            '__pycache__', '.pytest_cache',  # Python cache
            '.idea', '.vscode', '.vs',       # IDE directories
            'build', 'dist', '.build',       # Build directories
            '.tox', '.coverage',             # Testing/coverage
            'logs', 'log', 'tmp', 'temp',    # Temporary directories
            '.cache', 'cache'                # Cache directories
        }
        
        # Scan directory structure
        for item in self.root_path.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                item.name.lower() not in ignore_dirs):
                category_info = self._analyze_category(item)
                if category_info['file_count'] > 0:
                    structure['categories'][item.name] = category_info
                    structure['total_files'] += category_info['file_count']
        
        # Also check for files in root
        root_files = self._get_doc_files(self.root_path, recursive=False)
        if root_files:
            structure['categories']['_root'] = {
                'path': str(self.root_path),
                'description': 'Root documentation files',
                'file_count': len(root_files),
                'files': [f.name for f in root_files],
                'patterns': self._detect_patterns([f.name for f in root_files])
            }
            structure['total_files'] += len(root_files)
        
        # Detect overall patterns and depth
        structure['patterns'] = self._detect_global_patterns(structure['categories'])
        structure['depth'] = self._calculate_depth()
        structure['file_types'] = self._analyze_file_types()
        
        self.structure = structure
        return structure
    
    def _analyze_category(self, category_path: Path) -> Dict:
        """Analyze a specific category directory"""
        files = self._get_doc_files(category_path)
        
        return {
            'path': str(category_path),
            'description': self._generate_description(category_path.name, files),
            'file_count': len(files),
            'files': [f.name for f in files],
            'patterns': self._detect_patterns([f.name for f in files]),
            'subdirs': [d.name for d in category_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        }
    
    def _get_doc_files(self, path: Path, recursive: bool = True) -> List[Path]:
        """Get all documentation files in a directory, excluding ignored directories"""
        extensions = {'.md', '.txt', '.rst', '.pdf', '.html', '.htm', '.adoc', '.tex', '.docx', '.doc', '.odt'}
        files = []
        
        # Directories to ignore
        ignore_dirs = {
            'venv', 'env', '.venv', '.env',  # Virtual environments
            'node_modules', '.git', '.svn',  # Version control and dependencies
            '__pycache__', '.pytest_cache',  # Python cache
            '.idea', '.vscode', '.vs',       # IDE directories
            'build', 'dist', '.build',       # Build directories
            '.tox', '.coverage',             # Testing/coverage
            'logs', 'log', 'tmp', 'temp',    # Temporary directories
            '.cache', 'cache'                # Cache directories
        }
        
        if recursive:
            for ext in extensions:
                for file_path in path.rglob(f'*{ext}'):
                    # Check if any parent directory should be ignored
                    should_ignore = False
                    for parent in file_path.parents:
                        if parent.name.lower() in ignore_dirs or parent.name.startswith('.'):
                            should_ignore = True
                            break
                    
                    if not should_ignore and not file_path.name.startswith('.'):
                        files.append(file_path)
        else:
            for ext in extensions:
                files.extend(path.glob(f'*{ext}'))
        
        return [f for f in files if not f.name.startswith('.')]
    
    def _detect_patterns(self, filenames: List[str]) -> List[str]:
        """Detect naming patterns in filenames"""
        patterns = []
        
        # Common patterns
        common_patterns = [
            (r'^README', 'readme'),
            (r'^INSTALL', 'installation'),
            (r'^SETUP', 'setup'),
            (r'^CONFIG', 'configuration'),
            (r'^GUIDE', 'guide'),
            (r'^TUTORIAL', 'tutorial'),
            (r'^API', 'api'),
            (r'^REFERENCE', 'reference'),
            (r'^ARCH', 'architecture'),
            (r'^DESIGN', 'design'),
            (r'^IMPL', 'implementation'),
            (r'^TEST', 'testing'),
            (r'^FIX', 'fixes'),
            (r'^BUG', 'bugfix'),
            (r'^FEATURE', 'features'),
            (r'^CHANGE', 'changelog'),
            (r'^RELEASE', 'releases'),
            (r'^SECURITY', 'security'),
            (r'^DEPLOY', 'deployment'),
            (r'^TROUBLE', 'troubleshooting'),
            (r'^FAQ', 'faq'),
            (r'^LEGACY', 'legacy'),
            (r'^DEPRECAT', 'deprecated'),
            (r'\d{4}-\d{2}-\d{2}', 'dated'),
            (r'v\d+\.\d+', 'versioned'),
            (r'enhanced-plans', 'strategic'),
            (r'roadmaps', 'planning'),
            (r'user-guide', 'user_guides'),
            (r'technical', 'technical')
        ]
        
        for pattern, name in common_patterns:
            if any(re.search(pattern, f, re.IGNORECASE) for f in filenames):
                patterns.append(name)
        
        return patterns
    
    def _detect_global_patterns(self, categories: Dict) -> List[str]:
        """Detect global documentation patterns"""
        all_patterns = []
        for cat in categories.values():
            all_patterns.extend(cat.get('patterns', []))
        
        # Return unique patterns
        return list(set(all_patterns))
    
    def _generate_description(self, category_name: str, files: List[Path]) -> str:
        """Generate description for a category based on its name and contents"""
        name_lower = category_name.lower()
        
        # Predefined descriptions for common categories
        descriptions = {
            'api': 'API documentation and reference',
            'architecture': 'System architecture and design documents',
            'deployment': 'Deployment guides and procedures',
            'docs': 'General documentation',
            'enhanced-plans': 'Strategic planning and enhancement documents',
            'guides': 'User guides and tutorials',
            'legacy': 'Legacy and deprecated documentation',
            'roadmaps': 'Project roadmaps and planning documents',
            'security': 'Security documentation and policies',
            'technical': 'Technical specifications and references',
            'troubleshooting': 'Problem solving and debugging guides',
            'user-guide': 'User guides and how-to documents',
            'implementation': 'Implementation details and reports',
            'performance': 'Performance optimization documentation',
            'strategic': 'Strategic planning documents'
        }
        
        # Try to match category name
        for key, desc in descriptions.items():
            if key in name_lower:
                return f"{desc} ({len(files)} files)"
        
        # Fallback: generate based on content patterns
        patterns = self._detect_patterns([f.name for f in files])
        if patterns:
            return f"{patterns[0].title()} documentation ({len(files)} files)"
        
        return f"{category_name.title()} ({len(files)} files)"
    
    def _calculate_depth(self) -> int:
        """Calculate maximum directory depth"""
        max_depth = 0
        try:
            for item in self.root_path.rglob('*'):
                if item.is_file():
                    depth = len(item.relative_to(self.root_path).parts)
                    max_depth = max(max_depth, depth)
        except Exception:
            pass
        return max_depth
    
    def _analyze_file_types(self) -> Dict[str, int]:
        """Analyze file type distribution"""
        file_types = {}
        try:
            for file_path in self._get_doc_files(self.root_path):
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        except Exception:
            pass
        return file_types
    
    def generate_role_mappings(self) -> Dict[str, List[str]]:
        """Generate role-based document mappings using enhanced semantic analysis
        
        Enhanced Semantic Matching with 8 Role Categories:
        - New Users: 'readme', 'getting', 'start', 'intro', 'quick', 'begin', 'guide', 'first', 'launch', 'walkthrough'
        - Developers: 'api', 'arch', 'design', 'dev', 'code', 'impl', 'technical', 'spec', 'framework', 'orchestration'
        - Administrators: 'install', 'setup', 'config', 'deploy', 'admin', 'trouble', 'ops', 'management', 'monitoring'
        - Security Experts: 'security', 'auth', 'crypto', 'secure', 'vuln', 'audit', 'cert', 'encryption', 'bastion'
        - System Builders: 'build', 'kernel', 'driver', 'hardware', 'firmware', 'boot', 'compilation', 'microcode'
        - Network Engineers: 'network', 'mesh', 'vpn', 'tunnel', 'routing', 'protocol', 'gateway', 'dns', 'tls'
        - Project Managers: 'roadmap', 'plan', 'strategy', 'timeline', 'milestone', 'project', 'executive', 'summary'
        - QA/Testing: 'test', 'testing', 'qa', 'quality', 'validation', 'verification', 'benchmark', 'performance'
        
        Uses scoring system: filename + category matches, content analysis, multi-role support.
        """
        mappings = {
            'New Users': [],
            'Developers': [],
            'Administrators': [],
            'Security Experts': [],
            'System Builders': [],
            'Network Engineers': [],
            'Project Managers': [],
            'QA/Testing': []
        }
        
        # Enhanced semantic matching with scoring system
        for cat_name, cat_info in self.structure.get('categories', {}).items():
            cat_path = Path(cat_info['path'])
            files = cat_info.get('files', [])
            
            for file_name in files[:5]:  # Increased from 3 to 5 files per category
                try:
                    relative_path = str(Path(cat_path.name) / file_name) if cat_name != '_root' else file_name
                    name_lower = file_name.lower()
                    cat_lower = cat_name.lower()
                    
                    # Score-based matching for better accuracy
                    role_scores = {}
                    
                    # New Users (Getting Started)
                    score = 0
                    if any(pattern in name_lower for pattern in ['readme', 'getting', 'start', 'intro', 'quick', 'begin', 'guide', 'first', 'launch', 'walkthrough', 'primer', 'basics']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['guide', 'tutorial', 'intro', 'user-guide', 'getting-started', 'quickstart', 'onboarding']):
                        score += 1
                    if score > 0:
                        role_scores['New Users'] = score
                    
                    # Developers (Technical Implementation)
                    score = 0
                    if any(pattern in name_lower for pattern in ['api', 'arch', 'design', 'dev', 'code', 'impl', 'technical', 'spec', 'framework', 'orchestration', 'engine', 'core']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['api', 'architecture', 'technical', 'implementation', 'development', 'specs', 'reference', 'framework']):
                        score += 1
                    if score > 0:
                        role_scores['Developers'] = score
                    
                    # Administrators (Operations)
                    score = 0
                    if any(pattern in name_lower for pattern in ['install', 'setup', 'config', 'deploy', 'admin', 'trouble', 'ops', 'management', 'monitoring', 'maintenance']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['deployment', 'troubleshooting', 'installation', 'operations', 'maintenance', 'ops', 'infrastructure']):
                        score += 1
                    if score > 0:
                        role_scores['Administrators'] = score
                    
                    # Security Experts
                    score = 0
                    if any(pattern in name_lower for pattern in ['security', 'auth', 'crypto', 'secure', 'vuln', 'audit', 'cert', 'encryption', 'bastion', 'hardening']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['security', 'auth', 'crypto', 'compliance', 'audit', 'hardening']):
                        score += 1
                    if score > 0:
                        role_scores['Security Experts'] = score
                    
                    # System Builders (NEW)
                    score = 0
                    if any(pattern in name_lower for pattern in ['build', 'kernel', 'driver', 'hardware', 'firmware', 'boot', 'compilation', 'microcode', 'toolchain', 'bios']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['build', 'hardware', 'kernel', 'drivers', 'firmware', 'compilation', 'toolchain']):
                        score += 1
                    if score > 0:
                        role_scores['System Builders'] = score
                    
                    # Network Engineers (NEW)
                    score = 0
                    if any(pattern in name_lower for pattern in ['network', 'mesh', 'vpn', 'tunnel', 'routing', 'protocol', 'gateway', 'dns', 'tls', 'ssl', 'proxy']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['network', 'infrastructure', 'routing', 'protocol', 'mesh', 'vpn', 'tunnel']):
                        score += 1
                    if score > 0:
                        role_scores['Network Engineers'] = score
                    
                    # Project Managers (NEW)
                    score = 0
                    if any(pattern in name_lower for pattern in ['roadmap', 'plan', 'strategy', 'timeline', 'milestone', 'project', 'executive', 'summary', 'overview', 'coordination']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['roadmaps', 'plans', 'strategy', 'management', 'coordination', 'summaries', 'reports']):
                        score += 1
                    if score > 0:
                        role_scores['Project Managers'] = score
                    
                    # QA/Testing (NEW)
                    score = 0
                    if any(pattern in name_lower for pattern in ['test', 'testing', 'qa', 'quality', 'validation', 'verification', 'benchmark', 'performance', 'load', 'stress']):
                        score += 2
                    if any(pattern in cat_lower for pattern in ['testing', 'qa', 'validation', 'verification', 'benchmarks', 'performance', 'quality']):
                        score += 1
                    if score > 0:
                        role_scores['QA/Testing'] = score
                    
                    # Assign to roles based on scores (multi-role support)
                    for role, score in role_scores.items():
                        if score >= 1:  # Minimum score threshold
                            mappings[role].append(relative_path)
                
                except Exception:
                    continue
        
        # Remove empty roles and limit items with improved deduplication
        filtered_mappings = {}
        for role, docs in mappings.items():
            if docs:
                # Remove duplicates and limit to most relevant documents (max 10 per role for 8 categories)
                unique_docs = list(dict.fromkeys(docs))  # Remove duplicates while preserving order
                filtered_mappings[role] = unique_docs[:10]
        
        self.role_mappings = filtered_mappings
        return filtered_mappings
    
    def save_config(self):
        """Save configuration for future use"""
        config = {
            'version': '1.0',
            'root_path': str(self.root_path),
            'structure': self.structure,
            'role_mappings': self.role_mappings,
            'generated_at': str(Path.cwd()),
            'analyzed_at': self.root_path.stat().st_mtime
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

class UniversalDocumentationBrowser:
    """Universal documentation browser with PDF extraction"""
    
    def __init__(self, root: tk.Tk, docs_path: Path = None):
        self.root = root
        self.docs_path = docs_path or Path.cwd()
        self.current_file = None
        self.search_results = []
        self.extracted_pdfs = {}  # Cache for extracted PDF text
        
        # Initialize structure analyzer
        self.analyzer = DocumentationStructureAnalyzer(self.docs_path)
        self.structure = self.analyzer.analyze_structure()
        self.categories = self.structure.get('categories', {})
        self.role_mappings = self.analyzer.generate_role_mappings()
        
        # Save config for future use
        self.analyzer.save_config()
        
        self.setup_ui()
        self.load_documentation_tree()
        
    def setup_ui(self):
        """Setup the adaptive user interface"""
        project_name = self.docs_path.name.replace('-', ' ').replace('_', ' ').title()
        self.root.title(f"Universal Documentation Browser - {project_name}")
        self.root.geometry("1500x1000")
        self.root.minsize(1200, 700)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top toolbar
        self.create_toolbar(main_frame)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Left panel - Navigation
        self.create_navigation_panel(content_frame)
        
        # Right panel - Content viewer
        self.create_content_panel(content_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
    def create_toolbar(self, parent):
        """Create adaptive toolbar"""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        toolbar.columnconfigure(2, weight=1)
        
        # PDF status indicator
        pdf_frame = ttk.LabelFrame(toolbar, text="PDF Support", padding="3")
        pdf_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 5))
        
        pdf_status = "✓ Available" if PDF_EXTRACTION_AVAILABLE else "✗ Install pdfplumber"
        pdf_color = "green" if PDF_EXTRACTION_AVAILABLE else "red"
        ttk.Label(pdf_frame, text=pdf_status, foreground=pdf_color, font=('Arial', 8)).pack()
        
        # Role-based quick access (only if roles were detected)
        if self.role_mappings:
            role_frame = ttk.LabelFrame(toolbar, text="Quick Access by Role", padding="5")
            role_frame.grid(row=0, column=1, sticky=(tk.W, tk.N), padx=(0, 10))
            
            for i, role in enumerate(self.role_mappings.keys()):
                btn = ttk.Button(role_frame, text=role, 
                               command=lambda r=role: self.show_role_documents(r))
                btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky=(tk.W, tk.E))
        else:
            # Fallback: directory browser
            dir_frame = ttk.LabelFrame(toolbar, text="Directory Navigation", padding="5")
            dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.N), padx=(0, 10))
            
            ttk.Button(dir_frame, text="Change Directory", 
                      command=self.change_directory).pack(pady=2)
            ttk.Button(dir_frame, text="Refresh", 
                      command=self.refresh_structure).pack(pady=2)
        
        # Search frame
        search_frame = ttk.LabelFrame(toolbar, text="Search Documentation", padding="5")
        search_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        search_frame.columnconfigure(0, weight=1)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_change)
        
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Arial', 10))
        search_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        search_btn = ttk.Button(search_frame, text="Search", command=self.perform_search)
        search_btn.grid(row=0, column=1)
        
        # Bind Enter key to search
        search_entry.bind('<Return>', lambda e: self.perform_search())
    
    def create_navigation_panel(self, parent):
        """Create adaptive navigation panel"""
        nav_frame = ttk.LabelFrame(parent, text=f"Documentation Structure ({self.docs_path.name})", padding="5")
        nav_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        nav_frame.rowconfigure(0, weight=1)
        nav_frame.columnconfigure(0, weight=1)
        
        # Tree view with scrollbar
        tree_frame = ttk.Frame(nav_frame)
        tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(tree_frame, selectmode='browse')
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for tree
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=v_scroll.set)
        
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.tree.configure(xscrollcommand=h_scroll.set)
        
        # Configure tree columns
        self.tree.configure(columns=('info',), show='tree headings')
        self.tree.heading('#0', text='Document/Folder')
        self.tree.heading('info', text='Information')
        self.tree.column('#0', width=350, minwidth=250)
        self.tree.column('info', width=300, minwidth=200)
        
        # Bind tree selection
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        
        # Statistics and info
        stats_frame = ttk.Frame(nav_frame)
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="Analyzing structure...", 
                                   font=('Arial', 9), foreground='blue')
        self.stats_label.pack(anchor=tk.W)
        
        # Path info
        self.path_label = ttk.Label(stats_frame, text=f"Path: {self.docs_path}", 
                                   font=('Arial', 8), foreground='gray')
        self.path_label.pack(anchor=tk.W)
    
    def create_content_panel(self, parent):
        """Create content viewing panel with PDF extraction support"""
        content_frame = ttk.LabelFrame(parent, text="Document Viewer", padding="5")
        content_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)
        
        # Content viewer with scrollbar
        viewer_frame = ttk.Frame(content_frame)
        viewer_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        viewer_frame.rowconfigure(0, weight=1)
        viewer_frame.columnconfigure(0, weight=1)
        
        self.content_text = tk.Text(viewer_frame, wrap=tk.WORD, font=('Courier', 10),
                                  bg='white', fg='black', selectbackground='lightblue')
        self.content_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for content
        content_v_scroll = ttk.Scrollbar(viewer_frame, orient=tk.VERTICAL, 
                                       command=self.content_text.yview)
        content_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.content_text.configure(yscrollcommand=content_v_scroll.set)
        
        content_h_scroll = ttk.Scrollbar(viewer_frame, orient=tk.HORIZONTAL, 
                                       command=self.content_text.xview)
        content_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.content_text.configure(xscrollcommand=content_h_scroll.set)
        
        # Content toolbar
        content_toolbar = ttk.Frame(content_frame)
        content_toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(content_toolbar, text="Open External", 
                  command=self.open_external).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(content_toolbar, text="Copy Path", 
                  command=self.copy_file_path).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(content_toolbar, text="Show in Folder", 
                  command=self.show_in_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        # PDF specific buttons
        self.pdf_extract_btn = ttk.Button(content_toolbar, text="Extract PDF Text", 
                                         command=self.extract_pdf_text, state='disabled')
        self.pdf_extract_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pdf_info_btn = ttk.Button(content_toolbar, text="PDF Info", 
                                      command=self.show_pdf_info, state='disabled')
        self.pdf_info_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # File info label
        self.file_info_label = ttk.Label(content_toolbar, text="No file selected", 
                                       font=('Arial', 9), foreground='gray')
        self.file_info_label.pack(side=tk.RIGHT)
    
    def create_status_bar(self, parent):
        """Create adaptive status bar"""
        self.status_var = tk.StringVar()
        total_files = self.structure.get('total_files', 0)
        categories = len(self.categories)
        pdf_count = self.structure.get('file_types', {}).get('.pdf', 0)
        status_text = f"Ready - {total_files} files in {categories} categories"
        if pdf_count > 0:
            status_text += f" ({pdf_count} PDFs)"
        status_text += f" - {self.docs_path}"
        self.status_var.set(status_text)
        
        status_bar = ttk.Label(parent, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def load_documentation_tree(self):
        """Load the documentation structure into tree view"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            total_files = 0
            
            # Load each category
            for cat_name, cat_info in self.categories.items():
                if cat_name == '_root':
                    # Handle root files
                    for filename in cat_info.get('files', []):
                        file_path = self.docs_path / filename
                        file_info = self._get_file_info(file_path)
                        self.tree.insert('', 'end',
                                       text=filename,
                                       values=(file_info,),
                                       tags=('file',))
                        total_files += 1
                else:
                    cat_path = Path(cat_info['path'])
                    file_count = cat_info['file_count']
                    total_files += file_count
                    
                    # Insert category node
                    category_id = self.tree.insert('', 'end',
                                                 text=f"{cat_name}/ ({file_count} files)",
                                                 values=(cat_info['description'],),
                                                 tags=('category',))
                    
                    # Add files in this category
                    files = self._get_category_files(cat_path)
                    for file_path in sorted(files):
                        try:
                            file_info = self._get_file_info(file_path)
                            self.tree.insert(category_id, 'end',
                                           text=file_path.name,
                                           values=(file_info,),
                                           tags=('file',))
                        except Exception:
                            # Handle files that might not exist or have permission issues
                            self.tree.insert(category_id, 'end',
                                           text=file_path.name,
                                           values=("File info unavailable",),
                                           tags=('file',))
            
            # Configure tags
            self.tree.tag_configure('category', background='lightblue', font=('Arial', 10, 'bold'))
            self.tree.tag_configure('file', background='white')
            
            # Start with categories collapsed for cleaner initial view
            for item in self.tree.get_children():
                self.tree.item(item, open=False)
            
            # Update statistics
            cat_count = len([c for c in self.categories.keys() if c != '_root'])
            patterns = ", ".join(self.structure.get('patterns', []))[:60]
            if len(patterns) > 60:
                patterns += "..."
            
            stats_text = f"Total: {total_files} files, {cat_count} categories"
            if patterns:
                stats_text += f"\nPatterns: {patterns}"
            
            # Add file type breakdown
            file_types = self.structure.get('file_types', {})
            if file_types:
                type_summary = ", ".join([f"{ext}({count})" for ext, count in sorted(file_types.items()) if count > 0])
                stats_text += f"\nTypes: {type_summary}"
            
            self.stats_label.config(text=stats_text)
            self.status_var.set(f"Loaded {total_files} files from {self.docs_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load documentation tree: {e}")
            self.status_var.set(f"Error loading documentation: {e}")
    
    def _get_file_info(self, file_path: Path) -> str:
        """Get file information string"""
        try:
            size = file_path.stat().st_size
            size_str = self.format_file_size(size)
            ext = file_path.suffix.lower()
            
            if ext == '.pdf' and PDF_EXTRACTION_AVAILABLE:
                pdf_info = PDFExtractor.get_pdf_info(file_path)
                if 'pages' in pdf_info:
                    return f"PDF - {pdf_info['pages']} pages, {size_str}"
                else:
                    return f"PDF - {size_str}"
            else:
                return f"{ext.upper()[1:] if ext else 'File'} - {size_str}"
        except Exception:
            return "File info unavailable"
    
    def _get_category_files(self, category_path: Path) -> List[Path]:
        """Get all documentation files in a category"""
        extensions = {'.md', '.txt', '.rst', '.pdf', '.html', '.htm', '.adoc', '.tex', '.docx', '.doc', '.odt'}
        files = []
        
        try:
            for ext in extensions:
                files.extend(category_path.rglob(f'*{ext}'))
        except Exception:
            pass
        
        return [f for f in files if not f.name.startswith('.')]
    
    def format_file_size(self, size: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def on_tree_select(self, event):
        """Handle tree selection"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_text = self.tree.item(item, 'text')
        
        # Check if it's a file (not a category)
        if not item_text.endswith(')'):  # Categories end with file count in parentheses
            parent = self.tree.parent(item)
            if parent:
                # File in category
                category = self.tree.item(parent, 'text').split('/')[0]
                category_info = self.categories.get(category, {})
                if category_info:
                    category_path = Path(category_info['path'])
                    file_path = self._find_file_in_category(category_path, item_text)
                    if file_path:
                        self.load_file_content(file_path)
            else:
                # Root file
                file_path = self.docs_path / item_text
                if file_path.exists():
                    self.load_file_content(file_path)
    
    def _find_file_in_category(self, category_path: Path, filename: str) -> Optional[Path]:
        """Find a file within a category directory"""
        try:
            # First try direct path
            direct_path = category_path / filename
            if direct_path.exists():
                return direct_path
            
            # Then search recursively
            for file_path in self._get_category_files(category_path):
                if file_path.name == filename:
                    return file_path
        except Exception:
            pass
        return None
    
    def on_tree_double_click(self, event):
        """Handle double-click on tree items"""
        self.open_external()
    
    def load_file_content(self, file_path: Path):
        """Load and display file content with PDF extraction support"""
        try:
            self.current_file = file_path
            
            # Enable/disable PDF buttons
            is_pdf = file_path.suffix.lower() == '.pdf'
            pdf_state = 'normal' if (is_pdf and PDF_EXTRACTION_AVAILABLE) else 'disabled'
            self.pdf_extract_btn.config(state=pdf_state)
            self.pdf_info_btn.config(state=pdf_state)
            
            if file_path.suffix.lower() == '.pdf':
                # Handle PDF files
                self.content_text.config(state=tk.NORMAL)
                self.content_text.delete(1.0, tk.END)
                
                # Check if we have cached extracted text
                if str(file_path) in self.extracted_pdfs:
                    self.content_text.insert(1.0, self.extracted_pdfs[str(file_path)])
                else:
                    # Show PDF info and extraction option
                    self.content_text.insert(1.0, f"PDF Document: {file_path.name}\n\n")
                    
                    if PDF_EXTRACTION_AVAILABLE:
                        pdf_info = PDFExtractor.get_pdf_info(file_path)
                        self.content_text.insert(tk.END, "PDF Information:\n")
                        for key, value in pdf_info.items():
                            if key != 'error':
                                self.content_text.insert(tk.END, f"  {key.title()}: {value}\n")
                        
                        self.content_text.insert(tk.END, "\nClick 'Extract PDF Text' to extract and display text content.\n")
                        self.content_text.insert(tk.END, "Click 'Open External' to view with system PDF viewer.\n\n")
                    else:
                        self.content_text.insert(tk.END, "PDF text extraction not available.\n")
                        self.content_text.insert(tk.END, "Install with: pip install pdfplumber\n\n")
                        self.content_text.insert(tk.END, "Click 'Open External' to view with system PDF viewer.\n\n")
                    
                    self.content_text.insert(tk.END, f"File path: {file_path}\n")
                    self.content_text.insert(tk.END, f"File size: {self.format_file_size(file_path.stat().st_size)}\n")
                
                self.content_text.config(state=tk.DISABLED)
                
            elif file_path.suffix.lower() in ['.docx', '.doc', '.odt']:
                # Handle other binary document formats
                self.content_text.config(state=tk.NORMAL)
                self.content_text.delete(1.0, tk.END)
                self.content_text.insert(1.0, f"Binary Document: {file_path.name}\n\n")
                self.content_text.insert(tk.END, f"Type: {file_path.suffix.upper()} file\n")
                self.content_text.insert(tk.END, "Click 'Open External' to view this file.\n\n")
                self.content_text.insert(tk.END, f"File path: {file_path}\n")
                self.content_text.insert(tk.END, f"File size: {self.format_file_size(file_path.stat().st_size)}\n")
                self.content_text.config(state=tk.DISABLED)
                
            else:
                # Handle text files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except Exception:
                        content = "Error: Could not read file with available encodings."
                
                self.content_text.config(state=tk.NORMAL)
                self.content_text.delete(1.0, tk.END)
                self.content_text.insert(1.0, content)
                self.content_text.config(state=tk.DISABLED)
            
            # Update file info
            file_size = self.format_file_size(file_path.stat().st_size)
            self.file_info_label.config(text=f"{file_path.name} ({file_size})")
            self.status_var.set(f"Loaded: {file_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.status_var.set(f"Error loading file: {e}")
    
    def extract_pdf_text(self):
        """Extract text from current PDF file"""
        if not self.current_file or self.current_file.suffix.lower() != '.pdf':
            messagebox.showwarning("PDF Extraction", "No PDF file selected")
            return
        
        if not PDF_EXTRACTION_AVAILABLE:
            messagebox.showerror("PDF Extraction", "PDF extraction not available. Install pdfplumber:\npip install pdfplumber")
            return
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Extracting PDF Text")
        progress_window.geometry("400x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="Extracting PDF text...", font=('Arial', 10))
        progress_label.pack(pady=20)
        
        # Extract text in separate thread
        def extraction_thread():
            try:
                def update_progress(message):
                    progress_label.config(text=message)
                    progress_window.update()
                
                extracted_text = PDFExtractor.extract_text(self.current_file, update_progress)
                
                # Cache the extracted text
                self.extracted_pdfs[str(self.current_file)] = extracted_text
                
                # Update content view
                self.content_text.config(state=tk.NORMAL)
                self.content_text.delete(1.0, tk.END)
                self.content_text.insert(1.0, extracted_text)
                self.content_text.config(state=tk.DISABLED)
                
                progress_window.destroy()
                
                # Show extraction summary
                lines = len(extracted_text.splitlines())
                chars = len(extracted_text)
                messagebox.showinfo("PDF Extraction Complete", 
                                  f"Successfully extracted text from {self.current_file.name}\n\n"
                                  f"Characters: {chars:,}\n"
                                  f"Lines: {lines:,}")
                
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("PDF Extraction Error", f"Failed to extract text: {e}")
        
        # Start extraction in background
        threading.Thread(target=extraction_thread, daemon=True).start()
    
    def show_pdf_info(self):
        """Show detailed PDF information"""
        if not self.current_file or self.current_file.suffix.lower() != '.pdf':
            messagebox.showwarning("PDF Info", "No PDF file selected")
            return
        
        if not PDF_EXTRACTION_AVAILABLE:
            messagebox.showerror("PDF Info", "PDF processing not available. Install pdfplumber:\npip install pdfplumber")
            return
        
        try:
            pdf_info = PDFExtractor.get_pdf_info(self.current_file)
            
            info_text = f"PDF Information for: {self.current_file.name}\n\n"
            for key, value in pdf_info.items():
                if key != 'error':
                    info_text += f"{key.title()}: {value}\n"
            
            file_size = self.format_file_size(self.current_file.stat().st_size)
            info_text += f"File Size: {file_size}\n"
            info_text += f"Full Path: {self.current_file}\n"
            
            messagebox.showinfo("PDF Information", info_text)
            
        except Exception as e:
            messagebox.showerror("PDF Info Error", f"Failed to get PDF info: {e}")
    
    def change_directory(self):
        """Change the documentation directory"""
        new_dir = filedialog.askdirectory(
            title="Select Documentation Directory",
            initialdir=str(self.docs_path.parent)
        )
        
        if new_dir:
            self.docs_path = Path(new_dir)
            self.refresh_structure()
    
    def refresh_structure(self):
        """Refresh the documentation structure analysis"""
        try:
            # Re-analyze structure
            self.analyzer = DocumentationStructureAnalyzer(self.docs_path)
            self.structure = self.analyzer.analyze_structure()
            self.categories = self.structure.get('categories', {})
            self.role_mappings = self.analyzer.generate_role_mappings()
            
            # Clear PDF cache
            self.extracted_pdfs.clear()
            
            # Update UI
            project_name = self.docs_path.name.replace('-', ' ').replace('_', ' ').title()
            self.root.title(f"Universal Documentation Browser - {project_name}")
            
            # Reload tree
            self.load_documentation_tree()
            
            # Update path label
            self.path_label.config(text=f"Path: {self.docs_path}")
            
            messagebox.showinfo("Refresh", "Documentation structure refreshed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh structure: {e}")
    
    def show_in_folder(self):
        """Show current file in system folder"""
        if not self.current_file:
            messagebox.showwarning("Show in Folder", "No file selected")
            return
        
        try:
            folder_path = self.current_file.parent
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', str(folder_path)])
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', str(folder_path)])
            elif sys.platform.startswith('win'):  # Windows
                subprocess.run(['explorer', str(folder_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show in folder: {e}")
    
    def show_role_documents(self, role: str):
        """Show documents for a specific role"""
        docs = self.role_mappings.get(role, [])
        if not docs:
            messagebox.showinfo("Role Documents", f"No documents found for role: {role}")
            return
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"Quick Access - {role}")
        popup.geometry("700x500")
        popup.transient(self.root)
        popup.grab_set()
        
        frame = ttk.Frame(popup, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Recommended documents for {role}:", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        listbox = tk.Listbox(list_frame, font=('Arial', 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        # Add documents to listbox
        for doc in docs:
            full_path = self.docs_path / doc
            if full_path.exists():
                # Add file type indicator
                ext = full_path.suffix.lower()
                type_indicator = {"pdf": "📄", ".md": "📝", ".txt": "📄", ".html": "🌐"}.get(ext, "📁")
                listbox.insert(tk.END, f"{type_indicator} {doc}")
            else:
                listbox.insert(tk.END, f"❌ {doc} (not found)")
        
        # Button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        def open_selected():
            selection = listbox.curselection()
            if selection:
                doc_path = docs[selection[0]]
                full_path = self.docs_path / doc_path
                if full_path.exists():
                    self.load_file_content(full_path)
                    popup.destroy()
                else:
                    messagebox.showerror("Error", f"Document not found: {doc_path}")
        
        ttk.Button(btn_frame, text="Open Document", command=open_selected).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=popup.destroy).pack(side=tk.RIGHT)
        
        listbox.bind('<Double-1>', lambda e: open_selected())
    
    def on_search_change(self, *args):
        """Handle search text change"""
        # Real-time search could be implemented here
        pass
    
    def perform_search(self):
        """Perform search across all documents including PDF content"""
        search_term = self.search_var.get().strip()
        if not search_term:
            messagebox.showwarning("Search", "Please enter a search term")
            return
        
        self.status_var.set(f"Searching for: {search_term}")
        results = []
        
        try:
            # Search through all accessible files
            for cat_name, cat_info in self.categories.items():
                cat_path = Path(cat_info['path'])
                
                if cat_name == '_root':
                    files_to_search = [self.docs_path / f for f in cat_info.get('files', [])]
                else:
                    files_to_search = self._get_category_files(cat_path)
                
                for file_path in files_to_search:
                    # Search in filename
                    if search_term.lower() in file_path.name.lower():
                        results.append((file_path, "filename", f"Found in filename: {file_path.name}"))
                    
                    # Search in content based on file type
                    if file_path.suffix.lower() == '.pdf':
                        # Search in PDF content (use cached if available)
                        if PDF_EXTRACTION_AVAILABLE:
                            if str(file_path) in self.extracted_pdfs:
                                pdf_content = self.extracted_pdfs[str(file_path)]
                            else:
                                # Extract text for search (don't cache for search only)
                                pdf_content = PDFExtractor.extract_text(file_path)
                            
                            if pdf_content and search_term.lower() in pdf_content.lower():
                                # Find context
                                lines = pdf_content.split('\n')
                                for i, line in enumerate(lines, 1):
                                    if search_term.lower() in line.lower():
                                        context = line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                                        results.append((file_path, f"PDF line {i}", context))
                                        break  # Only show first match per PDF
                    
                    elif file_path.suffix.lower() in ['.md', '.txt', '.rst', '.html', '.htm']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Search in content
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if search_term.lower() in line.lower():
                                    context = line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                                    results.append((file_path, f"line {i}", context))
                                    if len([r for r in results if r[0] == file_path]) >= 3:  # Max 3 results per file
                                        break
                        
                        except Exception:
                            continue  # Skip files that can't be read
            
            self.show_search_results(search_term, results)
            
        except Exception as e:
            messagebox.showerror("Search Error", f"Search failed: {e}")
            self.status_var.set("Search failed")
    
    def show_search_results(self, search_term: str, results):
        """Show search results in popup with enhanced display"""
        if not results:
            messagebox.showinfo("Search Results", f"No results found for: {search_term}")
            self.status_var.set(f"No results found for: {search_term}")
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Search Results - {search_term}")
        results_window.geometry("1000x600")
        results_window.transient(self.root)
        
        frame = ttk.Frame(results_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Found {len(results)} results for: {search_term}", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Results tree
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        columns = ('file', 'location', 'context')
        results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        results_tree.heading('file', text='File')
        results_tree.heading('location', text='Location')
        results_tree.heading('context', text='Context')
        
        results_tree.column('file', width=300)
        results_tree.column('location', width=150)
        results_tree.column('context', width=500)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=results_tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=results_tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        results_tree.configure(xscrollcommand=h_scrollbar.set)
        
        results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add results
        for file_path, location, context in results:
            try:
                relative_path = file_path.relative_to(self.docs_path)
                # Add file type indicator
                ext = file_path.suffix.lower()
                type_indicator = {"pdf": "📄", ".md": "📝", ".txt": "📄", ".html": "🌐"}.get(ext, "📁")
                results_tree.insert('', 'end', values=(f"{type_indicator} {relative_path}", location, context))
            except ValueError:
                # Handle files outside the docs path
                results_tree.insert('', 'end', values=(str(file_path), location, context))
        
        def open_result():
            selection = results_tree.selection()
            if selection:
                item = results_tree.item(selection[0])
                file_name = item['values'][0]
                # Remove type indicator
                clean_name = re.sub(r'^📄 |^📝 |^🌐 |^📁 ', '', file_name)
                file_path = self.docs_path / clean_name
                if file_path.exists():
                    self.load_file_content(file_path)
                    results_window.destroy()
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Open Selected", command=open_result).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=results_window.destroy).pack(side=tk.RIGHT)
        
        results_tree.bind('<Double-1>', lambda e: open_result())
        
        self.status_var.set(f"Found {len(results)} results for: {search_term}")
    
    def open_external(self):
        """Open current file in external editor"""
        if not self.current_file:
            messagebox.showwarning("Open External", "No file selected")
            return
        
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', str(self.current_file)])
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', str(self.current_file)])
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(str(self.current_file))
            else:
                messagebox.showinfo("Open External", f"Please open manually: {self.current_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open external editor: {e}")
    
    def copy_file_path(self):
        """Copy current file path to clipboard"""
        if not self.current_file:
            messagebox.showwarning("Copy Path", "No file selected")
            return
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(str(self.current_file))
            self.status_var.set(f"Copied path: {self.current_file.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy path: {e}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Universal Documentation Browser with PDF extraction')
    parser.add_argument('directory', nargs='?', default=None,
                       help='Documentation directory (default: current directory or auto-detect)')
    parser.add_argument('--config', help='Use specific configuration file')
    
    args = parser.parse_args()
    
    # Determine documentation directory
    if args.directory:
        docs_path = Path(args.directory).resolve()
    else:
        # Try to find documentation directory
        current = Path.cwd()
        candidates = [
            current / 'docs',
            current / 'doc', 
            current / 'documentation',
            current
        ]
        
        docs_path = current
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Check if it contains documentation files
                doc_files = (list(candidate.glob('*.md')) + 
                           list(candidate.glob('*.rst')) + 
                           list(candidate.glob('*.txt')) + 
                           list(candidate.glob('*.pdf')))
                if doc_files or any(candidate.iterdir()):
                    docs_path = candidate
                    break
    
    if not docs_path.exists():
        print(f"Error: Directory not found: {docs_path}")
        sys.exit(1)
    
    if not docs_path.is_dir():
        print(f"Error: Not a directory: {docs_path}")
        sys.exit(1)
    
    print(f"Universal Documentation Browser with PDF extraction")
    print(f"Analyzing documentation at: {docs_path}")
    print(f"PDF extraction: {'✓ Available' if PDF_EXTRACTION_AVAILABLE else '✗ Install pdfplumber'}")
    
    # Create and run the application
    root = tk.Tk()
    
    try:
        app = UniversalDocumentationBrowser(root, docs_path)
        
        def on_closing():
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start browser: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()