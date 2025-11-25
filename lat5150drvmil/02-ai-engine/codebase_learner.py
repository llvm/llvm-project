#!/usr/bin/env python3
"""
Codebase Learner for Local Claude Code
Incremental learning from codebases to improve code generation and editing
"""

import json
import time
import hashlib
import ast
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """Learned code pattern"""
    pattern_id: str
    pattern_type: str  # 'function', 'class', 'idiom', 'architecture'
    name: str
    description: str
    code_example: str
    file_path: str
    language: str
    frequency: int = 1
    quality_score: float = 0.5  # 0-1, based on usage and structure
    tags: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    contexts: List[str] = field(default_factory=list)  # Where this pattern appears


@dataclass
class CodingStyle:
    """Learned coding style preferences"""
    indent_style: str = "spaces"  # 'spaces' or 'tabs'
    indent_size: int = 4
    quote_style: str = "double"  # 'single' or 'double'
    line_length: int = 100
    naming_convention: Dict[str, str] = field(default_factory=dict)  # class, function, variable
    import_style: str = "grouped"  # 'grouped', 'alphabetical', 'type-based'
    docstring_style: str = "google"  # 'google', 'numpy', 'sphinx'
    type_hints: bool = True


@dataclass
class ArchitecturalPattern:
    """Architectural design pattern"""
    pattern_name: str
    pattern_type: str  # 'mvc', 'layered', 'microservices', etc.
    description: str
    components: List[str]
    relationships: Dict[str, List[str]]
    examples: List[str]  # File paths demonstrating pattern


class CodebaseLearner:
    """
    Learns from codebases to improve code generation

    Features:
    - Pattern recognition (functions, classes, architectures)
    - Style learning (naming, formatting, conventions)
    - Best practice extraction
    - Common idiom detection
    - Architecture understanding
    - Incremental learning as code is processed
    """

    def __init__(
        self,
        workspace_root: str = ".",
        rag_system=None,
        storage_system=None
    ):
        """
        Initialize codebase learner

        Args:
            workspace_root: Project root directory
            rag_system: RAG system for semantic search
            storage_system: Storage system for persistence
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.rag = rag_system
        self.storage = storage_system

        # Learned knowledge
        self.patterns: Dict[str, CodePattern] = {}
        self.coding_style = CodingStyle()
        self.architectural_patterns: List[ArchitecturalPattern] = []

        # Statistics
        self.files_analyzed = 0
        self.functions_learned = 0
        self.classes_learned = 0
        self.patterns_learned = 0

        # Common patterns tracking
        self.function_signatures: Counter = Counter()
        self.class_hierarchies: Dict[str, List[str]] = defaultdict(list)
        self.import_patterns: Counter = Counter()
        self.naming_patterns: Dict[str, Counter] = {
            'class': Counter(),
            'function': Counter(),
            'variable': Counter()
        }

        # Phase 3: Call graph analysis
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)  # func -> set of called funcs
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)  # func -> set of callers
        self.function_locations: Dict[str, Dict[str, Any]] = {}  # func -> {file, line, class}
        self.class_methods: Dict[str, List[str]] = defaultdict(list)  # class -> methods

        # Knowledge persistence
        self.knowledge_file = self.workspace_root / ".local_claude_code" / "learned_knowledge.json"
        self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing knowledge
        self.load_knowledge()

        logger.info(f"CodebaseLearner initialized with {len(self.patterns)} patterns")

    def learn_from_file(self, filepath: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Learn patterns and style from a file

        Args:
            filepath: Path to file
            content: File content (if not provided, will read from disk)

        Returns:
            Summary of what was learned
        """
        filepath = Path(filepath).resolve()

        if not content:
            if not filepath.exists():
                return {"error": f"File not found: {filepath}"}

            try:
                content = filepath.read_text()
            except Exception as e:
                return {"error": f"Could not read file: {e}"}

        learned = {
            "filepath": str(filepath),
            "patterns": [],
            "style_updates": [],
            "functions": 0,
            "classes": 0
        }

        # Analyze based on file type
        if filepath.suffix == '.py':
            self._learn_from_python(filepath, content, learned)
        elif filepath.suffix in ['.js', '.ts']:
            self._learn_from_javascript(filepath, content, learned)
        # Add more language support as needed

        # Learn general style
        self._learn_style(content, learned)

        # Store in RAG if available
        if self.rag:
            self._index_in_rag(filepath, content, learned)

        self.files_analyzed += 1

        logger.info(f"Learned from {filepath.name}: {learned['functions']} functions, {learned['classes']} classes")

        return learned

    def _learn_from_python(self, filepath: Path, content: str, learned: Dict):
        """Learn patterns from Python code"""
        try:
            tree = ast.parse(content)

            # First pass: collect all function and class definitions
            current_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    current_class = node.name
                    self._learn_class_pattern(node, filepath, content, learned)

            # Second pass: extract patterns and build call graph
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Determine if this is a method or standalone function
                    parent_class = self._find_parent_class(node, tree)
                    self._learn_function_pattern(node, filepath, content, learned, parent_class)
                    self._build_call_graph_for_function(node, filepath, parent_class)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._learn_import_pattern(node, learned)

        except SyntaxError as e:
            learned['error'] = f"Syntax error: {e}"
        except Exception as e:
            learned['error'] = f"Parse error: {e}"

    def _learn_function_pattern(self, node: ast.FunctionDef, filepath: Path, content: str, learned: Dict, parent_class: Optional[str] = None):
        """Learn from function definition"""
        func_name = node.name

        # Create fully qualified name for methods
        if parent_class:
            qualified_name = f"{parent_class}.{func_name}"
            self.class_methods[parent_class].append(func_name)
        else:
            qualified_name = func_name

        # Record function location (Phase 3)
        self.function_locations[qualified_name] = {
            'file': str(filepath),
            'line': node.lineno,
            'class': parent_class,
            'is_method': parent_class is not None
        }

        # Extract function code
        try:
            func_lines = content.splitlines()[node.lineno - 1:node.end_lineno]
            func_code = '\n'.join(func_lines)
        except:
            func_code = ""

        # Generate pattern ID
        pattern_id = hashlib.md5(f"{filepath}:{func_name}".encode()).hexdigest()[:16]

        # Analyze function structure
        args = [arg.arg for arg in node.args.args]
        has_docstring = ast.get_docstring(node) is not None
        has_type_hints = any(arg.annotation for arg in node.args.args)
        decorators = [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]

        # Determine function type/tags
        tags = []
        if func_name.startswith('test_'):
            tags.append('test')
        if func_name.startswith('_'):
            tags.append('private')
        if decorators:
            tags.extend(decorators)
        if has_docstring:
            tags.append('documented')
        if has_type_hints:
            tags.append('typed')

        # Create or update pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = time.time()
        else:
            pattern = CodePattern(
                pattern_id=pattern_id,
                pattern_type='function',
                name=func_name,
                description=ast.get_docstring(node) or f"Function {func_name}",
                code_example=func_code[:500],  # Limit size
                file_path=str(filepath),
                language='python',
                tags=tags
            )
            self.patterns[pattern_id] = pattern

        # Update statistics
        self.function_signatures[func_name] += 1
        self.functions_learned += 1
        learned['functions'] += 1
        learned['patterns'].append({'type': 'function', 'name': func_name})

        # Learn naming convention
        if func_name.islower():
            self.naming_patterns['function']['snake_case'] += 1
        elif any(c.isupper() for c in func_name[1:]):
            self.naming_patterns['function']['camelCase'] += 1

    def _learn_class_pattern(self, node: ast.ClassDef, filepath: Path, content: str, learned: Dict):
        """Learn from class definition"""
        class_name = node.name

        # Extract class code
        try:
            class_lines = content.splitlines()[node.lineno - 1:node.end_lineno]
            class_code = '\n'.join(class_lines)
        except:
            class_code = ""

        # Generate pattern ID
        pattern_id = hashlib.md5(f"{filepath}:{class_name}".encode()).hexdigest()[:16]

        # Analyze class structure
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        base_classes = [self._get_name(base) for base in node.bases]
        has_docstring = ast.get_docstring(node) is not None

        # Tags
        tags = ['class']
        if base_classes:
            tags.append('inherits')
        if '__init__' in methods:
            tags.append('constructor')
        if has_docstring:
            tags.append('documented')

        # Create or update pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = time.time()
        else:
            pattern = CodePattern(
                pattern_id=pattern_id,
                pattern_type='class',
                name=class_name,
                description=ast.get_docstring(node) or f"Class {class_name}",
                code_example=class_code[:500],
                file_path=str(filepath),
                language='python',
                tags=tags
            )
            self.patterns[pattern_id] = pattern

        # Track inheritance
        if base_classes:
            self.class_hierarchies[class_name] = base_classes

        # Update statistics
        self.classes_learned += 1
        learned['classes'] += 1
        learned['patterns'].append({'type': 'class', 'name': class_name})

        # Learn naming convention
        if class_name[0].isupper():
            self.naming_patterns['class']['PascalCase'] += 1

    def _learn_import_pattern(self, node, learned: Dict):
        """Learn from import statements"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.import_patterns[alias.name] += 1
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self.import_patterns[node.module] += 1

    def _learn_from_javascript(self, filepath: Path, content: str, learned: Dict):
        """Learn patterns from JavaScript/TypeScript code"""
        # Basic pattern matching for JavaScript
        import re

        # Find function declarations
        func_pattern = r'function\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            self.function_signatures[func_name] += 1
            learned['functions'] += 1

        # Find class declarations
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            learned['classes'] += 1

    def _learn_style(self, content: str, learned: Dict):
        """Learn coding style from content"""
        lines = content.splitlines()

        if not lines:
            return

        # Detect indentation
        indent_counts = defaultdict(int)
        for line in lines:
            if line and line[0] in [' ', '\t']:
                if line.startswith('\t'):
                    indent_counts['tabs'] += 1
                else:
                    # Count leading spaces
                    spaces = len(line) - len(line.lstrip(' '))
                    if spaces > 0:
                        indent_counts[f'spaces_{spaces}'] += 1

        if indent_counts:
            most_common = max(indent_counts, key=indent_counts.get)
            if most_common == 'tabs':
                if self.coding_style.indent_style != 'tabs':
                    self.coding_style.indent_style = 'tabs'
                    learned['style_updates'].append('indent_style: tabs')
            elif most_common.startswith('spaces_'):
                size = int(most_common.split('_')[1])
                if self.coding_style.indent_style != 'spaces' or self.coding_style.indent_size != size:
                    self.coding_style.indent_style = 'spaces'
                    self.coding_style.indent_size = size
                    learned['style_updates'].append(f'indent: {size} spaces')

        # Detect quote style (Python/JavaScript)
        import re
        single_quotes = len(re.findall(r"'[^']*'", content))
        double_quotes = len(re.findall(r'"[^"]*"', content))

        if single_quotes > double_quotes * 1.5:
            if self.coding_style.quote_style != 'single':
                self.coding_style.quote_style = 'single'
                learned['style_updates'].append('quotes: single')
        elif double_quotes > single_quotes * 1.5:
            if self.coding_style.quote_style != 'double':
                self.coding_style.quote_style = 'double'
                learned['style_updates'].append('quotes: double')

        # Line length
        max_line_len = max(len(line) for line in lines) if lines else 0
        avg_line_len = sum(len(line) for line in lines) / len(lines) if lines else 0

        if avg_line_len > 90:
            self.coding_style.line_length = 120
        elif avg_line_len < 70:
            self.coding_style.line_length = 80

    def _index_in_rag(self, filepath: Path, content: str, learned: Dict):
        """Index learned patterns in RAG system"""
        try:
            # Index the file
            doc_id = str(filepath.relative_to(self.workspace_root))

            self.rag.index_document(
                text=content[:2000],  # First 2000 chars
                doc_id=doc_id,
                metadata={
                    'type': 'code',
                    'language': filepath.suffix[1:],
                    'functions': learned.get('functions', 0),
                    'classes': learned.get('classes', 0),
                    'filepath': str(filepath)
                }
            )

            # Index individual patterns
            for pattern_info in learned.get('patterns', []):
                pattern_id = f"{doc_id}:{pattern_info['name']}"

                pattern_text = f"{pattern_info['type']} {pattern_info['name']} from {filepath.name}"

                self.rag.index_document(
                    text=pattern_text,
                    doc_id=pattern_id,
                    metadata={
                        'type': 'pattern',
                        'pattern_type': pattern_info['type'],
                        'name': pattern_info['name'],
                        'filepath': str(filepath)
                    }
                )

        except Exception as e:
            logger.error(f"Error indexing in RAG: {e}")

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def get_similar_patterns(self, query: str, top_k: int = 5) -> List[CodePattern]:
        """Find similar patterns to query"""
        if self.rag:
            # Use RAG for semantic search
            try:
                results = self.rag.search(query, top_k=top_k)

                patterns = []
                for result in results:
                    pattern_id = result.metadata.get('pattern_id')
                    if pattern_id and pattern_id in self.patterns:
                        patterns.append(self.patterns[pattern_id])

                return patterns
            except Exception as e:
                logger.error(f"RAG search failed: {e}")

        # Fallback: simple text matching
        query_lower = query.lower()
        matches = []

        for pattern in self.patterns.values():
            score = 0
            if query_lower in pattern.name.lower():
                score += 2
            if query_lower in pattern.description.lower():
                score += 1
            for tag in pattern.tags:
                if query_lower in tag.lower():
                    score += 1

            if score > 0:
                matches.append((score, pattern))

        matches.sort(reverse=True)
        return [pattern for _, pattern in matches[:top_k]]

    def get_naming_recommendation(self, entity_type: str) -> str:
        """Get naming convention recommendation"""
        if entity_type not in self.naming_patterns:
            return "unknown"

        counter = self.naming_patterns[entity_type]
        if not counter:
            return "unknown"

        most_common = counter.most_common(1)[0][0]
        return most_common

    def get_style_guide(self) -> Dict[str, Any]:
        """Get learned coding style guide"""
        return {
            "indentation": {
                "style": self.coding_style.indent_style,
                "size": self.coding_style.indent_size if self.coding_style.indent_style == "spaces" else 1
            },
            "quotes": self.coding_style.quote_style,
            "line_length": self.coding_style.line_length,
            "naming_conventions": {
                "class": self.get_naming_recommendation('class'),
                "function": self.get_naming_recommendation('function'),
                "variable": self.get_naming_recommendation('variable')
            },
            "type_hints": self.coding_style.type_hints,
            "docstring_style": self.coding_style.docstring_style
        }

    def get_common_imports(self, top_n: int = 10) -> List[tuple]:
        """Get most common imports"""
        return self.import_patterns.most_common(top_n)

    # Phase 3: Call Graph Analysis Methods

    def _find_parent_class(self, func_node: ast.FunctionDef, tree: ast.AST) -> Optional[str]:
        """Find the parent class of a function node"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item == func_node or (isinstance(item, ast.FunctionDef) and item.name == func_node.name):
                        return node.name
        return None

    def _build_call_graph_for_function(self, node: ast.FunctionDef, filepath: Path, parent_class: Optional[str] = None):
        """Build call graph edges for a function by analyzing its calls"""
        if parent_class:
            caller = f"{parent_class}.{node.name}"
        else:
            caller = node.name

        # Find all function calls in this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = self._extract_call_name(child)

                if callee_name:
                    # Add to call graph
                    self.call_graph[caller].add(callee_name)
                    self.reverse_call_graph[callee_name].add(caller)

    def _extract_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from a Call node"""
        func = call_node.func

        if isinstance(func, ast.Name):
            # Simple function call: foo()
            return func.id

        elif isinstance(func, ast.Attribute):
            # Method call: obj.method() or self.method()
            if isinstance(func.value, ast.Name):
                if func.value.id == 'self':
                    # self.method() - need to qualify with current class
                    # This is handled during call graph construction
                    return func.attr
                else:
                    # obj.method()
                    return f"{func.value.id}.{func.attr}"
            else:
                return func.attr

        return None

    def find_dead_code(self) -> Dict[str, List[str]]:
        """
        Find functions that are never called (dead code)

        Returns:
            Dict with 'functions' and 'methods' lists
        """
        dead_functions = []
        dead_methods = []

        for func_name in self.function_locations.keys():
            # Check if function has no callers
            if func_name not in self.reverse_call_graph or len(self.reverse_call_graph[func_name]) == 0:
                # Exclude special methods and entry points
                if func_name.startswith('__') and func_name.endswith('__'):
                    continue
                if func_name in ['main', 'run', 'execute', 'start']:
                    continue
                if func_name.startswith('test_'):
                    continue

                func_info = self.function_locations[func_name]
                if func_info['is_method']:
                    dead_methods.append(func_name)
                else:
                    dead_functions.append(func_name)

        return {
            'functions': dead_functions,
            'methods': dead_methods,
            'total': len(dead_functions) + len(dead_methods)
        }

    def find_dependency_cycles(self) -> List[List[str]]:
        """
        Find circular dependencies in the call graph

        Returns:
            List of cycles, where each cycle is a list of function names
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) > 1:  # Ignore self-calls
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.call_graph.get(node, set()):
                dfs(neighbor, path.copy())

            rec_stack.remove(node)

        for func in self.call_graph.keys():
            if func not in visited:
                dfs(func, [])

        # Remove duplicate cycles
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            # Normalize cycle (rotate to start with smallest element)
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])

            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized))

        return unique_cycles

    def find_impact(self, function_name: str) -> Dict[str, Any]:
        """
        Analyze impact of modifying a function

        Returns what code would be affected if this function is changed

        Args:
            function_name: Name of function to analyze

        Returns:
            Dict with direct_callers, transitive_callers, impact_score
        """
        if function_name not in self.function_locations:
            return {
                'error': f"Function '{function_name}' not found",
                'direct_callers': [],
                'transitive_callers': [],
                'impact_score': 0
            }

        # Find all direct callers
        direct_callers = list(self.reverse_call_graph.get(function_name, set()))

        # Find transitive callers (all functions that depend on this, directly or indirectly)
        transitive_callers = set()

        def find_all_callers(func):
            for caller in self.reverse_call_graph.get(func, set()):
                if caller not in transitive_callers:
                    transitive_callers.add(caller)
                    find_all_callers(caller)

        find_all_callers(function_name)

        # Calculate impact score (0-100)
        total_functions = len(self.function_locations)
        if total_functions > 0:
            impact_score = min(100, int((len(transitive_callers) / total_functions) * 100))
        else:
            impact_score = 0

        return {
            'function': function_name,
            'direct_callers': direct_callers,
            'direct_caller_count': len(direct_callers),
            'transitive_callers': list(transitive_callers),
            'transitive_caller_count': len(transitive_callers),
            'impact_score': impact_score,
            'risk_level': 'high' if impact_score > 50 else 'medium' if impact_score > 20 else 'low'
        }

    def get_call_graph_stats(self) -> Dict[str, Any]:
        """Get call graph statistics"""
        dead_code = self.find_dead_code()
        cycles = self.find_dependency_cycles()

        # Find most called functions (hotspots)
        hotspots = sorted(
            [(func, len(callers)) for func, callers in self.reverse_call_graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Find functions that call the most other functions (complex functions)
        complex_functions = sorted(
            [(func, len(callees)) for func, callees in self.call_graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_functions': len(self.function_locations),
            'total_edges': sum(len(callees) for callees in self.call_graph.values()),
            'dead_code': dead_code,
            'dependency_cycles': cycles,
            'cycle_count': len(cycles),
            'hotspots': hotspots,
            'complex_functions': complex_functions,
            'avg_calls_per_function': sum(len(callees) for callees in self.call_graph.values()) / len(self.call_graph) if self.call_graph else 0
        }

    def save_knowledge(self):
        """Save learned knowledge to disk"""
        try:
            knowledge = {
                "metadata": {
                    "workspace_root": str(self.workspace_root),
                    "last_updated": time.time(),
                    "files_analyzed": self.files_analyzed,
                    "patterns_count": len(self.patterns)
                },
                "patterns": {
                    pid: asdict(pattern) for pid, pattern in self.patterns.items()
                },
                "coding_style": asdict(self.coding_style),
                "statistics": {
                    "functions_learned": self.functions_learned,
                    "classes_learned": self.classes_learned,
                    "common_imports": dict(self.import_patterns.most_common(50)),
                    "naming_patterns": {
                        k: dict(v.most_common(10)) for k, v in self.naming_patterns.items()
                    }
                },
                "call_graph": {
                    "call_graph": {k: list(v) for k, v in self.call_graph.items()},
                    "reverse_call_graph": {k: list(v) for k, v in self.reverse_call_graph.items()},
                    "function_locations": self.function_locations,
                    "class_methods": {k: v for k, v in self.class_methods.items()}
                }
            }

            with open(self.knowledge_file, 'w') as f:
                json.dump(knowledge, f, indent=2)

            logger.info(f"Knowledge saved: {len(self.patterns)} patterns")

        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")

    def load_knowledge(self) -> bool:
        """Load previously learned knowledge"""
        try:
            if not self.knowledge_file.exists():
                return False

            with open(self.knowledge_file, 'r') as f:
                knowledge = json.load(f)

            # Restore patterns
            for pid, pattern_data in knowledge.get('patterns', {}).items():
                self.patterns[pid] = CodePattern(**pattern_data)

            # Restore style
            if 'coding_style' in knowledge:
                self.coding_style = CodingStyle(**knowledge['coding_style'])

            # Restore statistics
            stats = knowledge.get('statistics', {})
            self.functions_learned = stats.get('functions_learned', 0)
            self.classes_learned = stats.get('classes_learned', 0)

            if 'common_imports' in stats:
                self.import_patterns = Counter(stats['common_imports'])

            if 'naming_patterns' in stats:
                for k, v in stats['naming_patterns'].items():
                    self.naming_patterns[k] = Counter(v)

            # Restore metadata
            metadata = knowledge.get('metadata', {})
            self.files_analyzed = metadata.get('files_analyzed', 0)

            # Restore call graph (Phase 3)
            if 'call_graph' in knowledge:
                cg_data = knowledge['call_graph']

                # Restore call graph
                if 'call_graph' in cg_data:
                    self.call_graph = defaultdict(set, {
                        k: set(v) for k, v in cg_data['call_graph'].items()
                    })

                # Restore reverse call graph
                if 'reverse_call_graph' in cg_data:
                    self.reverse_call_graph = defaultdict(set, {
                        k: set(v) for k, v in cg_data['reverse_call_graph'].items()
                    })

                # Restore function locations
                if 'function_locations' in cg_data:
                    self.function_locations = cg_data['function_locations']

                # Restore class methods
                if 'class_methods' in cg_data:
                    self.class_methods = defaultdict(list, cg_data['class_methods'])

            logger.info(f"Knowledge loaded: {len(self.patterns)} patterns, {len(self.function_locations)} functions from {self.files_analyzed} files")
            return True

        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
            return False

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "files_analyzed": self.files_analyzed,
            "patterns_learned": len(self.patterns),
            "functions_learned": self.functions_learned,
            "classes_learned": self.classes_learned,
            "common_function_names": self.function_signatures.most_common(10),
            "common_imports": self.import_patterns.most_common(10),
            "coding_style": self.get_style_guide()
        }


def main():
    """Example usage"""
    print("=== Codebase Learner Demo ===\n")

    learner = CodebaseLearner()

    # Example: Learn from a Python file
    example_code = '''
import os
import sys
from typing import List, Optional

class DataProcessor:
    """Process data with various transformations."""

    def __init__(self, data: List[str]):
        self.data = data

    def process(self) -> List[str]:
        """Process the data."""
        return [item.strip() for item in self.data]

    def filter_empty(self) -> List[str]:
        """Remove empty items."""
        return [item for item in self.data if item]

def helper_function(value: str) -> str:
    """Helper to process single value."""
    return value.lower().strip()
'''

    print("1. Learning from example code...")
    result = learner.learn_from_file("example.py", content=example_code)
    print(f"   Functions learned: {result['functions']}")
    print(f"   Classes learned: {result['classes']}")
    print(f"   Patterns: {len(result['patterns'])}")

    print("\n2. Style Guide:")
    style = learner.get_style_guide()
    print(f"   Indentation: {style['indentation']}")
    print(f"   Quotes: {style['quotes']}")
    print(f"   Naming: {style['naming_conventions']}")

    print("\n3. Common imports:")
    for imp, count in learner.get_common_imports(5):
        print(f"   {imp}: {count}")

    print("\n4. Learning stats:")
    stats = learner.get_learning_stats()
    print(f"   Files: {stats['files_analyzed']}")
    print(f"   Patterns: {stats['patterns_learned']}")

    print("\n5. Saving knowledge...")
    learner.save_knowledge()
    print(f"   Saved to: {learner.knowledge_file}")

    print("\n✓ Codebase Learner ready!")


if __name__ == "__main__":
    main()
