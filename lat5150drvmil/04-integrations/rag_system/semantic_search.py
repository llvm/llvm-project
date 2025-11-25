"""
Semantic Code Search (Phase 2.2)

Intelligent code search that goes beyond grep with natural language queries
and semantic understanding of code patterns.

Features:
- Natural language queries: "Find SQL queries without parameterization"
- Pattern-based search: "Functions with cyclomatic complexity > 10"
- Similarity search: "Find code similar to this snippet"
- Cross-reference analysis: "Where is this function called?"
- Dependency graphs: "Show all dependencies of this module"

Examples:
    >>> searcher = SemanticCodeSearch("/path/to/codebase")
    >>> results = searcher.search("Find all SQL injection vulnerabilities")
    >>> results = searcher.search("Functions longer than 50 lines")
    >>> results = searcher.search_similar(reference_code)
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from enum import Enum
import hashlib


class SearchIntentType(Enum):
    """Types of search intents"""
    VULNERABILITY = "vulnerability"         # Security issues
    COMPLEXITY = "complexity"               # Code complexity
    PATTERN = "pattern"                     # Code patterns
    SIMILARITY = "similarity"               # Similar code
    DEPENDENCY = "dependency"               # Dependencies
    USAGE = "usage"                         # Function/class usage
    GENERAL = "general"                     # General text search


@dataclass
class SearchIntent:
    """Parsed search intent"""
    type: SearchIntentType
    query: str
    threshold: Optional[float] = None       # For complexity searches
    pattern: Optional[str] = None           # For pattern searches
    reference_code: Optional[str] = None    # For similarity searches
    confidence: float = 1.0


@dataclass
class SearchResult:
    """A single search result"""
    file_path: str
    line_number: int
    match_type: str
    snippet: str
    context: str  # Surrounding code for context
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResultSet:
    """Collection of search results"""
    query: str
    intent: SearchIntent
    results: List[SearchResult]
    total_files_searched: int
    execution_time: float


class IntentParser:
    """Parse natural language search queries into structured intents"""

    # Vulnerability patterns
    VULNERABILITY_KEYWORDS = {
        'sql injection': r'(execute|cursor\.execute)\s*\([^)]*[%+]',
        'xss': r'(innerHTML|outerHTML|document\.write)',
        'path traversal': r'open\s*\([^)]*\.\.',
        'command injection': r'(os\.system|subprocess\.|exec\()',
        'hardcoded': r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+',
    }

    # Complexity patterns
    COMPLEXITY_KEYWORDS = ['complex', 'complicated', 'long', 'cyclomatic']

    # Pattern keywords
    PATTERN_KEYWORDS = ['pattern', 'using', 'contains', 'with']

    def parse(self, query: str) -> SearchIntent:
        """Parse natural language query into search intent"""

        query_lower = query.lower()

        # Check for vulnerability search
        for vuln_name, pattern in self.VULNERABILITY_KEYWORDS.items():
            if vuln_name in query_lower or 'vulnerability' in query_lower or 'vuln' in query_lower:
                return SearchIntent(
                    type=SearchIntentType.VULNERABILITY,
                    query=query,
                    pattern=pattern if vuln_name in query_lower else None,
                    confidence=0.9
                )

        # Check for complexity search
        if any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS):
            # Extract threshold if present
            threshold = self._extract_number(query)
            return SearchIntent(
                type=SearchIntentType.COMPLEXITY,
                query=query,
                threshold=threshold,
                confidence=0.85
            )

        # Check for similarity search
        if 'similar' in query_lower or 'like' in query_lower:
            return SearchIntent(
                type=SearchIntentType.SIMILARITY,
                query=query,
                confidence=0.8
            )

        # Check for dependency search
        if 'depend' in query_lower or 'import' in query_lower or 'use' in query_lower:
            return SearchIntent(
                type=SearchIntentType.DEPENDENCY,
                query=query,
                confidence=0.75
            )

        # Check for usage search
        if 'where' in query_lower and ('call' in query_lower or 'use' in query_lower):
            return SearchIntent(
                type=SearchIntentType.USAGE,
                query=query,
                confidence=0.8
            )

        # Default to general search
        return SearchIntent(
            type=SearchIntentType.GENERAL,
            query=query,
            confidence=0.5
        )

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric threshold from text"""
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return float(numbers[0]) if numbers else None


class CodeIndexer:
    """Index codebase for fast searching"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.file_index: Dict[str, Dict] = {}
        self.function_index: Dict[str, List[Tuple[str, int]]] = {}  # func_name -> [(file, line), ...]
        self.import_graph: Dict[str, Set[str]] = {}  # file -> set of imports
        self.call_graph: Dict[str, Set[str]] = {}    # function -> set of callees

    def build_index(self, file_extensions: List[str] = ['.py']) -> None:
        """Build search index for codebase"""

        for ext in file_extensions:
            for file_path in self.root_path.rglob(f'*{ext}'):
                if self._should_skip(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()

                    self._index_file(str(file_path), code)
                except (UnicodeDecodeError, PermissionError):
                    continue

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        return any(skip_dir in file_path.parts for skip_dir in skip_dirs)

    def _index_file(self, file_path: str, code: str) -> None:
        """Index a single file"""

        # Basic file metadata
        self.file_index[file_path] = {
            'lines': code.split('\n'),
            'line_count': len(code.split('\n')),
            'size': len(code),
            'hash': hashlib.md5(code.encode()).hexdigest()
        }

        # Parse AST for deeper indexing
        try:
            tree = ast.parse(code)
            self._index_ast(file_path, tree)
        except SyntaxError:
            pass

    def _index_ast(self, file_path: str, tree: ast.AST) -> None:
        """Index AST for functions, imports, and calls"""

        for node in ast.walk(tree):
            # Index function definitions
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                if func_name not in self.function_index:
                    self.function_index[func_name] = []
                self.function_index[func_name].append((file_path, node.lineno))

            # Index imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if file_path not in self.import_graph:
                    self.import_graph[file_path] = set()

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_graph[file_path].add(alias.name)
                else:
                    if node.module:
                        self.import_graph[file_path].add(node.module)

            # Index function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    caller = self._find_containing_function(tree, node)
                    if caller:
                        if caller not in self.call_graph:
                            self.call_graph[caller] = set()
                        self.call_graph[caller].add(node.func.id)

    def _find_containing_function(self, tree: ast.AST, target_node: ast.AST) -> Optional[str]:
        """Find the function containing a given node"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if child is target_node:
                        return node.name
        return None


class SemanticCodeSearch:
    """Natural language code search engine"""

    def __init__(self, codebase_path: str, auto_index: bool = True):
        self.codebase_path = Path(codebase_path)
        self.indexer = CodeIndexer(codebase_path)
        self.parser = IntentParser()

        if auto_index:
            self.indexer.build_index()

    def search(self, query: str, max_results: int = 20) -> SearchResultSet:
        """Search codebase with natural language query"""
        import time
        start_time = time.time()

        # Parse intent
        intent = self.parser.parse(query)

        # Route to appropriate search method
        if intent.type == SearchIntentType.VULNERABILITY:
            results = self._search_vulnerabilities(intent)
        elif intent.type == SearchIntentType.COMPLEXITY:
            results = self._search_complexity(intent)
        elif intent.type == SearchIntentType.SIMILARITY:
            results = self._search_similar(intent)
        elif intent.type == SearchIntentType.DEPENDENCY:
            results = self._search_dependencies(intent)
        elif intent.type == SearchIntentType.USAGE:
            results = self._search_usage(intent)
        else:
            results = self._search_general(intent)

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:max_results]

        return SearchResultSet(
            query=query,
            intent=intent,
            results=results,
            total_files_searched=len(self.indexer.file_index),
            execution_time=time.time() - start_time
        )

    def _search_vulnerabilities(self, intent: SearchIntent) -> List[SearchResult]:
        """Search for security vulnerabilities"""
        results = []

        # Use pattern if available, otherwise use general vulnerability patterns
        patterns = [intent.pattern] if intent.pattern else self.parser.VULNERABILITY_KEYWORDS.values()

        for file_path, file_data in self.indexer.file_index.items():
            code = '\n'.join(file_data['lines'])

            for pattern in patterns:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    line_no = code[:match.start()].count('\n') + 1
                    snippet = file_data['lines'][line_no - 1].strip()
                    context = self._get_context(file_data['lines'], line_no)

                    results.append(SearchResult(
                        file_path=file_path,
                        line_number=line_no,
                        match_type="vulnerability",
                        snippet=snippet,
                        context=context,
                        score=0.9,
                        metadata={'pattern': pattern}
                    ))

        return results

    def _search_complexity(self, intent: SearchIntent) -> List[SearchResult]:
        """Search for complex code"""
        results = []
        threshold = intent.threshold or 10

        for file_path, file_data in self.indexer.file_index.items():
            code = '\n'.join(file_data['lines'])

            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_complexity(node)

                        if complexity > threshold:
                            snippet = f"def {node.name}(...) - Complexity: {complexity}"
                            context = self._get_context(file_data['lines'], node.lineno)

                            # Score based on how much it exceeds threshold
                            score = min(1.0, complexity / (threshold * 2))

                            results.append(SearchResult(
                                file_path=file_path,
                                line_number=node.lineno,
                                match_type="high_complexity",
                                snippet=snippet,
                                context=context,
                                score=score,
                                metadata={'complexity': complexity, 'function': node.name}
                            ))
            except SyntaxError:
                continue

        return results

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _search_similar(self, intent: SearchIntent) -> List[SearchResult]:
        """Search for similar code patterns"""
        results = []

        # Extract keywords from query for similarity matching
        keywords = self._extract_keywords(intent.query)

        for file_path, file_data in self.indexer.file_index.items():
            code = '\n'.join(file_data['lines'])

            # Simple similarity: count keyword matches
            for i, line in enumerate(file_data['lines'], 1):
                matches = sum(1 for kw in keywords if kw in line.lower())
                if matches > 0:
                    score = matches / len(keywords) if keywords else 0

                    results.append(SearchResult(
                        file_path=file_path,
                        line_number=i,
                        match_type="similar",
                        snippet=line.strip(),
                        context=self._get_context(file_data['lines'], i),
                        score=score,
                        metadata={'keyword_matches': matches}
                    ))

        return results

    def _search_dependencies(self, intent: SearchIntent) -> List[SearchResult]:
        """Search for dependencies"""
        results = []

        # Extract module name from query
        keywords = self._extract_keywords(intent.query)

        for file_path, imports in self.indexer.import_graph.items():
            for imp in imports:
                # Check if any keyword matches import
                if any(kw in imp.lower() for kw in keywords):
                    file_data = self.indexer.file_index[file_path]

                    # Find the import line
                    for i, line in enumerate(file_data['lines'], 1):
                        if 'import' in line and imp in line:
                            results.append(SearchResult(
                                file_path=file_path,
                                line_number=i,
                                match_type="dependency",
                                snippet=line.strip(),
                                context=self._get_context(file_data['lines'], i),
                                score=0.8,
                                metadata={'module': imp}
                            ))
                            break

        return results

    def _search_usage(self, intent: SearchIntent) -> List[SearchResult]:
        """Search for function/class usage"""
        results = []

        # Extract function name from query
        keywords = self._extract_keywords(intent.query)

        for func_name, locations in self.indexer.function_index.items():
            # Check if function name matches any keyword
            if any(kw in func_name.lower() for kw in keywords):
                for file_path, line_no in locations:
                    file_data = self.indexer.file_index[file_path]
                    snippet = file_data['lines'][line_no - 1].strip()
                    context = self._get_context(file_data['lines'], line_no)

                    results.append(SearchResult(
                        file_path=file_path,
                        line_number=line_no,
                        match_type="definition",
                        snippet=snippet,
                        context=context,
                        score=0.9,
                        metadata={'function': func_name}
                    ))

                # Find usages in call graph
                if func_name in self.indexer.call_graph:
                    for caller in self.indexer.call_graph[func_name]:
                        results.append(SearchResult(
                            file_path="call_graph",
                            line_number=0,
                            match_type="call",
                            snippet=f"{caller} → {func_name}",
                            context="",
                            score=0.7,
                            metadata={'caller': caller, 'callee': func_name}
                        ))

        return results

    def _search_general(self, intent: SearchIntent) -> List[SearchResult]:
        """General text-based search"""
        results = []
        keywords = self._extract_keywords(intent.query)

        for file_path, file_data in self.indexer.file_index.items():
            for i, line in enumerate(file_data['lines'], 1):
                # Check if line contains any keywords
                matches = sum(1 for kw in keywords if kw in line.lower())
                if matches > 0:
                    score = matches / len(keywords) if keywords else 0

                    results.append(SearchResult(
                        file_path=file_path,
                        line_number=i,
                        match_type="text_match",
                        snippet=line.strip(),
                        context=self._get_context(file_data['lines'], i),
                        score=score,
                        metadata={'keyword_matches': matches}
                    ))

        return results

    def search_similar_to_code(self, reference_code: str, max_results: int = 10) -> SearchResultSet:
        """Find code similar to reference snippet"""
        import time
        start_time = time.time()

        results = []

        # Extract features from reference code
        ref_keywords = self._extract_keywords(reference_code)
        ref_ast_patterns = self._extract_ast_patterns(reference_code)

        for file_path, file_data in self.indexer.file_index.items():
            code = '\n'.join(file_data['lines'])

            # Try to parse and compare AST patterns
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_code = ast.get_source_segment(code, node)
                        if func_code:
                            similarity = self._calculate_similarity(
                                func_code, reference_code, ref_keywords, ref_ast_patterns
                            )

                            if similarity > 0.3:  # Threshold
                                results.append(SearchResult(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    match_type="similar_function",
                                    snippet=f"def {node.name}(...)",
                                    context=func_code[:200],
                                    score=similarity,
                                    metadata={'function': node.name, 'similarity': similarity}
                                ))
            except SyntaxError:
                continue

        results.sort(key=lambda x: x.score, reverse=True)

        return SearchResultSet(
            query=f"Similar to: {reference_code[:50]}...",
            intent=SearchIntent(type=SearchIntentType.SIMILARITY, query="similarity search", reference_code=reference_code),
            results=results[:max_results],
            total_files_searched=len(self.indexer.file_index),
            execution_time=time.time() - start_time
        )

    def _calculate_similarity(self, code1: str, code2: str, ref_keywords: Set[str], ref_patterns: Set[str]) -> float:
        """Calculate similarity between two code snippets"""

        # Keyword-based similarity
        code1_keywords = self._extract_keywords(code1)
        keyword_similarity = len(code1_keywords & ref_keywords) / max(len(ref_keywords), 1)

        # AST pattern similarity
        code1_patterns = self._extract_ast_patterns(code1)
        pattern_similarity = len(code1_patterns & ref_patterns) / max(len(ref_patterns), 1)

        # Weighted combination
        return 0.6 * keyword_similarity + 0.4 * pattern_similarity

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}

        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _extract_ast_patterns(self, code: str) -> Set[str]:
        """Extract AST patterns from code"""
        patterns = set()

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Record node types as patterns
                patterns.add(type(node).__name__)

                # Record specific patterns
                if isinstance(node, ast.FunctionDef):
                    patterns.add(f"func_{len(node.args.args)}_args")
                elif isinstance(node, ast.For):
                    patterns.add("for_loop")
                elif isinstance(node, ast.If):
                    patterns.add("if_statement")
        except SyntaxError:
            pass

        return patterns

    def _get_context(self, lines: List[str], line_no: int, context_size: int = 2) -> str:
        """Get surrounding context for a line"""
        start = max(0, line_no - context_size - 1)
        end = min(len(lines), line_no + context_size)
        context_lines = lines[start:end]
        return '\n'.join(context_lines)

    def format_results(self, result_set: SearchResultSet) -> str:
        """Format search results as readable text"""

        lines = []
        lines.append("=" * 80)
        lines.append(f"SEARCH RESULTS: {result_set.query}")
        lines.append("=" * 80)
        lines.append(f"Intent: {result_set.intent.type.value}")
        lines.append(f"Results: {len(result_set.results)} matches")
        lines.append(f"Files searched: {result_set.total_files_searched}")
        lines.append(f"Time: {result_set.execution_time:.2f}s")
        lines.append("")

        for i, result in enumerate(result_set.results[:10], 1):  # Show top 10
            lines.append(f"{i}. {result.file_path}:{result.line_number}")
            lines.append(f"   Type: {result.match_type} | Score: {result.score:.2f}")
            lines.append(f"   {result.snippet}")
            if result.metadata:
                metadata_str = ', '.join(f"{k}={v}" for k, v in result.metadata.items())
                lines.append(f"   Metadata: {metadata_str}")
            lines.append("")

        if len(result_set.results) > 10:
            lines.append(f"... and {len(result_set.results) - 10} more results")

        lines.append("=" * 80)

        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    # Test semantic search
    searcher = SemanticCodeSearch(".", auto_index=True)

    # Natural language queries
    print("Query 1: Find SQL injection vulnerabilities")
    results = searcher.search("Find SQL injection vulnerabilities")
    print(searcher.format_results(results))
    print()

    print("Query 2: Functions with high complexity")
    results = searcher.search("Find functions with cyclomatic complexity greater than 10")
    print(searcher.format_results(results))
    print()

    print("Query 3: Where is this function used?")
    results = searcher.search("Where is the review function called?")
    print(searcher.format_results(results))
