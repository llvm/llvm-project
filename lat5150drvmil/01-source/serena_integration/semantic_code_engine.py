#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Serena LSP-Based Semantic Code Engine
Integrates LSP (Language Server Protocol) for symbol-level code understanding

Based on: https://github.com/oraios/serena
Approach: Symbol-level code manipulation with IDE-parity tools
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlparse, unquote
from difflib import SequenceMatcher
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Serena: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class SymbolLocation:
    """Location of a symbol in the codebase."""
    file_path: str
    line: int
    column: int
    symbol_name: str
    symbol_type: str  # function, class, variable, method, etc.
    symbol_info: Dict[str, Any]
    language: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceLocation:
    """Location where a symbol is referenced."""
    file_path: str
    line: int
    column: int
    context: str  # surrounding code context
    reference_type: str  # usage, definition, declaration, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticMatch:
    """Result of semantic search."""
    file_path: str
    line: int
    symbol_name: str
    symbol_type: str
    relevance_score: float
    description: str
    code_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EditResult:
    """Result of code edit operation."""
    success: bool
    file_path: str
    lines_modified: List[int]
    original_content: str
    new_content: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Language server base
# --------------------------------------------------------------------------- #

class LanguageServer(ABC):
    """Abstract base class for language server implementations."""

    def __init__(self, workspace_path: str, language: str = "unknown") -> None:
        self.workspace_path = str(Path(workspace_path).resolve())
        self.language = language
        self.initialized: bool = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the language server."""
        ...

    @abstractmethod
    async def find_symbol(
        self,
        name: str,
        symbol_type: Optional[str] = None,
    ) -> List[SymbolLocation]:
        """Find symbol locations."""
        ...

    @abstractmethod
    async def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> List[ReferenceLocation]:
        """Find all references to a symbol."""
        ...

    @abstractmethod
    async def get_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[SymbolLocation]:
        """Get symbol definition."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the language server."""
        ...


# --------------------------------------------------------------------------- #
# Python (Pyright) language server
# --------------------------------------------------------------------------- #

class PythonLanguageServer(LanguageServer):
    """
    Python language server (Pyright).

    Phase 2: Real LSP Integration
    - Communicates with Pyright via JSON-RPC LSP protocol
    - Falls back to AST parsing if LSP unavailable
    """

    def __init__(self, workspace_path: str) -> None:
        super().__init__(workspace_path, language="python")
        self.lsp_client = None
        self.use_lsp: bool = False

    # -------------------------- helpers ---------------------------------- #

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        """Convert file:// URI to filesystem path."""
        if not uri:
            return ""
        parsed = urlparse(uri)
        return unquote(parsed.path)

    # -------------------------- lifecycle -------------------------------- #

    async def initialize(self) -> bool:
        """Initialize Pyright LSP server (if available) and AST fallback."""
        try:
            # Try to start LSP client
            from lsp_client import LSPClient  # type: ignore

            self.lsp_client = LSPClient(
                workspace_root=self.workspace_path,
                server_command=["pyright-langserver", "--stdio"],
            )

            if await self.lsp_client.start():
                self.use_lsp = True
                self.initialized = True
                logger.info("✅ Pyright LSP server connected (Phase 2: Real LSP)")
                return True

            logger.warning("⚠️  Pyright LSP failed - falling back to AST")
            self.use_lsp = False
            self.initialized = True
            return True

        except Exception as e:
            logger.warning("LSP initialization failed: %s - using AST fallback", e)
            self.use_lsp = False
            self.initialized = True
            return True

    # ------------------------ symbol search ------------------------------ #

    async def find_symbol(
        self,
        name: str,
        symbol_type: Optional[str] = None,
    ) -> List[SymbolLocation]:
        """
        Find Python symbols using LSP workspace/symbol (preferred) or AST fallback.

        Phase 2: Uses real Pyright LSP for cross-file symbol finding with type info.
        """
        # LSP path
        if self.use_lsp and self.lsp_client:
            try:
                lsp_symbols = await self.lsp_client.workspace_symbol(name)

                if lsp_symbols:
                    results: List[SymbolLocation] = []

                    # LSP symbol kinds: 1=File, 2=Module, 3=Namespace, 4=Package, 5=Class,
                    # 6=Method, 7=Property, 8=Field, 9=Constructor, 10=Enum, 11=Interface,
                    # 12=Function, 13=Variable, 14=Constant, etc.
                    kind_map = {
                        5: "class",
                        6: "method",
                        12: "function",
                        13: "variable",
                        14: "constant",
                        2: "module",
                    }

                    for sym in lsp_symbols:
                        sym_kind = sym.get("kind", 0)
                        sym_type_str = kind_map.get(sym_kind, "unknown")

                        # Filter by symbol type if specified
                        if symbol_type and symbol_type != "any" and sym_type_str != symbol_type:
                            continue

                        location = sym.get("location", {})
                        uri = location.get("uri", "")
                        file_path = self._uri_to_path(uri)

                        range_info = location.get("range", {})
                        start = range_info.get("start", {})

                        results.append(
                            SymbolLocation(
                                file_path=file_path,
                                line=start.get("line", 0) + 1,  # LSP is 0-indexed
                                column=start.get("character", 0),
                                symbol_name=sym.get("name", name),
                                symbol_type=sym_type_str,
                                symbol_info={
                                    "kind": sym_kind,
                                    "container": sym.get("containerName", ""),
                                    "source": "lsp",
                                },
                                language=self.language,
                            )
                        )

                    logger.debug("LSP found %d symbols for '%s'", len(results), name)
                    return results

            except Exception as e:
                logger.warning("LSP symbol search failed: %s - falling back to AST", e)

        # AST fallback
        return await self._find_symbol_ast(name, symbol_type)

    async def _find_symbol_ast(
        self,
        name: str,
        symbol_type: Optional[str] = None,
    ) -> List[SymbolLocation]:
        """AST-based symbol finding (fallback)."""
        import ast
        import glob

        results: List[SymbolLocation] = []

        python_files = glob.glob(
            os.path.join(self.workspace_path, "**/*.py"), recursive=True
        )

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source, filename=file_path)

                for node in ast.walk(tree):
                    node_name = getattr(node, "name", None)
                    if node_name != name:
                        continue

                    if isinstance(node, ast.FunctionDef):
                        sym_type = "function"
                    elif isinstance(node, ast.AsyncFunctionDef):
                        sym_type = "function"
                    elif isinstance(node, ast.ClassDef):
                        sym_type = "class"
                    elif isinstance(node, ast.Name):
                        sym_type = "variable"
                    else:
                        sym_type = "unknown"

                    if symbol_type and symbol_type != "any" and sym_type != symbol_type:
                        continue

                    symbol_info: Dict[str, Any] = {
                        "name": node_name,
                        "type": sym_type,
                        "source": "ast",
                    }

                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbol_info["args"] = [arg.arg for arg in node.args.args]
                        symbol_info["returns"] = (
                            ast.unparse(node.returns) if getattr(node, "returns", None) else None
                        )
                    elif isinstance(node, ast.ClassDef):
                        symbol_info["bases"] = [ast.unparse(base) for base in node.bases]

                    results.append(
                        SymbolLocation(
                            file_path=file_path,
                            line=getattr(node, "lineno", 0),
                            column=getattr(node, "col_offset", 0),
                            symbol_name=node_name,
                            symbol_type=sym_type,
                            symbol_info=symbol_info,
                            language=self.language,
                        )
                    )

            except Exception as e:
                logger.debug("Error parsing %s: %s", file_path, e)
                continue

        return results

    # ------------------------ reference search ---------------------------- #

    async def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> List[ReferenceLocation]:
        """
        Find references using LSP textDocument/references (preferred) or AST fallback.

        Phase 2: Real LSP provides cross-file reference tracking with type awareness.
        """
        # LSP path
        if self.use_lsp and self.lsp_client:
            try:
                lsp_refs = await self.lsp_client.text_document_references(
                    file_path, line, column, include_declaration=True
                )

                if lsp_refs:
                    results: List[ReferenceLocation] = []

                    for ref in lsp_refs:
                        uri = ref.get("uri", "")
                        ref_file = self._uri_to_path(uri)

                        range_info = ref.get("range", {})
                        start = range_info.get("start", {})
                        ref_line = start.get("line", 0) + 1
                        ref_col = start.get("character", 0)

                        # Read context
                        context = ""
                        try:
                            with open(ref_file, "r", encoding="utf-8") as f:
                                lines = f.readlines()

                            # line numbers are 1-based
                            start_line = max(1, ref_line - 3)
                            end_line = min(len(lines), ref_line + 3)
                            context = "".join(lines[start_line - 1:end_line])
                        except Exception:
                            pass

                        results.append(
                            ReferenceLocation(
                                file_path=ref_file,
                                line=ref_line,
                                column=ref_col,
                                context=context,
                                reference_type="lsp_reference",
                            )
                        )

                    logger.debug("LSP found %d references", len(results))
                    return results

            except Exception as e:
                logger.warning("LSP references failed: %s - falling back to AST", e)

        # AST fallback
        return await self._find_references_ast(file_path, line, column)

    async def _find_references_ast(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> List[ReferenceLocation]:
        """AST-based reference finding (fallback)."""
        import ast
        import glob

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return []

        tree = ast.parse(source, filename=file_path)
        target_symbol: Optional[str] = None

        for node in ast.walk(tree):
            if hasattr(node, "lineno") and node.lineno == line and hasattr(node, "name"):
                target_symbol = node.name  # type: ignore[attr-defined]
                break

        if not target_symbol:
            return []

        results: List[ReferenceLocation] = []
        python_files = glob.glob(
            os.path.join(self.workspace_path, "**/*.py"), recursive=True
        )

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    file_source = f.read()
                    lines = file_source.splitlines()

                file_tree = ast.parse(file_source, filename=py_file)

                for node in ast.walk(file_tree):
                    if isinstance(node, ast.Name) and node.id == target_symbol:
                        line_num = node.lineno
                        col_num = node.col_offset

                        start_line = max(1, line_num - 3)
                        end_line = min(len(lines), line_num + 3)
                        context = "\n".join(lines[start_line - 1:end_line])

                        if isinstance(node.ctx, ast.Store):
                            ref_type = "definition"
                        elif isinstance(node.ctx, ast.Load):
                            ref_type = "usage"
                        else:
                            ref_type = "unknown"

                        results.append(
                            ReferenceLocation(
                                file_path=py_file,
                                line=line_num,
                                column=col_num,
                                context=context,
                                reference_type=ref_type,
                            )
                        )

            except Exception as e:
                logger.debug("Error finding references in %s: %s", py_file, e)
                continue

        return results

    # ------------------------ definitions -------------------------------- #

    async def get_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[SymbolLocation]:
        """
        Get definition using LSP textDocument/definition (preferred) or AST fallback.

        Phase 2: Real LSP provides accurate cross-file goto definition.
        """
        # LSP path
        if self.use_lsp and self.lsp_client:
            try:
                lsp_defs = await self.lsp_client.text_document_definition(
                    file_path, line, column
                )

                if lsp_defs:
                    definition = lsp_defs[0]

                    uri = definition.get("uri", "")
                    def_file = self._uri_to_path(uri)

                    range_info = definition.get("range", {})
                    start = range_info.get("start", {})

                    def_line = start.get("line", 0) + 1
                    def_col = start.get("character", 0)

                    symbol_name = "unknown"

                    # Try to enrich name via hover
                    try:
                        hover = await self.lsp_client.text_document_hover(
                            def_file, def_line, def_col
                        )
                        if hover:
                            contents = hover.get("contents", {})
                            if isinstance(contents, dict):
                                value = contents.get("value", "")
                            elif isinstance(contents, str):
                                value = contents
                            else:
                                value = ""

                            if value:
                                first_line = value.split("\n", 1)[0]
                                symbol_name = first_line.split("(", 1)[0].strip("` ")
                    except Exception:
                        pass

                    return SymbolLocation(
                        file_path=def_file,
                        line=def_line,
                        column=def_col,
                        symbol_name=symbol_name,
                        symbol_type="lsp_symbol",
                        symbol_info={"source": "lsp"},
                        language=self.language,
                    )

            except Exception as e:
                logger.warning("LSP definition failed: %s - falling back to AST", e)

        # AST fallback
        return await self._get_definition_ast(file_path, line, column)

    async def _get_definition_ast(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[SymbolLocation]:
        """AST-based definition lookup (fallback)."""
        import ast

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return None

        tree = ast.parse(source, filename=file_path)

        for node in ast.walk(tree):
            if hasattr(node, "lineno") and node.lineno == line and hasattr(node, "name"):
                node_name = node.name  # type: ignore[attr-defined]
                sym_type = type(node).__name__.lower()
                return SymbolLocation(
                    file_path=file_path,
                    line=line,
                    column=column,
                    symbol_name=node_name,
                    symbol_type=sym_type,
                    symbol_info={"name": node_name, "type": type(node).__name__, "source": "ast"},
                    language=self.language,
                )

        return None

    # ------------------------ shutdown ----------------------------------- #

    async def shutdown(self) -> None:
        """Shutdown Python LSP."""
        if self.lsp_client:
            await self.lsp_client.shutdown()
            self.lsp_client = None

        self.use_lsp = False
        self.initialized = False


# --------------------------------------------------------------------------- #
# Semantic Code Engine
# --------------------------------------------------------------------------- #

class SemanticCodeEngine:
    """
    Main semantic code engine integrating multiple language servers.

    Inspired by Serena's approach to symbol-level code understanding.
    """

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = str(Path(workspace_path).resolve())
        self.language_servers: Dict[str, LanguageServer] = {}
        self.initialized: bool = False

    # async context manager, for `async with` usage
    async def __aenter__(self) -> "SemanticCodeEngine":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def initialize(self) -> None:
        """Initialize all language servers."""
        if self.initialized:
            return

        if not Path(self.workspace_path).is_dir():
            logger.warning("Workspace path does not exist: %s", self.workspace_path)

        logger.info("Initializing Semantic Code Engine...")

        # Initialize Python LSP
        python_lsp = PythonLanguageServer(self.workspace_path)
        if await python_lsp.initialize():
            self.language_servers["python"] = python_lsp
            logger.info("✅ Python LSP initialized")
        else:
            logger.warning("⚠️  Python LSP not available")

        # Additional language servers can be added here:
        # - rust_lsp = RustLanguageServer(self.workspace_path)
        # - typescript_lsp = TypeScriptLanguageServer(self.workspace_path)
        # - etc.

        self.initialized = True
        logger.info(
            "Semantic Code Engine initialized with %d language servers",
            len(self.language_servers),
        )

    # ------------------------ symbol ops --------------------------------- #

    async def find_symbol(
        self,
        name: str,
        symbol_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[SymbolLocation]:
        """
        Find symbol locations using LSP semantic understanding.

        Args:
            name: Symbol name to find
            symbol_type: Type filter (function, class, variable, etc.)
            language: Language filter (python, rust, etc.)

        Returns:
            List of SymbolLocation objects.
        """
        if not self.initialized:
            await self.initialize()

        results: List[SymbolLocation] = []

        # Determine which language servers to query
        if language:
            servers = [self.language_servers.get(language)]
        else:
            servers = list(self.language_servers.values())

        for server in servers:
            if not server:
                continue
            try:
                symbols = await server.find_symbol(name, symbol_type)
                results.extend(symbols)
            except Exception as e:
                logger.error("Error finding symbol in %s: %s", server.language, e)

        logger.info("Found %d symbols matching '%s'", len(results), name)
        return results

    async def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> List[ReferenceLocation]:
        """
        Find all references to a symbol.

        Equivalent to IDE "Find All References".
        """
        if not self.initialized:
            await self.initialize()

        ext = Path(file_path).suffix
        language = self._detect_language(ext)

        server = self.language_servers.get(language)
        if not server:
            logger.warning("No language server for %s", language)
            return []

        try:
            references = await server.find_references(file_path, line, column)
            logger.info("Found %d references", len(references))
            return references
        except Exception as e:
            logger.error("Error finding references: %s", e)
            return []

    async def insert_after_symbol(
        self,
        symbol: str,
        code: str,
        language: str = "python",
        preserve_indentation: bool = True,
    ) -> EditResult:
        """
        Insert code immediately after a symbol definition.

        Uses LSP/AST to find exact insertion point.
        """
        symbols = await self.find_symbol(symbol, language=language)

        if not symbols:
            return EditResult(
                success=False,
                file_path="",
                lines_modified=[],
                original_content="",
                new_content="",
                message=f"Symbol '{symbol}' not found",
            )

        target = symbols[0]

        try:
            with open(target.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            original_content = "".join(lines)

            insertion_line = max(1, target.line)  # 1-based

            # Detect indentation
            if preserve_indentation and insertion_line <= len(lines):
                base_line = lines[insertion_line - 1]
                indent_size = len(base_line) - len(base_line.lstrip())
                indent_prefix = base_line[:indent_size]
                indented_code = "\n".join(
                    [(indent_prefix + line) if line.strip() else "" for line in code.splitlines()]
                )
            else:
                indented_code = code

            # Ensure trailing newline
            if not indented_code.endswith("\n"):
                indented_code = indented_code + "\n"

            # Insert code after symbol definition line
            lines.insert(insertion_line, indented_code)

            new_content = "".join(lines)

            with open(target.file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(
                success=True,
                file_path=target.file_path,
                lines_modified=[insertion_line],
                original_content=original_content,
                new_content=new_content,
                message=f"Code inserted after symbol '{symbol}'",
            )

        except Exception as e:
            logger.error("Error inserting code: %s", e)
            return EditResult(
                success=False,
                file_path=target.file_path,
                lines_modified=[],
                original_content="",
                new_content="",
                message=f"Error: {e}",
            )

    # -------------------- semantic search helpers ------------------------ #

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize identifiers, paths, and snippets into normalized word tokens."""
        if not text:
            return []

        # Normalize separators
        text = text.replace("_", " ").replace("/", " ").replace("\\", " ")

        # Split camelCase / PascalCase: "DeviceReconMode" -> "Device Recon Mode"
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

        # Keep only alphanumerics, split into tokens
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        return [t.lower() for t in tokens if t]

    def _jaccard_overlap(self, a: List[str], b: List[str]) -> float:
        """Jaccard similarity between two token lists."""
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = sa & sb
        if not inter:
            return 0.0
        union = sa | sb
        return len(inter) / len(union)

    def _semantic_relevance(
        self,
        symbol: SymbolLocation,
        snippet: str,
        query_tokens: List[str],
    ) -> float:
        """
        Compute a heuristic semantic relevance score for a symbol + snippet.

        Features:
        - Name token overlap with query
        - Path token overlap with query
        - Snippet token overlap with query
        - Fuzzy name similarity
        - Role weighting (classes / functions prioritized)
        """
        if not query_tokens:
            return 0.0

        # Token views
        name_tokens = self._tokenize(symbol.symbol_name)
        path_tokens = self._tokenize(Path(symbol.file_path).stem)
        snippet_tokens = self._tokenize(snippet)

        # Overlaps
        name_overlap = self._jaccard_overlap(name_tokens, query_tokens)
        path_overlap = self._jaccard_overlap(path_tokens, query_tokens)
        snippet_overlap = self._jaccard_overlap(snippet_tokens, query_tokens)

        # Fuzzy similarity on raw string name
        query_str = " ".join(query_tokens)
        fuzzy_name = SequenceMatcher(
            None,
            symbol.symbol_name.lower(),
            query_str.lower(),
        ).ratio()

        # Role weighting
        role_weights = {
            "class": 1.2,
            "function": 1.1,
            "method": 1.1,
            "variable": 0.7,
            "constant": 0.7,
        }
        role_weight = role_weights.get(symbol.symbol_type, 1.0)

        # Description / type field bonus
        desc_field = symbol.symbol_info.get("type", symbol.symbol_type)
        desc_tokens = self._tokenize(str(desc_field))
        desc_overlap = self._jaccard_overlap(desc_tokens, query_tokens)

        # Combine: tuned but cheap
        base_score = (
            0.45 * name_overlap +
            0.15 * path_overlap +
            0.20 * snippet_overlap +
            0.10 * fuzzy_name +
            0.10 * desc_overlap
        )

        return base_score * role_weight

    # ------------------------ semantic search ---------------------------- #

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        language: Optional[str] = None,
    ) -> List[SemanticMatch]:
        """
        Search codebase using a heuristic semantic ranking.

        Features:
        - Token-based overlap between query and:
          - symbol name
          - file name
          - surrounding snippet
          - symbol type/description
        - Fuzzy similarity of name vs query
        - Role weighting (classes/functions > variables)
        """
        if not self.initialized:
            await self.initialize()

        results: List[SemanticMatch] = []

        servers = [self.language_servers.get(language)] if language else list(
            self.language_servers.values()
        )

        query_tokens = self._tokenize(query)

        for server in servers:
            if not server:
                continue

            try:
                # Broad symbol search – language server decides how fuzzy it is
                symbols = await server.find_symbol(query)

                for symbol in symbols:
                    try:
                        with open(symbol.file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as e:
                        logger.debug(
                            "Failed reading %s for semantic search: %s",
                            symbol.file_path,
                            e,
                        )
                        continue

                    start_line = max(1, symbol.line - 2)
                    end_line = min(len(lines), symbol.line + 5)
                    snippet = "".join(lines[start_line - 1:end_line])

                    relevance = self._semantic_relevance(
                        symbol=symbol,
                        snippet=snippet,
                        query_tokens=query_tokens,
                    )

                    desc_field = symbol.symbol_info.get("type", symbol.symbol_type)

                    results.append(
                        SemanticMatch(
                            file_path=symbol.file_path,
                            line=symbol.line,
                            symbol_name=symbol.symbol_name,
                            symbol_type=symbol.symbol_type,
                            relevance_score=relevance,
                            description=str(desc_field),
                            code_snippet=snippet,
                        )
                    )

            except Exception as e:
                logger.error("Error in semantic search for %s: %s", server.language, e)

        # Sort by semantic relevance (highest first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Filter out near-zero scores (noise) but keep shape predictable
        filtered = [r for r in results if r.relevance_score > 0.01]

        return filtered[:max_results]

    # ------------------------ definition API ----------------------------- #

    async def get_symbol_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[SymbolLocation]:
        """Get definition of symbol at location."""
        if not self.initialized:
            await self.initialize()

        ext = Path(file_path).suffix
        language = self._detect_language(ext)

        server = self.language_servers.get(language)
        if not server:
            return None

        try:
            return await server.get_definition(file_path, line, column)
        except Exception as e:
            logger.error("Error getting definition: %s", e)
            return None

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        lang_map = {
            ".py": "python",
            ".rs": "rust",
            ".ts": "typescript",
            ".js": "javascript",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        return lang_map.get(extension, "unknown")

    async def shutdown(self) -> None:
        """Shutdown all language servers."""
        for server in self.language_servers.values():
            try:
                await server.shutdown()
            except Exception as e:
                logger.error("Error during shutdown of %s: %s", server.language, e)

        self.language_servers.clear()
        self.initialized = False
        logger.info("Semantic Code Engine shutdown complete")


# --------------------------------------------------------------------------- #
# Example usage and testing
# --------------------------------------------------------------------------- #

async def main() -> None:
    """Test semantic code engine."""
    workspace = "/home/user/LAT5150DRVMIL"

    async with SemanticCodeEngine(workspace) as engine:
        # Test 1: Find symbols
        print("\n=== Test 1: Finding symbols ===")
        symbols = await engine.find_symbol("NSADeviceReconnaissance", symbol_type="class")
        for symbol in symbols:
            print(f"Found: {symbol.symbol_name} at {symbol.file_path}:{symbol.line}")

        # Test 2: Semantic search
        print("\n=== Test 2: Semantic search ===")
        matches = await engine.semantic_search("reconnaissance", max_results=5)
        for match in matches:
            print(
                f"Match: {match.symbol_name} ({match.relevance_score:.2f}) in {match.file_path}"
            )


if __name__ == "__main__":
    asyncio.run(main())
