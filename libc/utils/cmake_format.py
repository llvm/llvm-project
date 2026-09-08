#!/usr/bin/env python3
"""
LLVM and Subproject CMake Formatter Utility

Token-stream formatter with schema-aware keyword handling, based on the
`cmake-language(7)` EBNF grammar.

Architecture Overview:
  1. Lexer (tokenize): Converts raw CMake text into a flat list of typed Token
     objects according to the cmake-language(7) EBNF. Handles bracket comments,
     bracket arguments, quoted arguments, whitespace, newlines, and parens.
  2. Schema Scanner (scan_dynamic_schemas): Walks the token stream to learn
     custom keyword schemas from cmake_parse_arguments() calls and
     set(*_ARGS ...) variable conventions, populating a FormatterContext.
  3. Formatter (format_cmake_content): Iterates the token stream line-by-line,
     applying indentation, keyword casing, comment buffering, and multi-line
     argument layout rules using the populated FormatterContext.

Minimum Python Version: 3.12 (uses pathlib.Path.walk(), added in 3.12).

Grammar (from cmake-language(7)):
  file                ::= file_element*
  file_element        ::= command_invocation line_ending | (bracket_comment|space)* line_ending
  line_ending         ::= line_comment? newline
  command_invocation  ::= space* identifier space* '(' arguments ')'
  arguments           ::= argument? separated_arguments*
  separated_arguments ::= separation+ argument? | separation* '(' arguments ')'
  argument            ::= bracket_argument | quoted_argument | unquoted_argument

Keyword Schema Classification:
  1. Options / Flags (0 values): e.g. `OUTPUT_STRIP_TRAILING_WHITESPACE`, `EXCLUDE_FROM_ALL`, `POST_BUILD`, `PARENT_SCOPE`, `FORCE`, `PARSE_ARGV`, `PARSE_ARGN`.
     - Option keywords consume 0 arguments and close immediately unless inside an active multi-value list keyword scope.
  2. Single-Value Keywords (1 value): e.g. `RESULT_VARIABLE`, `OUTPUT_VARIABLE`, `TARGET`, `WORKING_DIRECTORY`, `ALIAS`, `SUITE`, `CACHE`, `DEPFILE`.
     - After value is consumed, keyword closes and subsequent keywords/values align at top-level keyword indent (+2 spaces).
  3. Multi-Value List Keywords (1+ values): e.g. `SRCS`, `HDRS`, `DEPENDS`, `FULL_BUILD_DEPENDS`, `COMPILE_OPTIONS`, `LINK_LIBRARIES`, `LOADER_ARGS`, `ARGS`, `ENV`, `PROPERTIES`, `OBJECT`, `STATIC`, `SHARED`, `MODULE`.
     - Keywords on the command header line (`cmd(KEYWORD...`) do NOT grant extra nesting to child lines (+2 spaces relative to call base).
     - Keywords on their own separate line (`\n KEYWORD...`) indent child list items +4 spaces (+2 relative to keyword line).
  4. Dynamic Schema Learning:
     - Automatically parses `cmake_parse_arguments(...)` calls in `function(...)` / `macro(...)` AST blocks.
     - Automatically learns custom option, single-value, and multi-value argument lists from `set(...)` calls using standard naming conventions (`*_OPTION_ARGS`, `*_SINGLE_VALUE_ARGS`, `*_MULTI_VALUE_ARGS`).

Formatting Rules Enforced:
  1. Command Casing: Built-in language commands cased in lowercase (`add_library`, `set`, `if`); module commands (like `ExternalProject_Add`) and custom functions retain canonical/declared casing.
  2. Parenthesis Spacing: No space between command name and opening `(`. Collapses multiple spaces between arguments down to a single space.
  3. Quoted String Immutability: Quoted arguments (`"..."`) and bracket arguments (`[=[...]=]`) are single immutable AST tokens. Multi-line quoted strings are preserved 100% untouched.
  4. Empty Closures: `endif()`, `else()`, `endfunction()`, `endmacro()`, `endforeach()`, `endwhile()`. (Legacy CMake permitted repeating condition/block names in closing commands; modern CMake standardizes on empty parentheses).
  5. Schema-Aware Keyword Casing: Keywords in command schema upper-cased; positional args, function parameters & file paths untouched.
  6. Multi-line Argument Layout: Keywords and positional args indented +2 spaces relative to call base; multi-value list items indented +4 spaces; closing `)` at +0 spaces.
  7. Control Block Indentation: 2-space indentation inside `if`/`foreach`/`function`/`macro`.
  8. Comment Formatting: Line comments buffer and align with the indentation level of the code element immediately following them, unless separated by a blank line (standalone comments) or immediately preceding a closing parenthesis `)`.
  9. Cleanliness: Trailing whitespace stripped, single trailing newline for non-empty files; empty files preserved 0-byte.

Usage:
  cmake_format.py [options] <file|directory>...

Options:
  -i, --inplace, --fix  Format files in-place.
  -n, --dry-run         Check formatting without modifying files (Evaluation Mode).
  --diff                Output unified diffs for files that need formatting.
  -j, --jobs N          Number of parallel worker processes to use (default: 1).
  -h, --help            Show this help message.
"""

import sys
import os
import re
import argparse
import difflib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import copy
from typing import NamedTuple


class LexError(ValueError):
    """Raised by tokenize() when the input CMake source is malformed.

    Attributes:
        msg:  Human-readable description of the problem.
        line: 1-based line number where the unterminated token started.
        col:  1-based column number where the unterminated token started.
    """

    def __init__(self, msg: str, line: int, col: int) -> None:
        super().__init__(f"line {line}, col {col}: {msg}")
        self.msg = msg
        self.line = line
        self.col = col


# Control block commands
CONTROL_START_BLOCKS = {"if", "function", "macro", "foreach", "while"}
CONTROL_MIDDLE_BLOCKS = {"elseif", "else"}
EMPTY_CLOSE_BLOCKS = {
    "endif",
    "else",
    "endfunction",
    "endmacro",
    "endforeach",
    "endwhile",
}

# Target creation commands
TARGET_CREATION_COMMANDS = {"add_library", "add_executable", "add_custom_target"}

# Formatting Constants
INDENT_WIDTH = 2
INDENT_STR = " " * INDENT_WIDTH
# Indent levels relative to a command's base_indent:
#   +1 level = keyword line or positional arg line (+2 spaces)
#   +2 levels = multi-value list item under a keyword on its own line (+4 spaces)
KEYWORD_INDENT_LEVELS = 1
LIST_ITEM_INDENT_LEVELS = 2


@dataclass
class FormatterContext:
    """Holds learned schema state for formatting operations to ensure isolation and thread-safety."""

    learned_options: set[str] = field(default_factory=set)
    learned_one_value: set[str] = field(default_factory=lambda: {"ALIAS"})
    learned_multi_value: set[str] = field(default_factory=set)
    list_keywords: set[str] = field(
        default_factory=lambda: {
            "SRCS",
            "HDRS",
            "DEPENDS",
            "FULL_BUILD_DEPENDS",
            "COMPILE_OPTIONS",
            "LINK_LIBRARIES",
            "LINK_LIBS",
            "FLAGS",
            "SOURCES",
            "BYPRODUCTS",
            "BUILD_BYPRODUCTS",
            "CMAKE_ARGS",
            "CMAKE_CACHE_ARGS",
            "LOADER_ARGS",
            "ARGS",
            "ENV",
            "COMPILE_DEFINITIONS",
            "PROPERTIES",
            "OBJECT",
            "STATIC",
            "SHARED",
            "MODULE",
        }
    )
    dynamic_schemas: dict[str, "CommandSchema"] = field(default_factory=dict)
    # Per-context cache for get_schema_for_cmd() results. Keyed by lowercased
    # command name. Invalidated whenever the context is mutated by
    # scan_dynamic_schemas(). Excluded from __init__ so it doesn't appear in
    # the constructor or backward-compat aliases.
    _schema_cache: dict[str, "CommandSchema"] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def clone(self) -> "FormatterContext":
        """Returns a deep copy of this context.

        Uses copy.deepcopy() so that any new fields added to FormatterContext
        or CommandSchema are automatically included — no lockstep update needed.
        """
        return copy.deepcopy(self)


WORKSPACE_CONTEXT = FormatterContext()


def _init_worker(ctx: FormatterContext) -> None:
    """Initializer for ProcessPoolExecutor workers.

    Replaces WORKSPACE_CONTEXT in the worker process with the fully
    pre-scanned context from the main process. Using an initializer
    (rather than relying on fork memory inheritance) ensures the
    pre-scanned schemas are available regardless of the multiprocessing
    start method ('fork' on Linux, 'spawn' on macOS/Windows).
    """
    global WORKSPACE_CONTEXT
    WORKSPACE_CONTEXT = ctx


# Pre-compiled regular expressions for Lexer, Schema Scanner, and Formatter.

# Matches uppercase identifier tokens (A-Z, 0-9, _) — used to extract
# keyword names from set(*_ARGS ...) string values.
RE_IDENTIFIER_TOKENS = re.compile(r"[A-Z0-9_]+")
# Matches mixed-case identifiers (a-zA-Z, 0-9, _) — used to extract
# keyword names from cmake_parse_arguments() string arguments, which may
# use any casing.
RE_IDENTIFIER_WORDS = re.compile(r"[a-zA-Z0-9_]+")

RE_OPTION_ARGS = re.compile(
    r"set\s*\(\s*([A-Z0-9_]*(?:OPTION|OPTIONAL)_ARGS)\s+([^)]+)\)", re.IGNORECASE
)
RE_SINGLE_VALUE_ARGS = re.compile(
    r"set\s*\(\s*([A-Z0-9_]*(?:SINGLE|ONE)_VALUE_ARGS)\s+([^)]+)\)", re.IGNORECASE
)
RE_MULTI_VALUE_ARGS = re.compile(
    r"set\s*\(\s*([A-Z0-9_]*(?:MULTI_VALUE|LIST)_ARGS)\s+([^)]+)\)", re.IGNORECASE
)

RE_BRACKET_COMMENT_START = re.compile(r"#\[(=*)\[")
RE_BRACKET_ARG_START = re.compile(r"\[(=*)\[")

RE_COMMENT_HASH = re.compile(r"^(#+)(.*)$")

RE_ARG_FORWARDING_VAR = re.compile(r'^"?\$(?:\{|\()ARG[NV]\d*(?:\}|\))"?$')
RE_COMMAND_INVOCATION = re.compile(r"^(\s*)([a-zA-Z0-9_]+)(\s*)\((.*)$", re.DOTALL)
# Matches the CMake CACHE keyword as a whole word, used to exclude CACHE variable
# declarations from dynamic schema learning.
RE_CACHE = re.compile(r"\bCACHE\b")


@dataclass
class CommandSchema:
    """Represents option, single-value, and multi-value keyword argument schemas for a CMake command."""

    options: set[str] = field(default_factory=set)
    one_value: set[str] = field(default_factory=set)
    multi_value: set[str] = field(default_factory=set)
    explicit_keywords: set[str] | None = None
    all_keywords: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.options = set(self.options)
        self.one_value = set(self.one_value)
        self.multi_value = set(self.multi_value)
        self.all_keywords = self.options | self.one_value | self.multi_value
        if self.explicit_keywords is None:
            self.explicit_keywords = set(self.all_keywords)
        else:
            self.explicit_keywords = set(self.explicit_keywords)


_IF_OPTIONS = {
    "NOT",
    "AND",
    "OR",
    "COMMAND",
    "POLICY",
    "TARGET",
    "EXISTS",
    "IS_DIRECTORY",
    "IS_SYMLINK",
    "IS_ABSOLUTE",
    "MATCHES",
    "LESS",
    "GREATER",
    "EQUAL",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "STRLESS",
    "STRGREATER",
    "STREQUAL",
    "STRLESS_EQUAL",
    "STRGREATER_EQUAL",
    "VERSION_LESS",
    "VERSION_GREATER",
    "VERSION_EQUAL",
    "VERSION_LESS_EQUAL",
    "VERSION_GREATER_EQUAL",
    "IN_LIST",
    "DEFINED",
}

# Official Built-in CMake Command Schema Registry
BUILTIN_COMMAND_SCHEMAS = {
    "function": CommandSchema(),
    "macro": CommandSchema(),
    "foreach": CommandSchema(options={"IN", "LISTS", "ITEMS", "ZIP_LISTS"}),
    "while": CommandSchema(),
    "if": CommandSchema(options=_IF_OPTIONS),
    "elseif": CommandSchema(options=_IF_OPTIONS),
    "else": CommandSchema(),
    "endif": CommandSchema(),
    "endfunction": CommandSchema(),
    "endmacro": CommandSchema(),
    "endforeach": CommandSchema(),
    "endwhile": CommandSchema(),
    "cmake_parse_arguments": CommandSchema(options={"PARSE_ARGV", "PARSE_ARGN"}),
    "execute_process": CommandSchema(
        options={
            "OUTPUT_STRIP_TRAILING_WHITESPACE",
            "ERROR_STRIP_TRAILING_WHITESPACE",
            "OUTPUT_QUIET",
            "ERROR_QUIET",
            "ECHO_OUTPUT_VARIABLE",
            "ECHO_ERROR_VARIABLE",
        },
        one_value={
            "WORKING_DIRECTORY",
            "TIMEOUT",
            "RESULT_VARIABLE",
            "RESULTS_VARIABLE",
            "OUTPUT_VARIABLE",
            "ERROR_VARIABLE",
            "INPUT_FILE",
            "OUTPUT_FILE",
            "ERROR_FILE",
            "ENCODING",
            "COMMAND_ERROR_IS_FATAL",
        },
        multi_value={"COMMAND"},
    ),
    "try_compile": CommandSchema(
        options={"GLOBAL_ERROR", "NO_CACHE"},
        one_value={"OUTPUT_VARIABLE", "COPY_FILE", "COPY_FILE_ERROR"},
        multi_value={
            "SOURCES",
            "COMPILE_DEFINITIONS",
            "LINK_LIBRARIES",
            "LINK_OPTIONS",
            "CMAKE_FLAGS",
        },
    ),
    "add_custom_command": CommandSchema(
        options={
            "POST_BUILD",
            "PRE_BUILD",
            "PRE_LINK",
            "VERBATIM",
            "APPEND",
            "USES_TERMINAL",
            "COMMAND_EXPAND_LISTS",
        },
        one_value={
            "TARGET",
            "MAIN_DEPENDENCY",
            "WORKING_DIRECTORY",
            "COMMENT",
            "DEPFILE",
            "JOB_POOL",
            "JOB_SERVER_AWARE",
        },
        multi_value={"COMMAND", "OUTPUT", "BYPRODUCTS", "DEPENDS", "IMPLICIT_DEPENDS"},
    ),
    "add_custom_target": CommandSchema(
        options={"ALL", "VERBATIM", "USES_TERMINAL", "COMMAND_EXPAND_LISTS"},
        one_value={"WORKING_DIRECTORY", "COMMENT"},
        multi_value={"COMMAND", "DEPENDS", "BYPRODUCTS", "SOURCES"},
    ),
    "ExternalProject_Add": CommandSchema(
        options={"EXCLUDE_FROM_ALL"},
        one_value={
            "PREFIX",
            "SOURCE_DIR",
            "BINARY_DIR",
            "INSTALL_DIR",
            "DOWNLOAD_COMMAND",
            "CONFIGURE_COMMAND",
            "BUILD_COMMAND",
            "INSTALL_COMMAND",
        },
        multi_value={
            "BUILD_BYPRODUCTS",
            "CMAKE_ARGS",
            "CMAKE_CACHE_ARGS",
            "STEP_TARGETS",
            "INDEPENDENT_STEP_TARGETS",
            "DEPENDS",
        },
    ),
    "find_package": CommandSchema(
        options={"EXACT", "QUIET", "REQUIRED", "CONFIG", "NO_MODULE"},
        multi_value={"COMPONENTS", "OPTIONAL_COMPONENTS"},
    ),
    "add_library": CommandSchema(
        options={"EXCLUDE_FROM_ALL", "GLOBAL", "IMPORTED"},
        one_value={"ALIAS"},
        multi_value={
            "STATIC",
            "SHARED",
            "MODULE",
            "OBJECT",
            "PUBLIC",
            "PRIVATE",
            "INTERFACE",
            "SOURCES",
        },
    ),
    "add_executable": CommandSchema(
        options={"WIN32", "MACOSX_BUNDLE", "EXCLUDE_FROM_ALL", "GLOBAL", "IMPORTED"},
        one_value={"ALIAS"},
        multi_value={"SOURCES"},
    ),
    "target_link_libraries": CommandSchema(
        multi_value={"PUBLIC", "PRIVATE", "INTERFACE", "LINK_PRIVATE", "LINK_PUBLIC"}
    ),
    "target_include_directories": CommandSchema(
        options={"BEFORE", "SYSTEM"}, multi_value={"PUBLIC", "PRIVATE", "INTERFACE"}
    ),
    "target_compile_options": CommandSchema(
        options={"BEFORE"}, multi_value={"PUBLIC", "PRIVATE", "INTERFACE"}
    ),
    "set_target_properties": CommandSchema(multi_value={"PROPERTIES"}),
    "set_source_files_properties": CommandSchema(multi_value={"PROPERTIES"}),
    "set_directory_properties": CommandSchema(multi_value={"PROPERTIES"}),
    "set_property": CommandSchema(
        options={
            "GLOBAL",
            "DIRECTORY",
            "TARGET",
            "SOURCE",
            "INSTALL",
            "TEST",
            "CACHE",
            "INHERITED",
        },
        one_value={"PROPERTY"},
        multi_value={"APPEND", "APPEND_STRING"},
    ),
    "get_target_property": CommandSchema(),
    "get_property": CommandSchema(
        options={
            "GLOBAL",
            "DIRECTORY",
            "TARGET",
            "SOURCE",
            "INSTALL",
            "TEST",
            "CACHE",
            "SET",
            "DEFINED",
            "BRIEF_DOCS",
            "FULL_DOCS",
        },
        one_value={"PROPERTY"},
    ),
    "list": CommandSchema(
        options={
            "APPEND",
            "PREPEND",
            "POP_BACK",
            "POP_FRONT",
            "REMOVE_AT",
            "REMOVE_ITEM",
            "REMOVE_DUPLICATES",
            "TRANSFORM",
            "SORT",
            "REVERSE",
            "JOIN",
            "SUBLIST",
            "FILTER",
            "FIND",
            "GET",
            "LENGTH",
            "INSERT",
        }
    ),
    "set": CommandSchema(options={"PARENT_SCOPE", "FORCE"}, one_value={"CACHE"}),
}

CANONICAL_CMD_CASING = {}
for k in BUILTIN_COMMAND_SCHEMAS.keys():
    if k != k.lower():
        CANONICAL_CMD_CASING[k.lower()] = k


def get_schema_for_cmd(
    cmd_name: str, ctx: FormatterContext | None = None
) -> CommandSchema:
    """Resolves official built-in or dynamically learned argument schema for a given CMake command."""
    cmd_lower = cmd_name.lower()

    # Standard built-in CMake commands use exact official schemas
    if cmd_lower in BUILTIN_COMMAND_SCHEMAS:
        return BUILTIN_COMMAND_SCHEMAS[cmd_lower]

    context = ctx if ctx is not None else WORKSPACE_CONTEXT

    # Return cached schema if available, avoiding repeated set allocations.
    cached = context._schema_cache.get(cmd_lower)
    if cached is not None:
        return cached

    options = set(context.learned_options)
    one_value_args = set(context.learned_one_value)
    multi_value_args = set(context.list_keywords) | set(context.learned_multi_value)

    explicit_kws = None
    if cmd_lower in context.dynamic_schemas:
        ds = context.dynamic_schemas[cmd_lower]
        options.update(ds.options)
        one_value_args.update(ds.one_value)
        multi_value_args.update(ds.multi_value)
        explicit_kws = set(ds.explicit_keywords)

    # Resolve overlaps so `multi_value` and `one_value` take precedence over `options`.
    options -= multi_value_args | one_value_args
    one_value_args -= multi_value_args

    schema = CommandSchema(
        options=options,
        one_value=one_value_args,
        multi_value=multi_value_args,
        explicit_keywords=explicit_kws if explicit_kws is not None else set(),
    )
    context._schema_cache[cmd_lower] = schema
    return schema


class TokenType(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    QUOTED_ARG = "QUOTED_ARG"
    BRACKET_ARG = "BRACKET_ARG"
    UNQUOTED_ARG = "UNQUOTED_ARG"
    LINE_COMMENT = "LINE_COMMENT"
    BRACKET_COMMENT = "BRACKET_COMMENT"
    WHITESPACE = "WHITESPACE"
    NEWLINE = "NEWLINE"


class Token(NamedTuple):
    """Represents a single AST token with its TokenType and string payload."""

    type: TokenType
    value: str


class KeywordType(str, Enum):
    ONE_VALUE = "ONE_VALUE"
    MULTI_VALUE = "MULTI_VALUE"
    OPTION = "OPTION"


def _lex_bracket_span(
    text: str,
    pos: int,
    eq_len: int,
    line: int,
    col: int,
    line_start: int,
    token_type: TokenType,
    desc: str,
) -> tuple[Token, int, int, int]:
    """Extracts a bracket comment or bracket argument, updating line/col tracking."""
    close_pat = "]" + "=" * eq_len + "]"
    end_idx = text.find(close_pat, pos)
    if end_idx == -1:
        prefix = "#" if token_type == TokenType.BRACKET_COMMENT else ""
        raise LexError(
            f"unterminated {desc} '{prefix}[{'=' * eq_len}['",
            line,
            col,
        )
    end_pos = end_idx + len(close_pat)
    span = text[pos:end_pos]
    newlines = span.count("\n")
    if newlines:
        line += newlines
        line_start = end_pos - len(span.rsplit("\n", 1)[-1])
    return Token(token_type, span), end_pos, line, line_start


def tokenize(text: str) -> list[Token]:
    """Formal Lexer based on cmake-language(7) EBNF specification.

    Raises:
        LexError: If the input contains an unterminated quoted argument,
            bracket argument, or bracket comment.
    """
    tokens = []
    i = 0
    n = len(text)
    # Track line/col for error reporting.  line and col are 1-based.
    line = 1
    line_start = 0

    while i < n:
        col = i - line_start + 1

        # Bracket comment or line comment starting with '#'
        if text[i] == "#":
            m = RE_BRACKET_COMMENT_START.match(text, pos=i)
            if m:
                tok, i, line, line_start = _lex_bracket_span(
                    text,
                    i,
                    len(m.group(1)),
                    line,
                    col,
                    line_start,
                    TokenType.BRACKET_COMMENT,
                    "bracket comment",
                )
                tokens.append(tok)
                continue

            # Line comment: #...
            end_idx = text.find("\n", i)
            if end_idx == -1:
                end_idx = n
            tokens.append(Token(TokenType.LINE_COMMENT, text[i:end_idx]))
            i = end_idx
            continue

        # Bracket argument: [=[...]=]
        if text[i] == "[":
            m = RE_BRACKET_ARG_START.match(text, pos=i)
            if m:
                tok, i, line, line_start = _lex_bracket_span(
                    text,
                    i,
                    len(m.group(1)),
                    line,
                    col,
                    line_start,
                    TokenType.BRACKET_ARG,
                    "bracket argument",
                )
                tokens.append(tok)
                continue

        # Quoted argument: "..." (Single immutable token!)
        if text[i] == '"':
            j = i + 1
            closed = False
            while j < n:
                if text[j] == "\\":
                    j = min(n, j + 2)
                elif text[j] == '"':
                    j += 1
                    closed = True
                    break
                else:
                    j += 1
            if not closed:
                raise LexError("unterminated quoted argument", line, col)
            token_val = text[i:j]
            tokens.append(Token(TokenType.QUOTED_ARG, token_val))
            newlines = token_val.count("\n")
            if newlines:
                line += newlines
                line_start = j - len(token_val.rsplit("\n", 1)[-1])
            i = j
            continue

        # Newline
        if text[i] == "\n":
            tokens.append(Token(TokenType.NEWLINE, "\n"))
            i += 1
            line += 1
            line_start = i
            continue

        # Whitespace
        if text[i] in " \t\r":
            j = i
            while j < n and text[j] in " \t\r":
                j += 1
            tokens.append(Token(TokenType.WHITESPACE, text[i:j]))
            i = j
            continue

        # Parens
        if text[i] == "(":
            tokens.append(Token(TokenType.LPAREN, "("))
            i += 1
            continue
        if text[i] == ")":
            tokens.append(Token(TokenType.RPAREN, ")"))
            i += 1
            continue

        # Unquoted argument / Identifier
        j = i
        while j < n and text[j] not in ' \t\r\n()#"':
            if text[j] == "\\":
                j = min(n, j + 2)
            else:
                j += 1
        token_val = text[i:j]
        tokens.append(Token(TokenType.UNQUOTED_ARG, token_val))
        i = j

    return tokens


def _next_significant_token(
    tokens: list[Token], start_idx: int
) -> tuple[Token, int] | tuple[None, -1]:
    """Finds the next non-whitespace, non-newline, non-comment token from start_idx."""
    n = len(tokens)
    for idx in range(start_idx, n):
        tok = tokens[idx]
        if tok.type not in (
            TokenType.WHITESPACE,
            TokenType.NEWLINE,
            TokenType.LINE_COMMENT,
            TokenType.BRACKET_COMMENT,
        ):
            return tok, idx
    return None, -1


def scan_dynamic_schemas(
    content: str,
    ctx: FormatterContext | None = None,
    tokens: list[Token] | None = None,
) -> None:
    """Scans cmake_parse_arguments and set(..._ARGS) calls to learn keyword schemas dynamically.

    Mutates ctx in-place: appends newly discovered keywords to ctx.learned_options,
    ctx.learned_one_value, ctx.learned_multi_value, ctx.list_keywords, and
    ctx.dynamic_schemas. If ctx is None, mutates the module-level WORKSPACE_CONTEXT.
    """
    context = ctx if ctx is not None else WORKSPACE_CONTEXT
    context._schema_cache.clear()

    # 1. Parse set(*_OPTION_ARGS ...), set(*_SINGLE_VALUE_ARGS ...), set(*_MULTI_VALUE_ARGS ...), etc.
    schema_patterns = [
        (RE_OPTION_ARGS, context.learned_options),
        (RE_SINGLE_VALUE_ARGS, context.learned_one_value),
        (RE_MULTI_VALUE_ARGS, context.learned_multi_value),
    ]
    for regex, target_set in schema_patterns:
        for m in regex.finditer(content):
            if not RE_CACHE.search(m.group(2)):
                words = set(RE_IDENTIFIER_TOKENS.findall(m.group(2)))
                target_set.update(words)
                if target_set is context.learned_multi_value:
                    context.list_keywords.update(words)

    # 2. Token-based scanning for function()/macro() declarations and cmake_parse_arguments
    if tokens is None:
        tokens = tokenize(content)
    n = len(tokens)
    current_fn_name = None

    for i in range(n):
        tok = tokens[i]
        if tok.type != TokenType.UNQUOTED_ARG:
            continue

        cmd_lower = tok.value.lower()

        # Track function/macro start
        if cmd_lower in ("function", "macro"):
            tok_lparen, j = _next_significant_token(tokens, i + 1)
            if tok_lparen and tok_lparen.type == TokenType.LPAREN:
                tok_name, _ = _next_significant_token(tokens, j + 1)
                if tok_name and tok_name.type in (
                    TokenType.UNQUOTED_ARG,
                    TokenType.QUOTED_ARG,
                ):
                    current_fn_name = tok_name.value.strip('"')

        # Track function/macro end
        elif cmd_lower in ("endfunction", "endmacro"):
            current_fn_name = None

        # Parse cmake_parse_arguments(...) call inside function/macro
        elif cmd_lower == "cmake_parse_arguments" and current_fn_name:
            tok_lparen, lparen_idx = _next_significant_token(tokens, i + 1)
            if tok_lparen and tok_lparen.type == TokenType.LPAREN:
                arg_tokens = []
                paren_depth = 1
                for j in range(lparen_idx + 1, n):
                    tok_j = tokens[j]
                    if tok_j.type in (
                        TokenType.WHITESPACE,
                        TokenType.NEWLINE,
                        TokenType.LINE_COMMENT,
                        TokenType.BRACKET_COMMENT,
                    ):
                        continue
                    if tok_j.type == TokenType.LPAREN:
                        paren_depth += 1
                    elif tok_j.type == TokenType.RPAREN:
                        paren_depth -= 1
                        if paren_depth <= 0:
                            break
                    else:
                        arg_tokens.append(tok_j.value)

                # cmake_parse_arguments positional layout:
                #   cmake_parse_arguments(<prefix> <options> <one_value> <multi_value> <args>...)
                #   cmake_parse_arguments(PARSE_ARGV <n> <prefix> <options> <one_value> <multi_value>)
                # When PARSE_ARGV/PARSE_ARGN is present, the first two tokens are the
                # mode keyword and the index argument; the prefix/options/... follow at +2.
                _CPA_MIN_ARGS = 4  # prefix + options + one_value + multi_value
                _CPA_PARSE_ARGV_EXTRA = 2  # extra tokens: mode keyword + integer index
                _CPA_PARSE_ARGV_MIN_ARGS = _CPA_MIN_ARGS + _CPA_PARSE_ARGV_EXTRA

                if len(arg_tokens) >= _CPA_MIN_ARGS:
                    offset = 0
                    if (
                        arg_tokens[0].upper() in ("PARSE_ARGV", "PARSE_ARGN")
                        and len(arg_tokens) >= _CPA_PARSE_ARGV_MIN_ARGS
                    ):
                        offset = _CPA_PARSE_ARGV_EXTRA

                    if len(arg_tokens) >= offset + _CPA_MIN_ARGS:
                        # arg_tokens[offset + 0] is the prefix — not needed for schema building
                        opt_str = arg_tokens[offset + 1]
                        one_str = arg_tokens[offset + 2]
                        multi_str = arg_tokens[offset + 3]

                        def _extract_literal_kws(arg_s: str) -> set[str]:
                            cleaned = arg_s.strip('"\t\r\n ')
                            if cleaned.startswith("$"):
                                return set()
                            non_var = re.sub(r"\$\{[^}]*\}", "", cleaned)
                            return set(RE_IDENTIFIER_WORDS.findall(non_var))

                        opts = _extract_literal_kws(opt_str)
                        ones = _extract_literal_kws(one_str)
                        multis = _extract_literal_kws(multi_str)
                        context.list_keywords.update(multis)
                        fn_key = current_fn_name.lower()
                        schema = CommandSchema(
                            options=opts, one_value=ones, multi_value=multis
                        )
                        context.dynamic_schemas[fn_key] = schema


# Filesystem markers that indicate a project root boundary. Used by pre_scan_workspace_modules
# to stop walking upward before escaping the repository tree.
_PROJECT_ROOT_SENTINELS = frozenset({".git", ".gitmodules"})


def pre_scan_workspace_modules(
    paths: list[str], ctx: FormatterContext | None = None
) -> None:
    """Pre-scans all CMake module files in the repository directory tree to learn dynamic schemas before formatting."""
    context = ctx if ctx is not None else WORKSPACE_CONTEXT
    module_files = set()
    scanned_base_dirs = set()

    for p in paths:
        path = Path(p)
        base_dir = path.resolve() if path.is_dir() else path.resolve().parent
        if base_dir in scanned_base_dirs:
            continue
        scanned_base_dirs.add(base_dir)

        # Walk upward to find cmake/modules/, stopping at recognized repository
        # root boundaries (.git, .gitmodules) so we don't accidentally scan
        # unrelated cmake/modules/ directories in parent directories.
        curr = base_dir
        while curr and curr != curr.parent:
            mod_dir = curr / "cmake" / "modules"
            if mod_dir.is_dir():
                module_files.update(str(f) for f in mod_dir.glob("*.cmake"))
                break
            # Stop after checking current dir if we've reached a repository root boundary
            if any((curr / sentinel).exists() for sentinel in _PROJECT_ROOT_SENTINELS):
                break
            curr = curr.parent

    for mf in module_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                scan_dynamic_schemas(f.read(), ctx=context)
        except (OSError, UnicodeDecodeError, LexError) as e:
            print(f"Error pre-scanning module {mf}: {e}", file=sys.stderr)
            sys.exit(1)


# Regex matching directive comments that should not have a space inserted after '#'
# (e.g., native CMake template directives like #cmakedefine / #cmakedefine01, tool directives
# like # cmake-lint / # cmake-format / #clang-format / #nolint).
#
# NOTE: C preprocessor directives (#include, #define, #pragma, #ifdef, etc.) are included
# here to support embedded C code snippets and C header template code written inside
# CMake files and CMake comments, ensuring that formatting does not corrupt the embedded C.
RE_DIRECTIVE_COMMENT = re.compile(
    r"^#(?:[ \t]*cmake-lint|[ \t]*cmake-format|cmakedefine01|cmakedefine|nolint|pragma|include|clang-format|define|undef|ifdef|ifndef|error|warning)\b",
    re.IGNORECASE,
)


def _format_line_comment(comment_text: str) -> str:
    """Formats a single line comment, ensuring a space after '#' while preserving special markers (#---, #===, #!).

    This function is only called for LINE_COMMENT tokens. Bracket-style comments
    (#[=[...]=]) are handled separately by the lexer as BRACKET_COMMENT tokens and
    are never passed here.
    """
    m_hash = RE_COMMENT_HASH.match(comment_text)
    if m_hash:
        hash_prefix, rest_comment = m_hash.groups()
        if not (
            comment_text.startswith("#---")
            or comment_text.startswith("#===")
            or comment_text.startswith("#!")
            or RE_DIRECTIVE_COMMENT.match(comment_text)
        ):
            if rest_comment and not (
                rest_comment.startswith(" ") or rest_comment.startswith("\t")
            ):
                return f"{hash_prefix} {rest_comment}"
    return comment_text


@dataclass
class CallState:
    """Tracks formatter state across lines during a multi-line command invocation.

    Lifecycle:
        Created when the opening line of a multi-line command is encountered
        (i.e., the line's paren balance is still positive after the command).
        Destroyed when the closing ``)``) line is processed or paren balance
        drops to zero.

    Key state variables:
        cmd_name: The command being formatted (lowercase).
        base_indent: Indent level of the command itself (used to compute child indents).
        paren_balance: Running count of unmatched ``(`` tokens; zero signals end of call.
        schema: Resolved CommandSchema providing options/one_value/multi_value sets.
        active_kw: The currently open keyword (None if no keyword is active).
        active_kw_type: ONE_VALUE, MULTI_VALUE, or None — governs indent of child lines.
        active_kw_on_cmd_line: True if active_kw was opened on the command line itself,
            which suppresses extra nesting for its child list items.
        one_value_count: Number of values consumed under the current ONE_VALUE keyword;
            the keyword closes after the first value is seen.
    """

    cmd_name: str = ""
    base_indent: int = 0
    paren_balance: int = 0
    schema: CommandSchema | None = None
    arg_count: int = 0
    first_arg_on_separate_line: bool = False
    active_kw: str | None = None
    active_kw_type: KeywordType | None = None
    active_kw_on_cmd_line: bool = False
    one_value_count: int = 0
    last_line_was_keyword: bool = False
    in_call: bool = True

    def __post_init__(self) -> None:
        if self.schema is None:
            self.schema = CommandSchema()

    def reset_kw(self) -> None:
        self.active_kw = None
        self.active_kw_type = None
        self.active_kw_on_cmd_line = False

    def is_keyword_token(self, clean_word: str) -> bool:
        """Determines contextually whether an unquoted identifier token is a keyword or a positional argument."""
        if not clean_word.isidentifier():
            return False
        clean_upper = clean_word.upper()
        if clean_upper in self.schema.explicit_keywords:
            return True
        if clean_upper in self.schema.all_keywords:
            if self.arg_count > 0 or self.cmd_name in BUILTIN_COMMAND_SCHEMAS:
                return True
        return False

    def process_tokens(self, tokens: list[Token], is_cmd_line: bool = False) -> str:
        """Processes line tokens contextually, updating active keyword state and reconstructing formatted text."""
        reconstructed_parts = []
        for tok in tokens:
            if tok.type == TokenType.WHITESPACE:
                reconstructed_parts.append(" ")
            elif tok.type in (
                TokenType.QUOTED_ARG,
                TokenType.BRACKET_ARG,
            ) or self.cmd_name in {"add_subdirectory", "include"}:
                reconstructed_parts.append(tok.value)
                if RE_ARG_FORWARDING_VAR.match(tok.value):
                    self.reset_kw()
                elif self.active_kw_type == KeywordType.ONE_VALUE:
                    self.one_value_count += 1
                    if self.one_value_count >= 1:
                        self.reset_kw()
            elif tok.type == TokenType.UNQUOTED_ARG:
                # Strip trailing syntactic noise (`,`, `;`, `)`) and leading `(` that
                # CMake allows to attach to unquoted argument tokens.
                clean_word = tok.value.rstrip("),;").lstrip("(")
                if RE_ARG_FORWARDING_VAR.match(clean_word):
                    reconstructed_parts.append(tok.value)
                    self.reset_kw()
                elif self.is_keyword_token(clean_word):
                    kw = clean_word.upper()
                    formatted_tok = tok.value.replace(clean_word, kw, 1)
                    reconstructed_parts.append(formatted_tok)
                    if kw in self.schema.options:
                        if self.active_kw_type != KeywordType.MULTI_VALUE:
                            self.reset_kw()
                        self.one_value_count = 0
                    elif kw in self.schema.one_value:
                        self.active_kw = kw
                        self.active_kw_type = KeywordType.ONE_VALUE
                        self.active_kw_on_cmd_line = is_cmd_line
                        self.one_value_count = 0
                    elif kw in self.schema.multi_value:
                        self.active_kw = kw
                        self.active_kw_type = KeywordType.MULTI_VALUE
                        self.active_kw_on_cmd_line = is_cmd_line
                        self.one_value_count = 0
                else:
                    reconstructed_parts.append(tok.value)
                    if self.active_kw_type == KeywordType.ONE_VALUE:
                        self.one_value_count += 1
                        if self.one_value_count >= 1:
                            self.reset_kw()
            else:
                reconstructed_parts.append(tok.value)

        return "".join(reconstructed_parts)


@dataclass
class _FormatterState:
    """Mutable state carried through a single formatting pass over a CMake file.

    Encapsulates all the variables that must be shared across line iterations
    (output buffer, pending comments, indent level, active call tracker) so
    that the per-line helper functions can read and update them without relying
    on Python closure capture.
    """

    formatted_lines: list[str] = field(default_factory=list)
    pending_comments: list[str] = field(default_factory=list)
    block_indent_level: int = 0
    current_call_state: CallState | None = None

    def flush_comments(self, indent_str: str) -> None:
        """Appends all buffered comments at the given indent level, then clears the buffer."""
        for ctext in self.pending_comments:
            self.formatted_lines.append(f"{indent_str}{ctext}")
        self.pending_comments.clear()


def _trim_line_toks(line_toks: list[Token]) -> list[Token]:
    """Returns a slice of line_toks with leading and trailing WHITESPACE tokens removed."""
    start = next(
        (i for i, t in enumerate(line_toks) if t.type != TokenType.WHITESPACE),
        len(line_toks),
    )
    end = next(
        (
            i
            for i in range(len(line_toks) - 1, start - 1, -1)
            if line_toks[i].type != TokenType.WHITESPACE
        ),
        start - 1,
    )
    return line_toks[start : end + 1]


def _process_call_continuation_line(
    trimmed_line_toks: list[Token],
    state: _FormatterState,
) -> None:
    """Handles a continuation line inside an active multi-line command invocation (section 3).

    Updates paren_balance on the active CallState. If the line starts with ')' the
    call is closed. Otherwise the indent level is computed from the active keyword
    context, the line is formatted via CallState.process_tokens(), and the result
    is appended to state.formatted_lines.
    """
    cs = state.current_call_state
    if cs is None:
        raise RuntimeError(
            "_process_call_continuation_line called with no active CallState"
        )

    # Safe: the lexer only emits LPAREN/RPAREN for syntactic parens,
    # never inside quoted arguments, bracket arguments, or comments.
    lparens = sum(1 for tok in trimmed_line_toks if tok.type == TokenType.LPAREN)
    rparens = sum(1 for tok in trimmed_line_toks if tok.type == TokenType.RPAREN)

    # Check if this line is a standalone closing parenthesis line (e.g. ')' or ') # comment').
    non_comment_toks = [
        tok
        for tok in trimmed_line_toks
        if tok.type not in (TokenType.WHITESPACE, TokenType.LINE_COMMENT)
    ]
    is_standalone_close_paren = (
        len(non_comment_toks) == 1 and non_comment_toks[0].type == TokenType.RPAREN
    )

    if is_standalone_close_paren and (cs.paren_balance - 1) <= 0:
        cs.in_call = False
        comment_indent = INDENT_STR * (
            cs.base_indent + LIST_ITEM_INDENT_LEVELS
            if cs.last_line_was_keyword
            else cs.base_indent + KEYWORD_INDENT_LEVELS
        )
        line_indent = INDENT_STR * cs.base_indent
        cs.reset_kw()
        state.flush_comments(comment_indent)
        reconstructed_close = "".join(tok.value for tok in trimmed_line_toks).rstrip()
        state.formatted_lines.append(f"{line_indent}{reconstructed_close}")
        state.current_call_state = None
        return

    cs.paren_balance += lparens - rparens

    # Determine whether the first meaningful token on this line is a keyword or
    # an argument-forwarding variable (${ARGN} etc.), as both affect indentation.
    first_unquoted: str | None = None
    for tok in trimmed_line_toks:
        if tok.type in (
            TokenType.UNQUOTED_ARG,
            TokenType.QUOTED_ARG,
            TokenType.BRACKET_ARG,
        ):
            first_unquoted = tok.value
            break

    line_starts_new_keyword = False
    is_arg_var_line = False
    if first_unquoted:
        clean_first = first_unquoted.rstrip("),;").lstrip("(")
        if RE_ARG_FORWARDING_VAR.match(clean_first):
            is_arg_var_line = True
        elif cs.is_keyword_token(clean_first):
            line_starts_new_keyword = True

    cs.last_line_was_keyword = line_starts_new_keyword

    # Compute indentation level for this continuation line.
    if line_starts_new_keyword or is_arg_var_line:
        indent_str = INDENT_STR * (cs.base_indent + KEYWORD_INDENT_LEVELS)
    elif cs.active_kw is not None:
        if cs.active_kw_on_cmd_line:
            indent_str = INDENT_STR * (cs.base_indent + KEYWORD_INDENT_LEVELS)
        else:
            indent_str = INDENT_STR * (cs.base_indent + LIST_ITEM_INDENT_LEVELS)
    elif cs.cmd_name == "set" and cs.arg_count >= 1 and cs.first_arg_on_separate_line:
        indent_str = INDENT_STR * (cs.base_indent + LIST_ITEM_INDENT_LEVELS)
    else:
        indent_str = INDENT_STR * (cs.base_indent + KEYWORD_INDENT_LEVELS)

    reconstructed_line = cs.process_tokens(trimmed_line_toks, is_cmd_line=False)
    cs.arg_count += sum(
        1
        for tok in trimmed_line_toks
        if tok.type
        in (TokenType.QUOTED_ARG, TokenType.BRACKET_ARG, TokenType.UNQUOTED_ARG)
    )

    state.flush_comments(indent_str)
    state.formatted_lines.append(f"{indent_str}{reconstructed_line}".rstrip())

    if cs.paren_balance <= 0:
        cs.in_call = False
        state.current_call_state = None


def _process_command_invocation_line(
    non_ws_toks: list[Token],
    trimmed_line_toks: list[Token],
    state: _FormatterState,
    file_ctx: FormatterContext,
) -> None:
    """Handles the opening line of a CMake command invocation (section 4).

    Applies command casing, strips arguments from empty-closure commands,
    computes the opening line's paren balance to decide if a CallState is
    needed for continuation lines, and adjusts block_indent_level for
    control-flow commands.
    """
    cmd_name = non_ws_toks[0].value
    lower_cmd = cmd_name.lower()

    # Apply canonical casing: module commands keep declared case, built-ins go lowercase.
    if lower_cmd in CANONICAL_CMD_CASING:
        cmd_name = CANONICAL_CMD_CASING[lower_cmd]
    elif (
        lower_cmd in BUILTIN_COMMAND_SCHEMAS
        or lower_cmd in EMPTY_CLOSE_BLOCKS
        or lower_cmd in CONTROL_START_BLOCKS
    ):
        cmd_name = lower_cmd

    # Find the opening '(' in the trimmed token list.
    lparen_idx = next(
        (i for i, tok in enumerate(trimmed_line_toks) if tok.type == TokenType.LPAREN),
        -1,
    )
    rest_tokens = trimmed_line_toks[lparen_idx + 1 :] if lparen_idx != -1 else []

    # Empty-closure commands (endif, else, ...) have their arguments stripped while preserving line comments.
    if lower_cmd in EMPTY_CLOSE_BLOCKS:
        rparen_seen = False
        filtered_tokens = []
        for rt in rest_tokens:
            if rparen_seen:
                filtered_tokens.append(rt)
            elif rt.type == TokenType.RPAREN:
                filtered_tokens.append(rt)
                rparen_seen = True
        rest_tokens = filtered_tokens

    lparens = sum(1 for tok in rest_tokens if tok.type == TokenType.LPAREN)
    rparens = sum(1 for tok in rest_tokens if tok.type == TokenType.RPAREN)
    line_paren_balance = 1 + lparens - rparens
    is_single_line = line_paren_balance <= 0

    # Compute indent for this line, adjusting block level for end/middle commands.
    if lower_cmd.startswith("end"):
        state.block_indent_level = max(0, state.block_indent_level - 1)
        indent_str = INDENT_STR * state.block_indent_level
    elif lower_cmd in CONTROL_MIDDLE_BLOCKS:
        indent_str = INDENT_STR * max(0, state.block_indent_level - 1)
    else:
        indent_str = INDENT_STR * state.block_indent_level

    cmd_schema = get_schema_for_cmd(lower_cmd, ctx=file_ctx)
    rest_arg_count = sum(
        1
        for tok in rest_tokens
        if tok.type
        in (TokenType.QUOTED_ARG, TokenType.BRACKET_ARG, TokenType.UNQUOTED_ARG)
    )
    first_arg_on_separate_line = rest_arg_count == 0

    temp_state = CallState(cmd_name=lower_cmd, schema=cmd_schema)
    rest_formatted = temp_state.process_tokens(rest_tokens, is_cmd_line=True)
    reconstructed = f"{indent_str}{cmd_name}({rest_formatted}".rstrip()

    state.flush_comments(indent_str)
    state.formatted_lines.append(reconstructed)

    if not is_single_line:
        state.current_call_state = CallState(
            cmd_name=lower_cmd,
            base_indent=state.block_indent_level,
            paren_balance=line_paren_balance,
            schema=cmd_schema,
            arg_count=rest_arg_count,
            first_arg_on_separate_line=first_arg_on_separate_line,
            active_kw=temp_state.active_kw,
            active_kw_type=temp_state.active_kw_type,
            active_kw_on_cmd_line=temp_state.active_kw_on_cmd_line,
            one_value_count=temp_state.one_value_count,
        )

    if lower_cmd in CONTROL_START_BLOCKS:
        state.block_indent_level += 1


def format_cmake_content(content: str, ctx: FormatterContext | None = None) -> str:
    """Formats CMake file content using AST token-stream parser.

    Applies lowercasing for built-in commands, proper indent nesting for control blocks,
    schema-aware keyword upper-casing, multi-line argument indentation (+2 / +4 spaces),
    and lookahead comment buffering alignment.

    The function dispatches each source line to one of five handlers:
      1. Empty line  — flush buffered comments and emit a blank line.
      2. Comment line — buffer the comment for lookahead alignment.
      3. Call continuation line — delegate to _process_call_continuation_line().
      4. Command invocation line — delegate to _process_command_invocation_line().
      5. Raw fallback — emit the line unchanged (bracket comments, bare tokens).
    """
    if not content.strip():
        return ""

    file_ctx = ctx.clone() if ctx is not None else WORKSPACE_CONTEXT.clone()
    tokens = tokenize(content)
    scan_dynamic_schemas(content, ctx=file_ctx, tokens=tokens)

    # Group tokens into lines (separated by TokenType.NEWLINE).
    lines_tokens: list[list[Token]] = []
    curr_line: list[Token] = []
    for tok in tokens:
        if tok.type == TokenType.NEWLINE:
            lines_tokens.append(curr_line)
            curr_line = []
        else:
            curr_line.append(tok)
    if curr_line or not lines_tokens:
        lines_tokens.append(curr_line)

    state = _FormatterState()

    for line_toks in lines_tokens:
        # Strip leading whitespace tokens for line classification.
        non_ws_toks = [t for t in line_toks if t.type != TokenType.WHITESPACE]
        in_call = (
            state.current_call_state is not None and state.current_call_state.in_call
        )

        # 1. Empty line — flush buffered comments and emit a blank line.
        if not non_ws_toks:
            if state.pending_comments:
                state.flush_comments(
                    INDENT_STR * (state.block_indent_level + (1 if in_call else 0))
                )
            state.formatted_lines.append("")
            if (
                state.current_call_state
                and state.current_call_state.in_call
                and state.current_call_state.paren_balance <= 0
            ):
                state.current_call_state.in_call = False
                state.current_call_state = None
            continue

        # 2. Comment line — buffer for lookahead alignment.
        if non_ws_toks[0].type == TokenType.LINE_COMMENT:
            state.pending_comments.append(_format_line_comment(non_ws_toks[0].value))
            continue

        trimmed_line_toks = _trim_line_toks(line_toks)

        # 3. Continuation line inside an active multi-line call.
        if state.current_call_state and state.current_call_state.in_call:
            _process_call_continuation_line(trimmed_line_toks, state)
            continue

        # 4. Command invocation line (identifier followed by '(').
        if (
            non_ws_toks[0].type == TokenType.UNQUOTED_ARG
            and len(non_ws_toks) >= 2
            and non_ws_toks[1].type == TokenType.LPAREN
        ):
            _process_command_invocation_line(
                non_ws_toks, trimmed_line_toks, state, file_ctx
            )
            continue

        # 5. Raw token line fallback (bracket comments, bare unrecognised tokens, etc.).
        raw_line_text = "".join(t.value for t in line_toks).rstrip()
        state.flush_comments(
            INDENT_STR * (state.block_indent_level + (1 if in_call else 0))
        )
        state.formatted_lines.append(raw_line_text)

    if state.pending_comments:
        state.flush_comments(INDENT_STR * state.block_indent_level)

    return "\n".join(state.formatted_lines).rstrip() + "\n"


def process_file(
    filepath: str, inplace: bool = False, dry_run: bool = False, show_diff: bool = False
) -> bool | None:
    """Reads, formats, and optionally writes back a single CMake file.

    Returns:
        True   — file had formatting changes (or would have, in dry-run mode).
        False  — file is already correctly formatted; no changes needed.
        None   — a read or write error occurred; the file was not modified.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None

    try:
        formatted = format_cmake_content(original)
    except LexError as e:
        print(f"Error parsing file {filepath}: {e}", file=sys.stderr)
        return None

    has_changes = original != formatted

    if has_changes:
        if show_diff:
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                formatted.splitlines(keepends=True),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )
            sys.stdout.writelines(diff)

        if inplace and not dry_run:
            temp_file = None
            try:
                target_path = Path(filepath)
                temp_file = target_path.with_name(
                    f".{target_path.name}.tmp_{os.getpid()}"
                )
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(formatted)
                os.replace(temp_file, target_path)
                print(f"Formatted {filepath}")
            except Exception as e:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                print(f"Error writing file {filepath}: {e}", file=sys.stderr)
                return None
        elif dry_run:
            print(f"Formatting needed: {filepath}")

    return has_changes


def find_cmake_files(paths: list[str]) -> list[str]:
    """Recursively discovers CMake files (CMakeLists.txt and *.cmake) under the given paths.

    Directories starting with '.' and directories starting with 'build' are excluded from traversal.
    Plain file paths are included directly without filtering.
    Returns files in sorted order for deterministic output across platforms and runs.
    """
    cmake_files = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            cmake_files.append(str(path))
        elif path.is_dir():
            for root, dirs, files in path.walk():
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not d.startswith(".") and not d.startswith("build")
                )
                for f in sorted(files):
                    if f == "CMakeLists.txt" or f.endswith(".cmake"):
                        cmake_files.append(str(root / f))
    return cmake_files


def main() -> None:
    """Entry point: parses CLI arguments and drives file discovery, pre-scanning, and formatting."""
    parser = argparse.ArgumentParser(
        description="LLVM and LLVM-libc CMake Formatter Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="*", help="Files or directories to format.")
    parser.add_argument(
        "-i", "--inplace", "--fix", action="store_true", help="Format files in-place."
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Check formatting without modifying files (Evaluation Mode).",
    )
    parser.add_argument(
        "--diff", action="store_true", help="Show unified diffs of formatting changes."
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes to use (default: 1).",
    )

    args = parser.parse_args()

    if not args.paths:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            try:
                formatted = format_cmake_content(content)
            except LexError as e:
                print(f"Error parsing stdin: {e}", file=sys.stderr)
                sys.exit(1)
            if args.dry_run:
                if content != formatted:
                    if args.diff:
                        diff = difflib.unified_diff(
                            content.splitlines(keepends=True),
                            formatted.splitlines(keepends=True),
                            fromfile="stdin",
                            tofile="formatted",
                        )
                        sys.stdout.writelines(diff)
                    sys.exit(1)
                else:
                    sys.exit(0)
            else:
                sys.stdout.write(formatted)
                sys.exit(0)
        else:
            parser.print_help()
            sys.exit(1)

    cmake_files = find_cmake_files(args.paths)
    if not cmake_files:
        print("No CMake files found.", file=sys.stderr)
        sys.exit(0)

    # Mutates WORKSPACE_CONTEXT in-place so that subsequent format_cmake_content()
    # calls (which clone it per-file) inherit the pre-scanned module schemas.
    pre_scan_workspace_modules(args.paths)

    files_needing_format = 0
    if args.jobs > 1 and len(cmake_files) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Each worker process receives the pre-scanned WORKSPACE_CONTEXT via
        # the initializer, which pickles it once per worker. This works
        # correctly with any start method ('fork' on Linux, 'spawn' on
        # macOS/Windows) — no need to force 'fork'.
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(WORKSPACE_CONTEXT,),
        ) as executor:
            future_to_file = {
                executor.submit(
                    process_file, fpath, args.inplace, args.dry_run, args.diff
                ): fpath
                for fpath in cmake_files
            }
            for future in as_completed(future_to_file):
                fpath = future_to_file[future]
                try:
                    result = future.result()
                    if result is True:
                        files_needing_format += 1
                except Exception as e:
                    print(f"Error processing {fpath}: {e}", file=sys.stderr)
    else:
        for fpath in cmake_files:
            changed = process_file(
                fpath, inplace=args.inplace, dry_run=args.dry_run, show_diff=args.diff
            )
            if changed is True:
                files_needing_format += 1

    if args.dry_run and files_needing_format > 0:
        print(f"\n{files_needing_format} file(s) need formatting.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
