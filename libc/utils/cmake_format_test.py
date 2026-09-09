#!/usr/bin/env python3
"""
Unit tests for LLVM and Subproject CMake Formatter Utility (`libc/utils/cmake_format.py`).

Verifies all 9 formatting rules, dynamic schema learning, codebase idempotency,
and golden output snapshots.
"""

import unittest
import os
import sys

# Add the directory containing cmake_format.py to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import cmake_format


class TestRule1_CommandCasing(unittest.TestCase):
    """Rule 1: Command Casing."""

    def test_1_1_builtin_lowercase(self):
        """Rule 1.1: Built-in language commands are cased in lowercase."""
        code = "SET(FOO bar)\nIF(FOO)\n  ADD_LIBRARY(lib STATIC)\nENDIF()\n"
        expected = "set(FOO bar)\nif(FOO)\n  add_library(lib STATIC)\nendif()\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_1_2_module_canonical_casing(self):
        """Rule 1.2: Module commands retain canonical declared casing."""
        code = "externalproject_add(\n  proj\n  PREFIX\n  dir\n)\n"
        expected = "ExternalProject_Add(\n  proj\n  PREFIX\n  dir\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_1_3_custom_function_casing(self):
        """Rule 1.3: Custom user functions/macros retain their original casing."""
        code = "add_libc_unittest(\n  my_test\n  SRCS\n    my_test.cpp\n)\n"
        expected = "add_libc_unittest(\n  my_test\n  SRCS\n    my_test.cpp\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule2_ParenthesisSpacingAndWhitespace(unittest.TestCase):
    """Rule 2: Parenthesis Spacing & Whitespace Collapsing."""

    def test_2_1_zero_space_before_opening_paren(self):
        """Rule 2.1: Zero space between command identifier and opening '('."""
        code = "set  (FOO bar)\n"
        expected = "set(FOO bar)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_2_2_collapse_arg_whitespace(self):
        """Rule 2.2: Collapses multiple spaces between single-line arguments down to one."""
        code = "add_entrypoint_object(target_name    DEPENDS   dep1    dep2)\n"
        expected = "add_entrypoint_object(target_name DEPENDS dep1 dep2)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule3_QuotedAndBracketImmutability(unittest.TestCase):
    """Rule 3: Quoted String & Bracket Immutability."""

    def test_3_1_quoted_and_bracket_tokens(self):
        """Rule 3.1: Quoted arguments and bracket arguments are single immutable tokens."""
        code = 'set(MSG "Hello \\"World\\"")\nset(CODE [=[raw string\n  content]=])\n'
        expected = (
            'set(MSG "Hello \\"World\\"")\nset(CODE [=[raw string\n  content]=])\n'
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_3_2_multiline_string_preservation(self):
        """Rule 3.2: Multi-line quoted strings are preserved 100% untouched."""
        code = 'set(DOC "\n  Line 1\n    Line 2\n")\n'
        expected = 'set(DOC "\n  Line 1\n    Line 2\n")\n'
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule4_EmptyClosures(unittest.TestCase):
    """Rule 4: Empty Closures."""

    def test_4_1_empty_closing_parens(self):
        """Rule 4.1: Control block closures strip old condition arguments down to empty ()."""
        code = "if(FOO)\n  set(BAR 1)\nelse(FOO)\n  set(BAR 0)\nendif(FOO)\n"
        expected = "if(FOO)\n  set(BAR 1)\nelse()\n  set(BAR 0)\nendif()\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule5_SchemaAwareKeywordCasing(unittest.TestCase):
    """Rule 5: Schema-Aware Keyword Casing."""

    def test_5_1_keyword_uppercasing(self):
        """Rule 5.1: Schema keywords are upper-cased."""
        code = "add_entrypoint_object(\n  target\n  srcs\n    src.cpp\n  depends\n    dep\n)\n"
        expected = "add_entrypoint_object(\n  target\n  SRCS\n    src.cpp\n  DEPENDS\n    dep\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_5_2_positional_args_untouched(self):
        """Rule 5.2: Positional arguments, file paths, and target names are left untouched."""
        code = "add_header_macro(\n  string\n  ../libc/include/string.yaml\n  string.h\n  DEPENDS\n    .llvm_libc_common_h\n)\n"
        expected = "add_header_macro(\n  string\n  ../libc/include/string.yaml\n  string.h\n  DEPENDS\n    .llvm_libc_common_h\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_5_3_custom_function_positional_args_untouched(self):
        """Rule 5.3: Positional output_var parameters in custom functions retain casing."""
        code = '_get_common_compile_options(compile_options "${ADD_OBJECT_FLAGS}")\n'
        expected = (
            '_get_common_compile_options(compile_options "${ADD_OBJECT_FLAGS}")\n'
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule6_MultilineArgumentLayout(unittest.TestCase):
    """Rule 6: Multi-line Argument Layout & Keyword Schema Mechanics."""

    def test_6_1_option_keyword_scope(self):
        """Rule 6.1: Option keywords (0 values) close immediately."""
        code = "add_library(\n  target\n  ALIAS\n    real_target\n)\n"
        expected = "add_library(\n  target\n  ALIAS\n    real_target\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_6_2_single_value_keyword_scope(self):
        """Rule 6.2: Single-value keywords consume 1 argument then close."""
        code = "execute_process(\n  WORKING_DIRECTORY\n    /tmp\n  OUTPUT_VARIABLE\n    out\n)\n"
        expected = "execute_process(\n  WORKING_DIRECTORY\n    /tmp\n  OUTPUT_VARIABLE\n    out\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_6_3a_multivalue_separate_line(self):
        """Rule 6.3a: Multi-value keyword on new line indents items +4 spaces (+2 relative to keyword)."""
        code = "add_entrypoint_object(\n  target\n  SRCS\n    s1.cpp\n    s2.cpp\n)\n"
        expected = (
            "add_entrypoint_object(\n  target\n  SRCS\n    s1.cpp\n    s2.cpp\n)\n"
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_6_3b_multivalue_header_line(self):
        """Rule 6.3b: Multi-value keyword on header line indents items +2 spaces relative to call base."""
        code = "add_entrypoint_object(target SRCS s1.cpp s2.cpp)\n"
        expected = "add_entrypoint_object(target SRCS s1.cpp s2.cpp)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_6_4_argument_forwarding_vars(self):
        """Rule 6.4: Argument forwarding variables (${ARGN}, ${ARGV}) align at +2 spaces and reset active keyword."""
        code = "add_entrypoint_object(\n  target\n  DEPENDS\n    dep1\n  ${ARGN}\n)\n"
        expected = (
            "add_entrypoint_object(\n  target\n  DEPENDS\n    dep1\n  ${ARGN}\n)\n"
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_6_5_closing_paren_indent(self):
        """Rule 6.5: Closing parenthesis ')' aligns at +0 spaces (call base)."""
        code = "add_entrypoint_object(\n  target\n  SRCS\n    s1.cpp\n)\n"
        expected = "add_entrypoint_object(\n  target\n  SRCS\n    s1.cpp\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule7_ControlBlockIndentation(unittest.TestCase):
    """Rule 7: Control Block Indentation."""

    def test_7_1_nested_control_blocks(self):
        """Rule 7.1: Nests 2 spaces per control block level."""
        code = "if(A)\nif(B)\nset(C 1)\nendif()\nendif()\n"
        expected = "if(A)\n  if(B)\n    set(C 1)\n  endif()\nendif()\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_7_2_middle_blocks(self):
        """Rule 7.2: else() and elseif() align at block base level."""
        code = "if(A)\nset(X 1)\nelseif(B)\nset(X 2)\nelse()\nset(X 3)\nendif()\n"
        expected = (
            "if(A)\n  set(X 1)\nelseif(B)\n  set(X 2)\nelse()\n  set(X 3)\nendif()\n"
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule8_CommentFormatting(unittest.TestCase):
    """Rule 8: Comment Formatting & Lookahead Buffering."""

    def test_8_1_lookahead_comment_buffering(self):
        """Rule 8.1: Line comment aligns with the indentation of the upcoming code line."""
        code = "add_entrypoint_object(\n  target\n  SRCS\n    # Comment for s1\n    s1.cpp\n)\n"
        expected = "add_entrypoint_object(\n  target\n  SRCS\n    # Comment for s1\n    s1.cpp\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_8_2_standalone_comments_blank_line(self):
        """Rule 8.2: Standalone comments followed by a blank line flush at current base indent."""
        code = "# Standalone comment\n\nadd_library(target s1.cpp)\n"
        expected = "# Standalone comment\n\nadd_library(target s1.cpp)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_8_3a_closing_paren_keyword_comment(self):
        """Rule 8.3a: Comment following a keyword before closing ')' indents under keyword (+4 spaces).

        In practice, UNIT_TEST_ONLY is learned as an option from cmake_parse_arguments in
        LLVMLibCTestRules.cmake via pre_scan_workspace_modules. We supply an equivalent
        inline definition so the test is self-contained.
        """
        # Inline definition matching the real LLVMLibCTestRules.cmake declaration.
        schema_stub = """
function(add_libc_test test_name)
  cmake_parse_arguments(
    "LIBC_TEST"
    "UNIT_TEST_ONLY;HERMETIC_TEST_ONLY"
    ""
    ""
    ${ARGN}
  )
endfunction()
"""
        ctx = cmake_format.FormatterContext()
        cmake_format.scan_dynamic_schemas(schema_stub, ctx=ctx)
        code = "add_libc_test(\n  hash_test\n  UNIT_TEST_ONLY\n    # Explanation for option\n)\n"
        expected = "add_libc_test(\n  hash_test\n  UNIT_TEST_ONLY\n    # Explanation for option\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code, ctx=ctx), expected)

    def test_8_3b_closing_paren_list_comment(self):
        """Rule 8.3b: Comment following a list item before closing ')' aligns at top-level call (+2 spaces)."""
        code = "add_libc_test(\n  stdbit_test\n  DEPENDS\n    dep1\n  # Top level call comment\n)\n"
        expected = "add_libc_test(\n  stdbit_test\n  DEPENDS\n    dep1\n  # Top level call comment\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_8_4_comment_space_hygiene(self):
        """Rule 8.4: Missing space after '#' is added; special markers (#---, #!) preserved."""
        code = "#no space\n#--- header ---\n#!shebang\n"
        expected = "# no space\n#--- header ---\n#!shebang\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestRule9_CleanlinessAndFileHygiene(unittest.TestCase):
    """Rule 9: Cleanliness & File Hygiene."""

    def test_9_1_strip_trailing_whitespace(self):
        """Rule 9.1: Trailing whitespace on all lines is stripped."""
        code = "set(FOO bar)   \nset(BAR baz)  \n"
        expected = "set(FOO bar)\nset(BAR baz)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_9_2_single_trailing_newline(self):
        """Rule 9.2: Non-empty files end with exactly one trailing newline."""
        code = "set(FOO bar)\n\n\n"
        expected = "set(FOO bar)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_9_3_empty_file_preservation(self):
        """Rule 9.3: Empty 0-byte files are preserved as 0-byte."""
        self.assertEqual(cmake_format.format_cmake_content(""), "")
        self.assertEqual(cmake_format.format_cmake_content("   \n\n  \t "), "")


class TestDynamicSchemaLearning(unittest.TestCase):
    """Dynamic Schema Learning Tests."""

    def test_function_block_parser(self):
        """Verifies cmake_parse_arguments extraction inside function/macro AST blocks."""
        code = """
function(custom_target_func target)
  cmake_parse_arguments(
    "FUNC"
    "OPTION1;OPTION2"
    "SINGLE1"
    "MULTI1;MULTI2"
    ${ARGN}
  )
endfunction()
"""
        cmake_format.scan_dynamic_schemas(code)
        schema = cmake_format.get_schema_for_cmd("custom_target_func")
        self.assertIn("OPTION1", schema.options)
        self.assertIn("SINGLE1", schema.one_value)
        self.assertIn("MULTI1", schema.multi_value)

    def test_variable_list_scanner(self):
        """Verifies dynamic learning from set(*_OPTION_ARGS), set(*_SINGLE_VALUE_ARGS), set(*_MULTI_VALUE_ARGS)."""
        code = """
set(CUSTOM_OPTION_ARGS OPT_A OPT_B)
set(CUSTOM_SINGLE_VALUE_ARGS SING_A)
set(CUSTOM_MULTI_VALUE_ARGS MULT_A MULT_B)
"""
        ctx = cmake_format.FormatterContext()
        cmake_format.scan_dynamic_schemas(code, ctx=ctx)
        self.assertIn("OPT_A", ctx.learned_options)
        self.assertIn("SING_A", ctx.learned_one_value)
        self.assertIn("MULT_A", ctx.learned_multi_value)

    def test_cache_variable_exclusion(self):
        """Verifies cache variable declarations like set(FOO "" CACHE STRING "...") are excluded."""
        code = 'set(LIBC_COMPILE_OPTIONS_DEFAULT "" CACHE STRING "Docstring")\n'
        ctx = cmake_format.FormatterContext()
        cmake_format.scan_dynamic_schemas(code, ctx=ctx)
        self.assertNotIn("STRING", ctx.learned_options)


class TestCodebaseIdempotency(unittest.TestCase):
    """Codebase Idempotency Test."""

    def test_libc_codebase_idempotency(self):
        """Verifies format(format(content)) == format(content) over libc/ files."""
        libc_dir = os.path.join(SCRIPT_DIR, "..")
        cmake_files = cmake_format.find_cmake_files([libc_dir])
        self.assertGreater(len(cmake_files), 0, "No CMake files found under libc/")

        cmake_format.pre_scan_workspace_modules([libc_dir])

        for fpath in cmake_files:
            with self.subTest(file=fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    original = f.read()
                formatted = cmake_format.format_cmake_content(original)
                reformatted = cmake_format.format_cmake_content(formatted)
                self.assertEqual(
                    formatted, reformatted, f"Formatting is not idempotent for {fpath}"
                )


class TestEdgeCasesAndIsolation(unittest.TestCase):
    """Edge cases for parenthesis counting in comments/strings and state isolation."""

    def test_parens_in_comments_and_strings(self):
        """Verifies parentheses inside comments and strings do not corrupt paren_balance or indentation."""
        code = (
            "add_custom_target(\n"
            "  mytarget\n"
            "  # Comment with parentheses () and (extra)\n"
            '  "file_with_(parens).cpp"\n'
            "  DEPENDS\n"
            "    dep1\n"
            ")\n"
        )
        expected = (
            "add_custom_target(\n"
            "  mytarget\n"
            "  # Comment with parentheses () and (extra)\n"
            '  "file_with_(parens).cpp"\n'
            "  DEPENDS\n"
            "    dep1\n"
            ")\n"
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_bracket_comment_with_equals(self):
        """Verifies bracket comments with '=' delimiters (#[=[...]=]) are properly preserved."""
        code = (
            "#[=[\n"
            "  Multi-line comment with equals\n"
            "  set(FOO bar)\n"
            "]=]\n"
            "set(BAR baz)\n"
        )
        expected = (
            "#[=[\n"
            "  Multi-line comment with equals\n"
            "  set(FOO bar)\n"
            "]=]\n"
            "set(BAR baz)\n"
        )
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_unterminated_quoted_argument_raises_lexerror(self):
        """Verifies unterminated quoted string arguments raise LexError."""
        code_quote = 'set(FOO "unterminated string'
        with self.assertRaises(cmake_format.LexError) as cm:
            cmake_format.format_cmake_content(code_quote)
        self.assertIn("unterminated quoted argument", str(cm.exception))

    def test_unterminated_bracket_comment_raises_lexerror(self):
        """Verifies unterminated bracket comments raise LexError."""
        code_bc = "#[=[ unterminated bracket comment"
        with self.assertRaises(cmake_format.LexError) as cm:
            cmake_format.format_cmake_content(code_bc)
        self.assertIn("unterminated bracket comment", str(cm.exception))

    def test_unterminated_bracket_argument_raises_lexerror(self):
        """Verifies unterminated bracket arguments raise LexError."""
        code_ba = "set(FOO [=[unterminated bracket arg)"
        with self.assertRaises(cmake_format.LexError) as cm:
            cmake_format.format_cmake_content(code_ba)
        self.assertIn("unterminated bracket argument", str(cm.exception))

    def test_global_state_isolation(self):
        """Verifies custom schemas learned in File A do not leak into formatting of File B when using isolated contexts."""
        file_a = (
            "set(MY_CUSTOM_OPTION_ARGS OPT_ISOLATED_A OPT_ISOLATED_B)\n"
            "my_custom_cmd(OPT_ISOLATED_A valA OPT_ISOLATED_B valB)\n"
        )
        file_b = "my_other_cmd(opt_isolated_a vala opt_isolated_b valb)\n"
        ctx_a = cmake_format.FormatterContext()
        ctx_b = cmake_format.FormatterContext()
        cmake_format.format_cmake_content(file_a, ctx=ctx_a)
        result_b = cmake_format.format_cmake_content(file_b, ctx=ctx_b)
        expected_b = "my_other_cmd(opt_isolated_a vala opt_isolated_b valb)\n"
        self.assertEqual(result_b, expected_b)

    def test_pre_scan_workspace_modules(self):
        """Verifies pre_scan_workspace_modules discovers and scans modules in cmake/modules/."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_dir = os.path.join(tmpdir, "cmake", "modules")
            os.makedirs(mod_dir, exist_ok=True)
            mod_file = os.path.join(mod_dir, "CustomModule.cmake")
            with open(mod_file, "w", encoding="utf-8") as f:
                f.write("set(MY_MOD_OPTION_ARGS MOD_OPT1 MOD_OPT2)\n")

            ctx = cmake_format.FormatterContext()
            cmake_format.pre_scan_workspace_modules([tmpdir], ctx=ctx)
            self.assertIn("MOD_OPT1", ctx.learned_options)
            self.assertIn("MOD_OPT2", ctx.learned_options)

    def test_pre_scan_workspace_modules_fails_fast_on_error(self):
        """Verifies pre_scan_workspace_modules exits with code 1 if a module contains invalid syntax."""
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_dir = os.path.join(tmpdir, "cmake", "modules")
            os.makedirs(mod_dir, exist_ok=True)
            mod_file = os.path.join(mod_dir, "BrokenModule.cmake")
            with open(mod_file, "w", encoding="utf-8") as f:
                f.write('message("unterminated string\n')

            ctx = cmake_format.FormatterContext()
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                with self.assertRaises(SystemExit) as cm:
                    cmake_format.pre_scan_workspace_modules([tmpdir], ctx=ctx)
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error pre-scanning module", err_buf.getvalue())

    def test_find_cmake_files_excludes_build_directories(self):
        """Verifies find_cmake_files excludes directories starting with 'build' and '.'."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for d in [
                "src",
                "build",
                "build-ninja",
                "build_debug",
                ".hidden",
            ]:
                sub = os.path.join(tmpdir, d)
                os.makedirs(sub, exist_ok=True)
                with open(
                    os.path.join(sub, "CMakeLists.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write("set(X 1)\n")

            found = cmake_format.find_cmake_files([tmpdir])
            rel_found = [os.path.relpath(p, tmpdir) for p in found]
            self.assertEqual(rel_found, [os.path.join("src", "CMakeLists.txt")])

    def test_stdin_pipe_formatting(self):
        """Verifies formatting unformatted CMake code via stdin."""
        code = "add_library(\n foo\n   STATIC\n   s1.cpp\n )\n"
        expected = "add_library(\n  foo\n  STATIC\n    s1.cpp\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


class TestParsingBugs(unittest.TestCase):
    """Regression tests for parsing and formatting bugs."""

    def test_bug_inline_comment_on_empty_closures(self):
        """Empty closure commands (endif, else, endfunction, etc.) preserve inline comments."""
        code = "if(FOO)\n  set(BAR 1)\nelse() # else comment\n  set(BAR 0)\nendif() # endif comment\n"
        expected = "if(FOO)\n  set(BAR 1)\nelse() # else comment\n  set(BAR 0)\nendif() # endif comment\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_bug_escaped_chars_in_unquoted_args(self):
        """Unquoted args containing escaped characters (e.g. \\#, \\(, \\), \\", \\ ) are preserved as single tokens."""
        code = "add_entrypoint_object(\n  target\n  SRCS\n    bar\\#baz\n    bar\\(baz\\)\n    C:\\Program\\ Files\\Foo\n)\n"
        expected = "add_entrypoint_object(\n  target\n  SRCS\n    bar\\#baz\n    bar\\(baz\\)\n    C:\\Program\\ Files\\Foo\n)\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_bug_continuation_line_starting_with_paren(self):
        """Continuation lines starting with ')' when paren balance > 0 do not prematurely close call state."""
        code = "if(\n  (A AND B)\n  ) AND (C AND D)\n)\n  set(X 1)\nendif()\n"
        expected = "if(\n  (A AND B)\n  ) AND (C AND D)\n)\n  set(X 1)\nendif()\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)

    def test_bug_multiline_set_schema_learning(self):
        """Multi-line set(*_MULTI_VALUE_ARGS ...) definitions are parsed correctly by schema scanner."""
        code = "set(CUSTOM_MULTI_VALUE_ARGS\n  MULT_LINE1\n  MULT_LINE2\n)\n"
        ctx = cmake_format.FormatterContext()
        cmake_format.scan_dynamic_schemas(code, ctx=ctx)
        self.assertIn("MULT_LINE1", ctx.learned_multi_value)
        self.assertIn("MULT_LINE2", ctx.learned_multi_value)

    def test_bug_cmake_parse_arguments_var_refs(self):
        """Variable references like ${MY_OPTIONS} in cmake_parse_arguments are not registered as literal keywords."""
        code = """
function(my_func)
  cmake_parse_arguments("ARG" "${MY_OPTIONS}" "${MY_ONE_VAL}" "${MY_MULTI_VAL}" ${ARGN})
endfunction()
"""
        ctx = cmake_format.FormatterContext()
        cmake_format.scan_dynamic_schemas(code, ctx=ctx)
        schema = cmake_format.get_schema_for_cmd("my_func", ctx=ctx)
        self.assertNotIn("MY_OPTIONS", schema.options)
        self.assertNotIn("MY_ONE_VAL", schema.one_value)
        self.assertNotIn("MY_MULTI_VAL", schema.multi_value)

    def test_bug_directive_comment_preservation(self):
        """Directives like #nolint, #pragma, #include, # cmake-lint, # cmake-format in comments are preserved without forced spaces."""
        code = "#nolint\n#pragma once\n#include <foo>\n# cmake-lint: disable=syntax-error\n# cmake-format: off\n#cmake-lint: disable=syntax-error\n"
        expected = "#nolint\n#pragma once\n#include <foo>\n# cmake-lint: disable=syntax-error\n# cmake-format: off\n#cmake-lint: disable=syntax-error\n"
        self.assertEqual(cmake_format.format_cmake_content(code), expected)


if __name__ == "__main__":
    unittest.main()
