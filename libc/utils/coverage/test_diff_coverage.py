#!/usr/bin/env python3
#
# ===- Unit tests for diff coverage analyzer -----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_coverage import (
    CoverageJSONParser,
    DiffHunk,
    DiffParser,
    FilePatchMetrics,
    PatchCoverageSummary,
    calculate_patch_statistics,
    format_annotated_diff,
    format_breakdown_table,
    format_line_ranges,
    format_metadata_section,
    format_status_banner,
    is_executable_line,
    render_patch_report,
)


class TestDiffParser(unittest.TestCase):
    """Exhaustive unit tests for unified diff parsing."""

    def test_single_file_single_hunk(self) -> None:
        raw_diff = """diff --git a/libc/src/ctype/isalpha.cpp b/libc/src/ctype/isalpha.cpp
--- a/libc/src/ctype/isalpha.cpp
+++ b/libc/src/ctype/isalpha.cpp
@@ -10,3 +10,4 @@
 int isalpha(int c) {
+  int x = c;
   return x;
 }
"""
        files = DiffParser.parse(raw_diff)
        self.assertIn("libc/src/ctype/isalpha.cpp", files)
        hunks = files["libc/src/ctype/isalpha.cpp"]
        self.assertEqual(len(hunks), 1)
        self.assertEqual(len(hunks[0].lines), 4)
        self.assertEqual(hunks[0].lines[0], (" ", "int isalpha(int c) {", 10))
        self.assertEqual(hunks[0].lines[1], ("+", "  int x = c;", 11))
        self.assertEqual(hunks[0].lines[2], (" ", "  return x;", 12))
        self.assertEqual(hunks[0].lines[3], (" ", "}", 13))

    def test_multi_file_multi_hunk(self) -> None:
        raw_diff = """diff --git a/libc/src/ctype/isalpha.cpp b/libc/src/ctype/isalpha.cpp
--- a/libc/src/ctype/isalpha.cpp
+++ b/libc/src/ctype/isalpha.cpp
@@ -5,2 +5,3 @@
+// Header comment
 int isalpha(int c);
@@ -20,2 +21,3 @@
+  int z = 1;
   return z;
diff --git a/libc/src/math/sin.cpp b/libc/src/math/sin.cpp
--- a/libc/src/math/sin.cpp
+++ b/libc/src/math/sin.cpp
@@ -1,3 +1,4 @@
+// Math file
 double sin(double x) {
+  return x;
 }
"""
        files = DiffParser.parse(raw_diff)
        self.assertEqual(len(files), 2)
        self.assertIn("libc/src/ctype/isalpha.cpp", files)
        self.assertIn("libc/src/math/sin.cpp", files)
        self.assertEqual(len(files["libc/src/ctype/isalpha.cpp"]), 2)
        self.assertEqual(len(files["libc/src/math/sin.cpp"]), 1)

    def test_deleted_and_renamed_files(self) -> None:
        raw_diff = """diff --git a/libc/src/old.cpp b/libc/src/old.cpp
deleted file mode 100644
--- a/libc/src/old.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-int old_func();
"""
        files = DiffParser.parse(raw_diff)
        self.assertEqual(len(files), 0)

    def test_diff_from_temporary_file(self) -> None:
        raw_diff = """diff --git a/libc/src/string/strlen.cpp b/libc/src/string/strlen.cpp
--- a/libc/src/string/strlen.cpp
+++ b/libc/src/string/strlen.cpp
@@ -1,2 +1,3 @@
 size_t strlen(const char *s) {
+  return 0;
 }
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(raw_diff)
            f_path = f.name

        try:
            files = DiffParser.parse(f_path)
            self.assertIn("libc/src/string/strlen.cpp", files)
            self.assertEqual(len(files["libc/src/string/strlen.cpp"][0].lines), 3)
        finally:
            os.remove(f_path)

    def test_diff_with_no_newline_warning(self) -> None:
        raw_diff = """diff --git a/libc/src/stdio/puts.cpp b/libc/src/stdio/puts.cpp
--- a/libc/src/stdio/puts.cpp
+++ b/libc/src/stdio/puts.cpp
@@ -1,2 +1,2 @@
-int puts(const char *s);
+int puts(const char *str);
\\ No newline at end of file
"""
        files = DiffParser.parse(raw_diff)
        self.assertIn("libc/src/stdio/puts.cpp", files)
        lines = files["libc/src/stdio/puts.cpp"][0].lines
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], ("+", "int puts(const char *str);", 1))


class TestExecutableLineFilter(unittest.TestCase):
    """Exhaustive tests for filtering executable lines vs comments and declarations."""

    def test_non_executable_comments_and_whitespace(self) -> None:
        test_cases = [
            "",
            "   ",
            "\t\t",
            "// Single line comment",
            "   // Indented comment",
            "/* Multi-line block start",
            " * Continuation line",
            " */ End of comment block",
        ]
        for line in test_cases:
            with self.subTest(line=line):
                self.assertFalse(is_executable_line(line))

    def test_non_executable_syntax_and_preprocessor(self) -> None:
        test_cases = [
            "{",
            "}",
            "};",
            "{};",
            ": m_val(0)",
            "#include <stddef.h>",
            "#define FOO 1",
            "#ifdef LIBC_ENABLE_COVERAGE",
            "#endif",
            "namespace __llvm_libc {",
            "extern \"C\" {",
            "using size_t = unsigned long;",
            "template <typename T>",
            "typedef int (*func_ptr)(void);",
            "__attribute__((noinline))",
        ]
        for line in test_cases:
            with self.subTest(line=line):
                self.assertFalse(is_executable_line(line))

    def test_non_executable_type_definitions(self) -> None:
        test_cases = [
            "struct Foo {",
            "struct Foo;",
            "class Bar {",
            "class Bar;",
            "enum Color {",
            "enum class Status : int {",
        ]
        for line in test_cases:
            with self.subTest(line=line):
                self.assertFalse(is_executable_line(line))

    def test_executable_statements(self) -> None:
        test_cases = [
            "int x = 5;",
            "return a + b;",
            "struct Foo f = init_foo();",
            "class Bar b(10);",
            "if (c >= 'a' && c <= 'z')",
            "for (int i = 0; i < 10; ++i) {",
            "while (*s++) {",
            "switch (op) {",
            "case 1:",
            "break;",
            "continue;",
            "goto cleanup;",
            "foo(); // inline comment",
        ]
        for line in test_cases:
            with self.subTest(line=line):
                self.assertTrue(is_executable_line(line))


class TestFormatLineRanges(unittest.TestCase):
    """Tests for line number formatting into human-readable spans."""

    def test_formatting_variations(self) -> None:
        self.assertEqual(format_line_ranges(set()), "None")
        self.assertEqual(format_line_ranges({42}), "`L42`")
        self.assertEqual(format_line_ranges({10, 11, 12}), "`L10-L12`")
        self.assertEqual(
            format_line_ranges({5, 6, 7, 10, 15, 16, 20}),
            "`L5-L7`, `L10`, `L15-L16`, `L20`",
        )


class TestCoverageJSONParser(unittest.TestCase):
    """Tests for parsing llvm-cov JSON export structures and mapping to diffs."""

    def test_invalid_json_handling(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("{ invalid json")
            f_path = f.name

        try:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    CoverageJSONParser.load(f_path)
        finally:
            os.remove(f_path)

    def test_extract_patch_matrix_matching(self) -> None:
        cov_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/runner/work/llvm-project/libc/src/math/cos.cpp",
                            "segments": [
                                [10, 1, 5, True, True, False],
                                [12, 1, 0, True, True, False],
                                [14, 1, 0, False, False, False],
                            ],
                            "mcdc_records": [
                                [10, 5, 10, 25, 2, 2, 2, 1, 1, [True, True]]
                            ],
                        }
                    ]
                }
            ]
        }
        diff_files = {"libc/src/math/cos.cpp": []}
        matrix = CoverageJSONParser.extract_patch_matrix(cov_data, diff_files)

        self.assertIn("libc/src/math/cos.cpp", matrix)
        self.assertIn(10, matrix["libc/src/math/cos.cpp"]["covered"])
        self.assertIn(11, matrix["libc/src/math/cos.cpp"]["covered"])
        self.assertIn(12, matrix["libc/src/math/cos.cpp"]["missed"])
        self.assertEqual(len(matrix["libc/src/math/cos.cpp"]["mcdc_decisions"]), 1)


class TestStatisticsAndReporting(unittest.TestCase):
    """Tests for patch coverage calculation, diagnostic generation, and Markdown rendering."""

    def test_line_coverage_calculation(self) -> None:
        diff_text = """diff --git a/libc/src/math/fabs.cpp b/libc/src/math/fabs.cpp
--- a/libc/src/math/fabs.cpp
+++ b/libc/src/math/fabs.cpp
@@ -10,3 +10,5 @@
 double fabs(double x) {
+  if (x < 0)
+    return -x;
   return x;
 }
"""
        diff_files = DiffParser.parse(diff_text)
        cov_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "libc/src/math/fabs.cpp",
                            "segments": [
                                [10, 1, 10, True, True, False],
                                [12, 1, 0, True, True, False],
                                [14, 1, 0, False, False, False],
                            ],
                        }
                    ]
                }
            ]
        }
        matrix = CoverageJSONParser.extract_patch_matrix(cov_data, diff_files)
        stats = calculate_patch_statistics(diff_files, matrix)

        self.assertEqual(stats.total_covered_lines, 1)  # line 11 (if x < 0)
        self.assertEqual(stats.total_missed_lines, 1)   # line 12 (return -x)
        self.assertEqual(stats.total_lines, 2)
        self.assertEqual(stats.line_coverage_percentage, 50.0)
        self.assertFalse(stats.has_mcdc)

    def test_mcdc_decision_diagnostics(self) -> None:
        diff_text = """diff --git a/libc/src/ctype/isspace.cpp b/libc/src/ctype/isspace.cpp
--- a/libc/src/ctype/isspace.cpp
+++ b/libc/src/ctype/isspace.cpp
@@ -10,2 +10,3 @@
 int isspace(int c) {
+  if (c == ' ' || c == '\t' || c == '\n')
     return 1;
"""
        diff_files = DiffParser.parse(diff_text)
        cov_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "libc/src/ctype/isspace.cpp",
                            "segments": [
                                [10, 1, 10, True, True, False],
                                [13, 1, 0, False, False, False],
                            ],
                            "mcdc_records": [
                                [11, 7, 11, 40, 3, 2, 3, 1, 1, [True, True, False]]
                            ],
                        }
                    ]
                }
            ]
        }
        matrix = CoverageJSONParser.extract_patch_matrix(cov_data, diff_files)
        stats = calculate_patch_statistics(diff_files, matrix)

        self.assertTrue(stats.has_mcdc)
        self.assertEqual(stats.total_mcdc_covered_conditions, 2)
        self.assertEqual(stats.total_mcdc_total_conditions, 3)
        self.assertEqual(stats.mcdc_coverage_percentage, 66.66666666666666)
        self.assertEqual(stats.total_decisions_count, 1)
        self.assertEqual(stats.fully_verified_decisions, 0)

        file_metrics = stats.files["libc/src/ctype/isspace.cpp"]
        self.assertIn("C3 unverified", file_metrics.condition_diagnostics[0])
        self.assertEqual(file_metrics.unverified_decision_lines[11], ["C3"])

    def test_format_status_banner(self) -> None:
        # Full statement & full MC/DC
        summary_full = PatchCoverageSummary(
            total_covered_lines=10,
            total_missed_lines=0,
            total_mcdc_covered_conditions=4,
            total_mcdc_total_conditions=4,
            total_decisions_count=2,
            fully_verified_decisions=2,
        )
        banner_full = format_status_banner(summary_full)
        self.assertIn("100.00% Line", banner_full)
        self.assertIn("100.00% MC/DC", banner_full)

        # Full statement, partial MC/DC
        summary_partial = PatchCoverageSummary(
            total_covered_lines=10,
            total_missed_lines=0,
            total_mcdc_covered_conditions=3,
            total_mcdc_total_conditions=4,
            total_decisions_count=2,
            fully_verified_decisions=1,
        )
        banner_partial = format_status_banner(summary_partial)
        self.assertIn("75.0% MC/DC", banner_partial)

        # Lines missed
        summary_warn = PatchCoverageSummary(
            total_covered_lines=8,
            total_missed_lines=2,
        )
        banner_warn = format_status_banner(summary_warn)
        self.assertIn("80.00%", banner_warn)
        self.assertIn("unexecuted lines detected in patch", banner_warn)

    def test_format_metadata_section(self) -> None:
        metadata_string = format_metadata_section(
            base_commit_sha="abcdef1234567890",
            head_commit_sha="123456abcdef7890",
            base_branch_name="main",
            head_branch_name="my-pr",
            targeted_tests_string="libc.test.src.math.sin_test libc.test.src.math.cos_test",
            base_repository="llvm/llvm-project",
            head_repository="user/llvm-project",
        )
        self.assertIn("abcdef1", metadata_string)
        self.assertIn("123456a", metadata_string)
        self.assertIn("`libc.test.src.math.sin_test`, `libc.test.src.math.cos_test`", metadata_string)

    def test_format_breakdown_table_standard_and_mcdc(self) -> None:
        file_metrics = FilePatchMetrics(
            file_path="libc/src/math/tan.cpp",
            covered_lines={10, 11},
            missed_lines={12},
            mcdc_covered_conditions=2,
            mcdc_total_conditions=2,
            decisions_verified=1,
            decisions_total=1,
            condition_diagnostics=["`L10`: 2/2 verified"],
        )
        summary_mcdc = PatchCoverageSummary(
            files={"libc/src/math/tan.cpp": file_metrics},
            total_covered_lines=2,
            total_missed_lines=1,
            total_mcdc_covered_conditions=2,
            total_mcdc_total_conditions=2,
            total_decisions_count=1,
            fully_verified_decisions=1,
        )
        table_mcdc = format_breakdown_table(summary_mcdc, head_repository="llvm/llvm-project")
        self.assertIn("MC/DC Conditions", table_mcdc)
        self.assertIn("`L10`: 2/2 verified", table_mcdc)

        summary_std = PatchCoverageSummary(
            files={"libc/src/math/tan.cpp": file_metrics},
            total_covered_lines=2,
            total_missed_lines=1,
        )
        table_std = format_breakdown_table(summary_std, head_repository="llvm/llvm-project")
        self.assertIn("Unexecuted Line Spans", table_std)
        self.assertIn("`L12`", table_std)

    def test_render_patch_report_empty_diff(self) -> None:
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            render_patch_report(
                diff_files={},
                coverage_matrix={},
                base_commit_sha="base",
                head_commit_sha="head",
                base_branch_name="main",
                head_branch_name="patch",
            )
        output_text = stdout_buffer.getvalue()
        self.assertIn("### Coverage Summary", output_text)
        self.assertIn("No `.cpp` source files in `libc/src/` were modified", output_text)


if __name__ == "__main__":
    unittest.main()
