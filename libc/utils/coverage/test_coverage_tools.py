#!/usr/bin/env python3
# ===-- test_coverage_tools.py - Unit tests for coverage utilities --------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Add parent directory to path to import coverage tools
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from patch_report import (
    DiffParser,
    CoverageJSONParser,
    is_executable_line,
    format_line_ranges,
    render_patch_report,
)
from full_report import render_full_report


class TestDiffParser(unittest.TestCase):
    def test_parse_unified_diff(self):
        sample_diff = """diff --git a/libc/src/string/memchr.cpp b/libc/src/string/memchr.cpp
index 1234567..89abcdef 100644
--- a/libc/src/string/memchr.cpp
+++ b/libc/src/string/memchr.cpp
@@ -10,6 +10,8 @@
 void *memchr(const void *src, int c, size_t n) {
+  const unsigned char *p = (const unsigned char *)src;
+  // Check boundaries
+  if (n == 0) return nullptr;
   return nullptr;
 }
"""
        parsed = DiffParser.parse(sample_diff)
        self.assertIn("libc/src/string/memchr.cpp", parsed)
        hunks = parsed["libc/src/string/memchr.cpp"]
        self.assertEqual(len(hunks), 1)

        added_lines = [l_num for l_type, _, l_num in hunks[0].lines if l_type == "+"]
        self.assertEqual(added_lines, [11, 12, 13])

    def test_is_executable_line(self):
        self.assertTrue(is_executable_line("  int x = 42;"))
        self.assertTrue(is_executable_line("  return nullptr;"))
        self.assertTrue(is_executable_line("  if (a > b) {"))

        self.assertFalse(is_executable_line("  // Just a comment"))
        self.assertFalse(is_executable_line("  /* Block comment */"))
        self.assertFalse(is_executable_line("  #include <stddef.h>"))
        self.assertFalse(is_executable_line("  {"))
        self.assertFalse(is_executable_line("  }"))
        self.assertFalse(is_executable_line("   "))


class TestCoverageJSONParser(unittest.TestCase):
    def test_extract_patch_matrix_standard(self):
        sample_cov_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/workspace/llvm-project/libc/src/string/memchr.cpp",
                            "segments": [
                                [10, 1, 5, True, True, False],
                                [15, 1, 0, True, True, False],
                                [20, 1, 0, False, False, False],
                            ],
                            "summary": {
                                "lines": {"count": 20, "covered": 10, "percent": 50.0},
                                "functions": {"count": 1, "covered": 1, "percent": 100.0},
                            },
                        }
                    ]
                }
            ]
        }

        diff_files = {"libc/src/string/memchr.cpp": []}
        matrix = CoverageJSONParser.extract_patch_matrix(sample_cov_data, diff_files)
        self.assertIn("libc/src/string/memchr.cpp", matrix)
        self.assertIn(10, matrix["libc/src/string/memchr.cpp"]["covered"])
        self.assertIn(14, matrix["libc/src/string/memchr.cpp"]["covered"])
        self.assertIn(15, matrix["libc/src/string/memchr.cpp"]["missed"])
        self.assertIn(19, matrix["libc/src/string/memchr.cpp"]["missed"])

    def test_extract_patch_matrix_with_mcdc(self):
        sample_mcdc_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/workspace/llvm-project/libc/src/string/memchr.cpp",
                            "segments": [
                                [10, 1, 5, True, True, False],
                                [20, 1, 0, False, False, False],
                            ],
                            "mcdc_records": [
                                [12, 5, 12, 25, 1, 1, 0, 0, 5, [True, False, False]]
                            ],
                            "summary": {
                                "lines": {"count": 10, "covered": 10, "percent": 100.0},
                                "functions": {"count": 1, "covered": 1, "percent": 100.0},
                                "mcdc": {"count": 3, "covered": 1, "notcovered": 2, "percent": 33.33},
                            },
                        }
                    ]
                }
            ]
        }

        diff_files = {"libc/src/string/memchr.cpp": []}
        matrix = CoverageJSONParser.extract_patch_matrix(sample_mcdc_data, diff_files)
        
        decisions = matrix["libc/src/string/memchr.cpp"]["mcdc_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["line_start"], 12)
        self.assertEqual(decisions[0]["total"], 3)
        self.assertEqual(decisions[0]["covered"], 1)


class TestLineRangeFormatter(unittest.TestCase):
    def test_format_contiguous_and_discrete_ranges(self):
        self.assertEqual(format_line_ranges(set()), "None")
        self.assertEqual(format_line_ranges({5}), "`L5`")
        self.assertEqual(format_line_ranges({10, 11, 12, 13}), "`L10-L13`")
        self.assertEqual(
            format_line_ranges({1, 2, 3, 7, 10, 11, 15}),
            "`L1-L3`, `L7`, `L10-L11`, `L15`",
        )


class TestPatchReportRendering(unittest.TestCase):
    def test_render_linear_pass(self):
        diff_text = """diff --git a/libc/src/string/memchr.cpp b/libc/src/string/memchr.cpp
--- a/libc/src/string/memchr.cpp
+++ b/libc/src/string/memchr.cpp
@@ -10,2 +10,2 @@
+  int a = 1;
+  int b = 2;
"""
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "libc/src/string/memchr.cpp": {
                "covered": {10, 11},
                "missed": set(),
                "mcdc_decisions": [],
            }
        }

        f = io.StringIO()
        with redirect_stdout(f):
            render_patch_report(
                diff_files,
                coverage_matrix,
                base_sha="abc1234567",
                head_sha="def8901234",
                base_branch="main",
                head_branch="patch-1",
                targets_str="libc.test.src.string.memchr_test.__unit__",
            )
        output = f.getvalue()

        self.assertIn("## LLVM-libc Patch Coverage Report", output)
        self.assertIn("100.00%", output)

    def test_render_full_mcdc(self):
        diff_text = """diff --git a/libc/src/string/memchr.cpp b/libc/src/string/memchr.cpp
--- a/libc/src/string/memchr.cpp
+++ b/libc/src/string/memchr.cpp
@@ -10,2 +10,2 @@
+  if (a && b) {
+    return nullptr;
"""
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "libc/src/string/memchr.cpp": {
                "covered": {10, 11},
                "missed": set(),
                "mcdc_decisions": [
                    {
                        "line_start": 10,
                        "line_end": 10,
                        "conditions": [True, True],
                        "covered": 2,
                        "total": 2,
                    }
                ],
            }
        }

        f = io.StringIO()
        with redirect_stdout(f):
            render_patch_report(
                diff_files,
                coverage_matrix,
                base_sha="abc1234567",
                head_sha="def8901234",
                base_branch="main",
                head_branch="patch-1",
                targets_str="libc.test.src.string.memchr_test.__unit__",
            )
        output = f.getvalue()

        self.assertIn("## LLVM-libc MC/DC Patch Coverage Report", output)
        self.assertIn("100.00% MC/DC", output)
        self.assertIn("Decisions (Verified / Total)", output)

    def test_render_partial_mcdc(self):
        diff_text = """diff --git a/libc/src/string/memchr.cpp b/libc/src/string/memchr.cpp
--- a/libc/src/string/memchr.cpp
+++ b/libc/src/string/memchr.cpp
@@ -10,2 +10,2 @@
+  if (a && (b || c)) {
+    return nullptr;
"""
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "libc/src/string/memchr.cpp": {
                "covered": {10, 11},
                "missed": set(),
                "mcdc_decisions": [
                    {
                        "line_start": 10,
                        "line_end": 10,
                        "conditions": [True, False, False],
                        "covered": 1,
                        "total": 3,
                    }
                ],
            }
        }

        f = io.StringIO()
        with redirect_stdout(f):
            render_patch_report(
                diff_files,
                coverage_matrix,
                base_sha="abc1234567",
                head_sha="def8901234",
                base_branch="main",
                head_branch="patch-1",
                targets_str="libc.test.src.string.memchr_test.__unit__",
            )
        output = f.getvalue()

        self.assertIn("33.3% MC/DC", output)
        self.assertIn("C2, C3 unverified", output)


class TestFullReportRendering(unittest.TestCase):
    @patch.dict(os.environ, {"GITHUB_REPOSITORY": "llvm/llvm-project"}, clear=True)
    def test_render_full_report_streamlined(self):
        cov_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/root/llvm-project/libc/src/string/memchr.cpp",
                            "mcdc_records": [
                                [10, 1, 10, 20, 1, 1, 0, 0, 5, [True, True]],
                            ],
                            "summary": {
                                "lines": {"count": 10, "covered": 10, "percent": 100.0},
                                "functions": {"count": 1, "covered": 1, "percent": 100.0},
                                "mcdc": {"count": 2, "covered": 2, "notcovered": 0, "percent": 100.0},
                            },
                        },
                        {
                            "filename": "/root/llvm-project/libc/src/math/sin.cpp",
                            "mcdc_records": [
                                [20, 1, 20, 30, 1, 1, 0, 0, 5, [False, False]],
                            ],
                            "summary": {
                                "lines": {"count": 20, "covered": 15, "percent": 75.0},
                                "functions": {"count": 1, "covered": 1, "percent": 100.0},
                                "mcdc": {"count": 2, "covered": 0, "notcovered": 2, "percent": 0.0},
                            },
                        }
                    ]
                }
            ]
        }

        f = io.StringIO()
        with redirect_stdout(f):
            render_full_report(cov_data)
        output = f.getvalue()

        self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
        self.assertIn("Overall", output)
        self.assertIn("Coverage Breakdown", output)
        self.assertNotIn("Status", output)
        self.assertNotIn("Health", output)
        self.assertNotIn("Safety Priority", output)


if __name__ == "__main__":
    unittest.main()
