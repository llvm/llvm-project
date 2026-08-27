# ====- Unit tests for diff_coverage.py ----------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import unittest
import json
from diff_coverage import (
    DiffParser,
    DiffHunk,
    is_executable_line,
    CoverageJSONParser,
    calculate_patch_statistics,
    format_status_banner,
    format_breakdown_table,
    PatchCoverageSummary,
    FilePatchMetrics
)

class TestExecutableLineHeuristics(unittest.TestCase):
    def test_heuristics_matrix(self):
        matrix = [
            ("int x = 5;", True, "Basic assignment"),
            ("return result;", True, "Return statement"),
            ("if (x > 0) {", True, "Condition start"),
            ("foo();", True, "Function call"),
            ("  // This is a comment", False, "Line comment"),
            ("/* Block comment */", False, "Block comment line"),
            ("#define LIBC_INLINE inline", False, "Preprocessor macro"),
            ("namespace LIBC_NAMESPACE {", False, "Namespace declaration"),
            ("};", False, "Struct/Class terminator"),
            ("  {", False, "Opening scope"),
            ("  }", False, "Closing scope"),
            ("[[maybe_unused]] int x;", True, "Attribute with variable"),
            ("", False, "Empty line"),
        ]
        for input_string, expected, description in matrix:
            with self.subTest(msg=description, input_string=input_string):
                self.assertEqual(is_executable_line(input_string), expected)

class TestDiffParser(unittest.TestCase):
    def test_parse_diff_hunks_and_lines(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,4 @@\n"
            " context_line_1();\n"
            "+added_line_1();\n"
            "+added_line_2();\n"
            " context_line_2();\n"
        )
        hunks_dict = DiffParser.parse(mock_diff)
        self.assertIn("src/math/sin.cpp", hunks_dict)
        hunks = hunks_dict["src/math/sin.cpp"]
        self.assertEqual(len(hunks), 1)
        
        hunk = hunks[0]
        self.assertEqual(hunk.header, "@@ -10,3 +10,4 @@")
        added_lines = [line for line in hunk.lines if line[0] == "+"]
        self.assertEqual(len(added_lines), 2)
        self.assertEqual(added_lines[0], ("+", "added_line_1();", 11))
        self.assertEqual(added_lines[1], ("+", "added_line_2();", 12))

class TestSegmentIntersection(unittest.TestCase):
    def test_calculate_patch_statistics_exact_match(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,2 @@\n"
            " context();\n"
            "+return 0;\n"
        )
        mock_json = {
            "data": [{
                "files": [{
                    "filename": "/workspace/src/math/sin.cpp",
                    "segments": [
                        [10, 0, 1, 1, 1],
                        [11, 0, 1, 1, 1]
                    ],
                    "branches": []
                }]
            }]
        }
        
        diff_files = DiffParser.parse(mock_diff)
        coverage_matrix = CoverageJSONParser.extract_patch_matrix(mock_json, diff_files)
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        
        self.assertEqual(summary.total_lines, 1)
        self.assertEqual(summary.total_covered_lines, 1)
        self.assertEqual(summary.total_missed_lines, 0)
        
    def test_calculate_patch_statistics_missed_line(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,2 @@\n"
            " context();\n"
            "+return 0;\n"
        )
        mock_json = {
            "data": [{
                "files": [{
                    "filename": "/workspace/src/math/sin.cpp",
                    "segments": [
                        [10, 0, 1, 1, 1],
                        [11, 0, 0, 1, 1]
                    ],
                    "branches": []
                }]
            }]
        }
        
        diff_files = DiffParser.parse(mock_diff)
        coverage_matrix = CoverageJSONParser.extract_patch_matrix(mock_json, diff_files)
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        
        self.assertEqual(summary.total_lines, 1)
        self.assertEqual(summary.total_covered_lines, 0)
        self.assertEqual(summary.total_missed_lines, 1)

class TestPatchReportRendering(unittest.TestCase):
    def test_format_status_banner(self):
        summary = PatchCoverageSummary(
            total_covered_lines=50,
            total_missed_lines=50,
            total_mcdc_total_conditions=10,
            total_mcdc_covered_conditions=5,
            files={}
        )
        banner = format_status_banner(summary)
        self.assertIn("### Patch Coverage:", banner)
        self.assertIn("50.00% Line", banner)
        
    def test_format_breakdown_table(self):
        file_stat = FilePatchMetrics(
            file_path="src/math/sin.cpp",
            covered_lines={10, 11},
            missed_lines=set(),
            added_lines={10, 11}
        )
        summary = PatchCoverageSummary(
            total_covered_lines=2,
            total_missed_lines=0,
            files={"src/math/sin.cpp": file_stat}
        )
        report = format_breakdown_table(summary)
        self.assertIn("[`src/math/sin.cpp`]", report)
        self.assertIn("**100.00%**", report)

class TestDiffParserEdgeCases(unittest.TestCase):
    def test_deleted_file(self):
        mock_diff = (
            "diff --git a/src/math/old.cpp b/src/math/old.cpp\n"
            "deleted file mode 100644\n"
            "--- a/src/math/old.cpp\n"
            "+++ /dev/null\n"
            "@@ -1,3 +0,0 @@\n"
            "-deleted_line_1();\n"
            "-deleted_line_2();\n"
        )
        hunks_dict = DiffParser.parse(mock_diff)
        self.assertEqual(len(hunks_dict), 0)

    def test_no_newline_at_eof(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -1,2 +1,3 @@\n"
            " context();\n"
            "+new_line();\n"
            "\\ No newline at end of file\n"
        )
        hunks_dict = DiffParser.parse(mock_diff)
        hunks = hunks_dict["src/math/sin.cpp"]
        added_lines = [line for line in hunks[0].lines if line[0] == "+"]
        self.assertEqual(len(added_lines), 1)

class TestPathResolution(unittest.TestCase):
    def test_fuzzy_path_matching(self):
        diff_files = {"libc/src/math/sin.cpp": []}
        mock_json = {
            "data": [{
                "files": [{
                    "filename": "/home/runner/work/llvm-project/libc/src/math/sin.cpp",
                    "segments": [[10, 0, 1, 1, 1]],
                    "mcdc_records": []
                }]
            }]
        }
        coverage_matrix = CoverageJSONParser.extract_patch_matrix(mock_json, diff_files)
        self.assertIn("libc/src/math/sin.cpp", coverage_matrix)

class TestZeroStateBoundaries(unittest.TestCase):
    def test_empty_json_data(self):
        diff_files = {"src/math/sin.cpp": []}
        mock_json = {"data": []}
        coverage_matrix = CoverageJSONParser.extract_patch_matrix(mock_json, diff_files)
        self.assertIn("src/math/sin.cpp", coverage_matrix)
        self.assertEqual(len(coverage_matrix["src/math/sin.cpp"]["covered"]), 0)

    def test_purely_cosmetic_patch(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,2 @@\n"
            " context();\n"
            "+// Just a comment\n"
        )
        diff_files = DiffParser.parse(mock_diff)
        coverage_matrix = {"src/math/sin.cpp": {"covered": set(), "missed": set(), "mcdc_decisions": []}}
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        self.assertEqual(summary.total_lines, 0)
        self.assertEqual(summary.line_coverage_percentage, 0.0)

class TestMCDCIntersection(unittest.TestCase):
    def test_mcdc_boolean_extraction(self):
        mock_diff = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,2 @@\n"
            " context();\n"
            "+if (a && b) { return 0; }\n"
        )
        mock_json = {
            "data": [{
                "files": [{
                    "filename": "/workspace/src/math/sin.cpp",
                    "segments": [
                        [11, 0, 1, 1, 1]
                    ],
                    "mcdc_records": [
                        [11, 4, 11, 14, 0, 0, 0, 0, 0, [True, False]]
                    ]
                }]
            }]
        }
        diff_files = DiffParser.parse(mock_diff)
        coverage_matrix = CoverageJSONParser.extract_patch_matrix(mock_json, diff_files)
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        
        self.assertEqual(summary.total_mcdc_total_conditions, 2)
        self.assertEqual(summary.total_mcdc_covered_conditions, 1)

if __name__ == '__main__':
    unittest.main()
