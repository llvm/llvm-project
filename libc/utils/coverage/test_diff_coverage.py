# ====- Unit tests for diff_coverage.py ----------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

"""Unit tests for diff_coverage.py."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_coverage import (
    DEFAULT_BASE_REPOSITORY,
    DEFAULT_HEAD_REPOSITORY,
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
    main,
    render_patch_report,
)


def _make_cov_json(filename, segments, mcdc_records=None):
    """Constructs a minimal llvm-cov JSON export payload."""
    return {
        "data": [
            {
                "files": [
                    {
                        "filename": filename,
                        "segments": segments,
                        "mcdc_records": mcdc_records or [],
                    }
                ]
            }
        ]
    }


class TestCoverageDataStructures(unittest.TestCase):
    """Tests FilePatchMetrics and PatchCoverageSummary ratio logic and zero-division guards."""

    def test_file_patch_metrics_calculations(self):
        """FilePatchMetrics must correctly compute line and MC/DC ratios or 0.0 on zero totals."""
        empty = FilePatchMetrics(file_path="src/math/sin.cpp")
        self.assertEqual(empty.total_lines, 0)
        self.assertEqual(empty.line_coverage_percentage, 0.0)
        self.assertEqual(empty.mcdc_coverage_percentage, 0.0)

        metrics = FilePatchMetrics(
            file_path="src/math/sin.cpp",
            covered_lines={10, 11, 12},
            missed_lines={13},
            mcdc_covered_conditions=3,
            mcdc_total_conditions=4,
        )
        self.assertEqual(metrics.total_lines, 4)
        self.assertAlmostEqual(metrics.line_coverage_percentage, 75.0, places=2)
        self.assertAlmostEqual(metrics.mcdc_coverage_percentage, 75.0, places=2)

    def test_patch_coverage_summary_aggregation(self):
        """PatchCoverageSummary must correctly aggregate metrics and condition indicators."""
        empty = PatchCoverageSummary()
        self.assertEqual(empty.total_lines, 0)
        self.assertEqual(empty.line_coverage_percentage, 0.0)
        self.assertEqual(empty.mcdc_coverage_percentage, 0.0)
        self.assertFalse(empty.has_mcdc)

        summary = PatchCoverageSummary(
            total_covered_lines=6,
            total_missed_lines=2,
            total_mcdc_covered_conditions=1,
            total_mcdc_total_conditions=2,
        )
        self.assertEqual(summary.total_lines, 8)
        self.assertAlmostEqual(summary.line_coverage_percentage, 75.0, places=2)
        self.assertAlmostEqual(summary.mcdc_coverage_percentage, 50.0, places=2)
        self.assertTrue(summary.has_mcdc)


class TestExecutableLineFiltering(unittest.TestCase):
    """Tests statement heuristics distinguishing executable C++ statements from non-code."""

    def test_statement_heuristics(self):
        """Verifies statement classification across distinct C/C++ syntactic forms."""
        test_cases = [
            ("int x = 42;", True, "Variable assignment"),
            ("x += y;", True, "Compound arithmetic assignment"),
            ("return result;", True, "Return statement"),
            ("if (x > 0) {", True, "Branch condition header"),
            ("for (size_t i = 0; i < count; ++i) {", True, "For loop header"),
            ("do_work(a, b);", True, "Function call"),
            ("struct Point p = {1, 2};", True, "Struct variable assignment"),
            ("int x = 42; // assignment", True, "Statement with line comment"),
            ("int x = 42; /* inline comment */", True, "Statement with block comment"),
            ("// Single line comment", False, "Line comment"),
            ("/* Block comment start", False, "Block comment start"),
            (" * Continuation line", False, "Block comment middle"),
            (" */", False, "Block comment end"),
            (
                "} // namespace LIBC_NAMESPACE_DECL",
                False,
                "Brace with namespace comment",
            ),
            ("}; // struct Point", False, "Scope end with comment"),
            ("{ // begin loop", False, "Opening brace with comment"),
            ("} /* namespace */", False, "Brace with block comment"),
            ("public:", False, "Access specifier"),
            ("private: // methods", False, "Access specifier with comment"),
            (
                'static_assert(sizeof(long) == 8, "msg");',
                False,
                "Compile-time assertion",
            ),
            ("friend class Peer;", False, "Friend declaration"),
            ("{", False, "Opening brace"),
            ("}", False, "Closing brace"),
            ("};", False, "Scope terminator"),
            (": value_(0) {", False, "Constructor initializer header"),
            ("#include <stddef.h>", False, "Preprocessor include"),
            ("namespace LIBC_NAMESPACE {", False, "Namespace definition"),
            ("using size_t = unsigned long;", False, "Type alias"),
            ("struct ListNode;", False, "Forward struct declaration"),
            ("enum class Status : uint8_t {", False, "Enum definition header"),
            ("", False, "Empty line"),
            ("   ", False, "Whitespace line"),
        ]
        for line, expected, desc in test_cases:
            with self.subTest(msg=desc, line=line):
                self.assertEqual(is_executable_line(line), expected)


class TestFormatLineRanges(unittest.TestCase):
    """Tests line number set formatting into concise span representations."""

    def test_formatting_spans(self):
        """Line number sets must format as empty, single, contiguous, or disjoint spans."""
        self.assertEqual(format_line_ranges(set()), "None")
        self.assertEqual(format_line_ranges({42}), "`L42`")
        self.assertEqual(format_line_ranges({10, 11, 12}), "`L10-L12`")
        disjoint = {1, 2, 5, 8, 9, 100}
        self.assertEqual(format_line_ranges(disjoint), "`L1-L2`, `L5`, `L8-L9`, `L100`")


class TestDiffParser(unittest.TestCase):
    """Tests Unified Diff parsing across single/multiple hunks, creations, and deletions."""

    def test_parse_diff_hunks(self):
        """DiffParser must extract added and context lines across hunks while ignoring deletions."""
        diff_text = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,3 +10,4 @@\n"
            " ctx1();\n"
            "-deleted();\n"
            "+added1();\n"
            "+added2();\n"
            "@@ -50,1 +51,2 @@\n"
            " ctx2();\n"
            "+added3();\n"
        )
        parsed = DiffParser.parse(diff_text)
        self.assertIn("src/math/sin.cpp", parsed)
        hunks = parsed["src/math/sin.cpp"]
        self.assertEqual(len(hunks), 2)
        added_hunk1 = [line for line in hunks[0].lines if line[0] == "+"]
        added_hunk2 = [line for line in hunks[1].lines if line[0] == "+"]
        self.assertEqual(added_hunk1, [("+", "added1();", 11), ("+", "added2();", 12)])
        self.assertEqual(added_hunk2, [("+", "added3();", 52)])

    def test_parse_special_files(self):
        """Newly created files must start at line 1, deleted files and headers must be skipped."""
        diff_text = (
            "diff --git a/src/math/new.cpp b/src/math/new.cpp\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/src/math/new.cpp\n"
            "index 0000..1111\n"
            "@@ -0,0 +1,1 @@\n"
            "+int new_func();\n"
            "diff --git a/src/math/old.cpp b/src/math/old.cpp\n"
            "--- a/src/math/old.cpp\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-deleted();\n"
        )
        parsed = DiffParser.parse(diff_text)
        self.assertIn("src/math/new.cpp", parsed)
        self.assertNotIn("src/math/old.cpp", parsed)
        self.assertEqual(
            parsed["src/math/new.cpp"][0].lines[0], ("+", "int new_func();", 1)
        )
        self.assertEqual(DiffParser.parse(""), {})

    def test_parse_from_disk_file(self):
        """DiffParser must successfully read and parse diff files from disk."""
        diff_text = "diff --git a/a b/b\n+++ b/src/math/f.cpp\n@@ -1,1 +1,2 @@\n ctx();\n+line();\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as tmp:
            tmp.write(diff_text)
            tmp_path = tmp.name
        try:
            parsed = DiffParser.parse(tmp_path)
            self.assertIn("src/math/f.cpp", parsed)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestCoverageJSONParser(unittest.TestCase):
    """Tests JSON parsing, segment expansion, MC/DC extraction, and loader safeguards."""

    def test_segment_expansion_and_path_normalization(self):
        """Segments spanning multiple lines must mark each line covered, uncounted must skip."""
        diff_files = {"libc/src/math/sin.cpp": [], "src/math/cos.cpp": []}
        json_data = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/runner/work/llvm-project/libc/src/math/sin.cpp",
                            "segments": [
                                [10, 0, 5, 1, 1],
                                [13, 0, 0, 1, 1],
                                [20, 0, 0, 0, 1],
                            ],
                            "mcdc_records": [],
                        },
                        {
                            # Tests target_path.endswith("/" + file_name)
                            "filename": "cos.cpp",
                            "segments": [[5, 0, 1, 1, 1]],
                            "mcdc_records": [],
                        },
                        {
                            # Unmatched file: tests continue branch
                            "filename": "/runner/work/llvm-project/libc/src/math/other.cpp",
                            "segments": [[1, 0, 1, 1, 1]],
                            "mcdc_records": [],
                        },
                    ]
                }
            ]
        }
        matrix = CoverageJSONParser.extract_patch_matrix(json_data, diff_files)
        covered = matrix["libc/src/math/sin.cpp"]["covered"]
        missed = matrix["libc/src/math/sin.cpp"]["missed"]
        self.assertEqual(covered, {10, 11, 12})
        self.assertIn(13, missed)
        self.assertIn(19, missed)
        self.assertNotIn(20, missed)

    def test_mcdc_records_extraction(self):
        """Valid MC/DC records must be extracted; truncated or empty records must be ignored."""
        diff_files = {"src/math/sin.cpp": []}
        json_data = _make_cov_json(
            filename="/workspace/src/math/sin.cpp",
            segments=[[10, 0, 1, 1, 1]],
            mcdc_records=[
                [10, 4, 10, 14, 0, 0, 0, 0, 0, [True, False]],  # Valid
                [11, 4, 11, 14],  # Truncated (< 10)
                [12, 4, 12, 14, 0, 0, 0, 0, 0, []],  # Empty condition vector
            ],
        )
        matrix = CoverageJSONParser.extract_patch_matrix(json_data, diff_files)
        decisions = matrix["src/math/sin.cpp"]["mcdc_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["line_start"], 10)
        self.assertEqual(decisions[0]["covered"], 1)
        self.assertEqual(decisions[0]["total"], 2)

    def test_load_and_empty_payload_safeguards(self):
        """CoverageJSONParser must load valid JSON, fallback on empty data, and exit on error."""
        diff_files = {"src/math/sin.cpp": []}
        empty_matrix = CoverageJSONParser.extract_patch_matrix({}, diff_files)
        self.assertEqual(len(empty_matrix["src/math/sin.cpp"]["covered"]), 0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"key": "val"}, tmp)
            tmp_path = tmp.name
        try:
            self.assertEqual(CoverageJSONParser.load(tmp_path), {"key": "val"})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        stderr_buf = io.StringIO()
        with patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit):
                CoverageJSONParser.load("/nonexistent/cov.json")
        self.assertIn("Error: Failed to parse coverage JSON", stderr_buf.getvalue())


class TestCalculatePatchStatistics(unittest.TestCase):
    """Tests correlating patch lines against coverage segments and MC/DC truth tables."""

    def test_calculate_patch_statistics(self):
        """Covered lines take precedence, uninstrumented files miss, and MC/DC diagnoses unverified."""
        diff_text = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "--- a/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,2 +10,3 @@\n"
            " ctx();\n"
            "+int covered_and_missed = 1;\n"
            "+if (a && b) return 1;\n"
            "diff --git a/src/math/untested.cpp b/src/math/untested.cpp\n"
            "--- a/src/math/untested.cpp\n"
            "+++ b/src/math/untested.cpp\n"
            "@@ -1,1 +1,2 @@\n"
            " ctx();\n"
            "+int untested = 1;\n"
        )
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "src/math/sin.cpp": {
                "covered": {11, 12},
                "missed": {11},  # Covered takes precedence
                "mcdc_decisions": [
                    {
                        "line_start": 12,
                        "line_end": 12,
                        "conditions": [True, False],
                        "covered": 1,
                        "total": 2,
                    }
                ],
            },
            "src/math/untested.cpp": {
                "covered": set(),
                "missed": set(),
                "mcdc_decisions": [],
            },
        }
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        self.assertEqual(summary.total_lines, 3)
        self.assertEqual(summary.total_covered_lines, 2)
        self.assertEqual(summary.total_missed_lines, 1)
        self.assertEqual(summary.total_mcdc_covered_conditions, 1)
        self.assertEqual(summary.total_mcdc_total_conditions, 2)

        sin_metric = summary.files["src/math/sin.cpp"]
        self.assertIn(
            "1/2 verified (C2 unverified)", sin_metric.condition_diagnostics[0]
        )
        self.assertEqual(sin_metric.unverified_decision_lines[12], ["C2"])

    def test_non_source_and_comment_files_skipped(self):
        """Test files, documentation, and files with only comment additions must be skipped."""
        diff_text = (
            "diff --git a/libc/test/src/math/sin_test.cpp b/libc/test/src/math/sin_test.cpp\n"
            "+++ b/libc/test/src/math/sin_test.cpp\n"
            "@@ -1,1 +1,2 @@\n"
            "+TEST(Foo, Bar) {}\n"
            "diff --git a/src/math/comment_only.cpp b/src/math/comment_only.cpp\n"
            "+++ b/src/math/comment_only.cpp\n"
            "@@ -1,1 +1,2 @@\n"
            "+// comment only\n"
        )
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "libc/test/src/math/sin_test.cpp": {
                "covered": set(),
                "missed": set(),
                "mcdc_decisions": [],
            },
            "src/math/comment_only.cpp": {
                "covered": set(),
                "missed": set(),
                "mcdc_decisions": [],
            },
        }
        summary = calculate_patch_statistics(diff_files, coverage_matrix)
        self.assertEqual(summary.total_lines, 0)
        self.assertEqual(len(summary.files), 0)


class TestPatchReportFormatting(unittest.TestCase):
    """Tests Markdown formatting across status banners, metadata, tables, and annotated diffs."""

    def test_format_status_banner_variants(self):
        """Verifies phrasing across all 5 status banner operational conditions."""
        cases = [
            (
                PatchCoverageSummary(total_covered_lines=5, total_missed_lines=0),
                "All **5** newly added",
            ),
            (
                PatchCoverageSummary(
                    total_covered_lines=5,
                    total_missed_lines=0,
                    total_mcdc_covered_conditions=2,
                    total_mcdc_total_conditions=2,
                    total_decisions_count=1,
                    fully_verified_decisions=1,
                ),
                "All **5** executable lines and **2** boolean conditions",
            ),
            (
                PatchCoverageSummary(
                    total_covered_lines=5,
                    total_missed_lines=0,
                    total_mcdc_covered_conditions=1,
                    total_mcdc_total_conditions=2,
                    total_decisions_count=1,
                    fully_verified_decisions=0,
                ),
                "Executed **5 / 5** lines. **1 / 2** boolean conditions",
            ),
            (
                PatchCoverageSummary(total_covered_lines=4, total_missed_lines=1),
                "Executed **4 / 5** lines (**1** unexecuted",
            ),
            (
                PatchCoverageSummary(
                    total_covered_lines=4,
                    total_missed_lines=1,
                    total_mcdc_covered_conditions=1,
                    total_mcdc_total_conditions=2,
                    total_decisions_count=1,
                    fully_verified_decisions=0,
                ),
                "(**1** unexecuted lines detected in patch).",
            ),
        ]
        for summary, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, format_status_banner(summary))

    def test_format_metadata_section(self):
        """Metadata section must format commits, tests, or return empty on missing arguments."""
        metadata = format_metadata_section(
            "1111111", "2222222", "main", "patch", "test_target"
        )
        self.assertIn("Base Branch", metadata)
        self.assertIn("`test_target`", metadata)
        self.assertEqual(format_metadata_section(None, None, None, None), "")

    def test_format_breakdown_table(self):
        """Breakdown tables must render line/MCDC stats and normalize paths to libc/ on GitHub."""
        file_mcdc = FilePatchMetrics(
            file_path="src/math/sin.cpp",
            covered_lines={10},
            missed_lines=set(),
            added_lines={10},
            mcdc_covered_conditions=2,
            mcdc_total_conditions=2,
            decisions_verified=1,
            decisions_total=1,
            condition_diagnostics=["`L10`: 2/2 verified"],
        )
        file_no_mcdc = FilePatchMetrics(
            file_path="src/string/strlen.cpp",
            covered_lines={20},
            missed_lines={21},
            added_lines={20, 21},
        )
        summary = PatchCoverageSummary(
            total_covered_lines=2,
            total_missed_lines=1,
            total_mcdc_covered_conditions=2,
            total_mcdc_total_conditions=2,
            fully_verified_decisions=1,
            total_decisions_count=1,
            files={
                "src/math/sin.cpp": file_mcdc,
                "src/string/strlen.cpp": file_no_mcdc,
            },
        )
        table = format_breakdown_table(summary, head_commit_sha="abcd123")
        self.assertIn("blob/abcd123/libc/src/math/sin.cpp", table)
        self.assertIn("MC/DC Conditions", table)
        self.assertIn("N/A | N/A", table)  # strlen has no MC/DC

    def test_format_annotated_diff(self):
        """Annotated diff must output covered, missed, partial MC/DC, non-executable, and context lines."""
        diff_text = (
            "diff --git a/src/math/sin.cpp b/src/math/sin.cpp\n"
            "+++ b/src/math/sin.cpp\n"
            "@@ -10,4 +10,5 @@\n"
            " ctx();\n"
            "+covered();\n"
            "+missed();\n"
            "+if (a && b) {}\n"
            "+{\n"
        )
        diff_files = DiffParser.parse(diff_text)
        file_metrics = FilePatchMetrics(
            file_path="src/math/sin.cpp",
            covered_lines={11, 13},
            missed_lines={12},
            unverified_decision_lines={13: ["C2"]},
        )
        summary = PatchCoverageSummary(files={"src/math/sin.cpp": file_metrics})
        annotated = format_annotated_diff(summary, diff_files)
        self.assertIn("  ctx();", annotated)
        self.assertIn("+ covered();", annotated)
        self.assertIn("- missed();  // [MISSED]", annotated)
        self.assertIn("! if (a && b) {}  // [PARTIAL MC/DC: C2 unverified]", annotated)
        self.assertIn("  {", annotated)


class TestRenderPatchReportEndToEnd(unittest.TestCase):
    """Tests full Markdown report composition from inputs to stdout."""

    def test_render_empty_diff(self):
        """Empty diff must render coverage notice without failing."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_patch_report({}, {}, "1111", "2222", "main", "feature")
        self.assertIn(
            "No executable lines were added or modified in this patch.", buf.getvalue()
        )

    def test_render_full_report_with_mcdc(self):
        """Patch report with MC/DC must display the MC/DC report title and full tables."""
        diff_text = "diff --git a/src/math/f.cpp b/src/math/f.cpp\n+++ b/src/math/f.cpp\n@@ -1,1 +1,2 @@\n ctx();\n+return a && b;\n"
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "src/math/f.cpp": {
                "covered": {2},
                "missed": set(),
                "mcdc_decisions": [
                    {
                        "line_start": 2,
                        "line_end": 2,
                        "conditions": [True, True],
                        "covered": 2,
                        "total": 2,
                    }
                ],
            }
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_patch_report(
                diff_files, coverage_matrix, "1111", "2222", "main", "feature"
            )
        output = buf.getvalue()
        self.assertIn("## LLVM-libc MC/DC Patch Coverage Report", output)
        self.assertIn(
            "### Patch Coverage: **100.00% Line** | **100.00% MC/DC**", output
        )
        self.assertIn("View Annotated Patch Diff", output)


class TestCommandLineInterfaceDiff(unittest.TestCase):
    """Tests CLI execution and file validation safeguards."""

    def test_cli_execution(self):
        """CLI must read diff and JSON files from disk and print the report to stdout."""
        diff_text = "diff --git a/src/math/s.cpp b/src/math/s.cpp\n+++ b/src/math/s.cpp\n@@ -1,1 +1,2 @@\n ctx();\n+int x = 1;\n"
        json_data = _make_cov_json("/workspace/src/math/s.cpp", [[2, 0, 1, 1, 1]])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", delete=False
        ) as f_diff:
            f_diff.write(diff_text)
            path_diff = f_diff.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f_json:
            json.dump(json_data, f_json)
            path_json = f_json.name

        try:
            buf = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "diff_coverage.py",
                    path_diff,
                    path_json,
                    "111",
                    "222",
                    "m",
                    "f",
                    "target",
                ],
            ):
                with redirect_stdout(buf):
                    main()
            self.assertIn("## LLVM-libc Patch Coverage Report", buf.getvalue())
            self.assertIn("[`src/math/s.cpp`]", buf.getvalue())
        finally:
            if os.path.exists(path_diff):
                os.remove(path_diff)
            if os.path.exists(path_json):
                os.remove(path_json)

    def test_cli_missing_files_exit(self):
        """CLI must exit with code 1 when diff file or JSON file is missing."""
        stderr_buf = io.StringIO()
        with patch.object(
            sys, "argv", ["diff_coverage.py", "/missing.diff", "/missing.json"]
        ):
            with patch("sys.stderr", stderr_buf):
                with self.assertRaises(SystemExit):
                    main()
        self.assertIn("Error: Diff file not found", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
