# ===- Unit tests for patch coverage ------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

# To run these tests:
# python3 -m unittest patch_coverage_test.py

"""Unit tests for patch coverage analyzer."""

import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from typing import List, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_coverage import (
    CoverageJSONParser,
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


def _make_unified_diff(
    filepath: str,
    added_lines: List[str],
    start_line: int = 1,
    context_lines: Optional[List[str]] = None,
) -> str:
    """Constructs a valid Git unified diff string for testing patch correlation."""
    ctx = context_lines or ["ctx();"]
    old_count = len(ctx)
    new_count = old_count + len(added_lines)
    header = f"@@ -{start_line},{old_count} +{start_line},{new_count} @@"
    lines = [
        f"diff --git a/{filepath} b/{filepath}",
        f"--- a/{filepath}",
        f"+++ b/{filepath}",
        header,
    ]
    for c in ctx:
        lines.append(f" {c}")
    for a in added_lines:
        lines.append(f"+{a}")
    lines.append("")
    return "\n".join(lines)


def _make_cov_file(
    filename: str, segments: list, mcdc_records: Optional[list] = None
) -> dict:
    """Constructs a single file coverage record for llvm-cov JSON export."""
    return {
        "filename": filename,
        "segments": segments,
        "mcdc_records": mcdc_records or [],
    }


def _make_cov_json(filename_or_files, segments=None, mcdc_records=None) -> dict:
    """Constructs a minimal llvm-cov JSON export payload."""
    if isinstance(filename_or_files, list):
        files = filename_or_files
    else:
        files = [_make_cov_file(filename_or_files, segments or [], mcdc_records)]
    return {"data": [{"files": files}]}


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

    def test_executable_statements(self):
        """Statements with assignments, function calls, returns, and control flow are executable."""
        cases = [
            ("int x = 42;", "Variable assignment"),
            ("x += y;", "Compound arithmetic assignment"),
            ("return result;", "Return statement"),
            ("if (x > 0) {", "Branch condition header"),
            ("for (size_t i = 0; i < count; ++i) {", "For loop header"),
            ("do_work(a, b);", "Function call"),
            ("struct Point p = {1, 2};", "Struct variable initialization"),
            ("int x = 42; // assignment", "Statement with trailing line comment"),
            ("int x = 42; /* inline comment */", "Statement with inline block comment"),
        ]
        for line, desc in cases:
            with self.subTest(msg=desc, line=line):
                self.assertTrue(is_executable_line(line))

    def test_structural_syntax_and_braces(self):
        """Standalone braces, access specifiers, and constructor colons are non-executable."""
        cases = [
            ("{", "Opening brace"),
            ("}", "Closing brace"),
            ("};", "Scope terminator"),
            ("public:", "Public access specifier"),
            ("private: // methods", "Private access specifier with comment"),
            (": value_(0) {", "Constructor initializer header"),
        ]
        for line, desc in cases:
            with self.subTest(msg=desc, line=line):
                self.assertFalse(is_executable_line(line))

    def test_comments_and_whitespace(self):
        """Line comments, block comments, blank lines, and commented braces are non-executable."""
        cases = [
            ("// Single line comment", "Single-line comment"),
            ("/* Block comment start", "Block comment start"),
            (" * Continuation line", "Block comment continuation"),
            (" */", "Block comment end"),
            (
                "} // namespace LIBC_NAMESPACE_DECL",
                "Closing brace with namespace comment",
            ),
            ("}; // struct Point", "Scope end with comment"),
            ("{ // begin loop", "Opening brace with comment"),
            ("} /* namespace */", "Closing brace with block comment"),
            ("", "Empty line"),
            ("   ", "Whitespace indentation only"),
        ]
        for line, desc in cases:
            with self.subTest(msg=desc, line=line):
                self.assertFalse(is_executable_line(line))

    def test_declarations_and_preprocessor(self):
        """Includes, namespaces, type aliases, forward declarations, and static asserts are non-executable."""
        cases = [
            ("#include <stddef.h>", "Preprocessor include"),
            ("namespace LIBC_NAMESPACE {", "Namespace definition"),
            ("using size_t = unsigned long;", "Type alias"),
            ("struct ListNode;", "Forward struct declaration"),
            ("enum class Status : uint8_t {", "Enum definition header"),
            (
                'static_assert(sizeof(long) == 8, "msg");',
                "Compile-time static assertion",
            ),
            ("friend class Peer;", "Friend class declaration"),
        ]
        for line, desc in cases:
            with self.subTest(msg=desc, line=line):
                self.assertFalse(is_executable_line(line))


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
        diff_text = textwrap.dedent(
            """\
            diff --git a/src/math/sin.cpp b/src/math/sin.cpp
            --- a/src/math/sin.cpp
            +++ b/src/math/sin.cpp
            @@ -10,3 +10,4 @@
             ctx1();
            -deleted();
            +added1();
            +added2();
            @@ -50,1 +51,2 @@
             ctx2();
            +added3();
            """
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
        diff_text = textwrap.dedent(
            """\
            diff --git a/src/math/new.cpp b/src/math/new.cpp
            new file mode 100644
            --- /dev/null
            +++ b/src/math/new.cpp
            index 0000..1111
            @@ -0,0 +1,1 @@
            +int new_func();
            diff --git a/src/math/old.cpp b/src/math/old.cpp
            --- a/src/math/old.cpp
            +++ /dev/null
            @@ -1,1 +0,0 @@
            -deleted();
            """
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
        diff_text = _make_unified_diff("src/math/f.cpp", ["line();"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "test.diff")
            with open(tmp_path, "w") as f:
                f.write(diff_text)
            parsed = DiffParser.parse(tmp_path)
            self.assertIn("src/math/f.cpp", parsed)


class TestCoverageJSONParser(unittest.TestCase):
    """Tests JSON parsing, segment expansion, MC/DC extraction, and loader safeguards."""

    def test_segment_expansion_and_path_normalization(self):
        """Segments spanning multiple lines must mark each line covered, uncounted must skip."""
        diff_files = {"libc/src/math/sin.cpp": [], "src/math/cos.cpp": []}
        json_data = _make_cov_json(
            [
                _make_cov_file(
                    "/runner/work/llvm-project/libc/src/math/sin.cpp",
                    segments=[
                        [10, 0, 5, 1, 1],
                        [13, 0, 0, 1, 1],
                        [20, 0, 0, 0, 1],
                    ],
                ),
                _make_cov_file("cos.cpp", segments=[[5, 0, 1, 1, 1]]),
                _make_cov_file(
                    "/runner/work/llvm-project/libc/src/math/other.cpp",
                    segments=[[1, 0, 1, 1, 1]],
                ),
            ]
        )
        matrix = CoverageJSONParser.extract_patch_matrix(json_data, diff_files)
        self.assertEqual(matrix["libc/src/math/sin.cpp"]["covered"], {10, 11, 12})
        self.assertEqual(matrix["src/math/cos.cpp"]["covered"], {5})

    def test_mcdc_records_extraction(self):
        """Valid MC/DC records must be extracted; truncated or empty records must be ignored."""
        diff_files = {"src/math/sin.cpp": []}
        json_data = _make_cov_json(
            "/runner/work/llvm-project/libc/src/math/sin.cpp",
            segments=[],
            mcdc_records=[
                [10, 4, 10, 14, 0, 0, 0, 0, 0, [True, False]],
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "cov.json")
            with open(tmp_path, "w") as tmp:
                json.dump({"key": "val"}, tmp)
            self.assertEqual(CoverageJSONParser.load(tmp_path), {"key": "val"})

        stderr_buf = io.StringIO()
        with patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit):
                CoverageJSONParser.load("/nonexistent/cov.json")
        self.assertIn("Error: Failed to parse coverage JSON", stderr_buf.getvalue())


class TestCalculatePatchStatistics(unittest.TestCase):
    """Tests correlating patch lines against coverage segments and MC/DC truth tables."""

    def test_calculate_patch_statistics(self):
        """Covered lines take precedence, uninstrumented files miss, and MC/DC diagnoses unverified."""
        diff_sin = _make_unified_diff(
            "src/math/sin.cpp",
            ["int covered_and_missed = 1;", "if (a && b) return 1;"],
            start_line=10,
        )
        diff_untested = _make_unified_diff(
            "src/math/untested.cpp",
            ["int untested = 1;"],
            start_line=1,
        )
        diff_files = DiffParser.parse(f"{diff_sin}\n{diff_untested}")
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
        self.assertIn("Line 12: C2 unverified", sin_metric.condition_diagnostics[0])
        self.assertEqual(sin_metric.unverified_decision_lines[12], ["C2"])

    def test_non_source_and_comment_files_skipped(self):
        """Test files, documentation, and files with only comment additions must be skipped."""
        diff_test = _make_unified_diff(
            "libc/test/src/math/sin_test.cpp", ["TEST(Foo, Bar) {}"]
        )
        diff_comment = _make_unified_diff(
            "src/math/comment_only.cpp", ["// comment only"]
        )
        diff_files = DiffParser.parse(f"{diff_test}\n{diff_comment}")
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
                "All modified executable lines and boolean conditions achieved full coverage.",
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
                "All **5** modified executable lines were executed, but **1** boolean condition(s) require additional test cases",
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
                "(**1** unexecuted line(s) detected in patch).",
            ),
        ]
        for summary, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, format_status_banner(summary))

    def test_format_metadata_section(self):
        """Metadata section must format targeted tests or return empty on missing arguments."""
        self.assertEqual(format_metadata_section(None, None, None, None), "")
        metadata = format_metadata_section(
            None, None, None, None, targeted_tests_string="libc-math-unit-tests"
        )
        self.assertIn("`libc-math-unit-tests`", metadata)
        self.assertIn("Targeted Tests Executed", metadata)

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
        table = format_breakdown_table(summary)
        self.assertIn("blob/main/libc/src/math/sin.cpp", table)
        self.assertIn("MC/DC Coverage", table)
        self.assertIn("N/A", table)  # strlen has no MC/DC

    def test_format_annotated_diff(self):
        """Annotated diff must output covered, missed, partial MC/DC, non-executable, and context lines."""
        diff_text = _make_unified_diff(
            "src/math/sin.cpp",
            ["covered();", "missed();", "if (a && b) {}", "{\n"],
            start_line=10,
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
        self.assertIn(" ctx();", annotated)
        self.assertIn("+covered();", annotated)
        self.assertIn("!missed();  // <-- UNEXECUTED", annotated)
        self.assertIn("+if (a && b) {}", annotated)
        self.assertIn("+{", annotated)


class TestRenderPatchReportEndToEnd(unittest.TestCase):
    """Tests full Markdown report composition from inputs to stdout."""

    def test_render_empty_diff(self):
        """Empty diff must render coverage notice without failing."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_patch_report(
                {},
                {},
                None,
                None,
                None,
                None,
            )
        self.assertIn(
            "No executable lines were added or modified in this patch.",
            buf.getvalue(),
        )

    def test_render_full_report_with_mcdc(self):
        """Patch report with MC/DC must display the MC/DC report title and full tables."""
        diff_text = _make_unified_diff("src/math/f.cpp", ["return a && b;"])
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
                diff_files,
                coverage_matrix,
                None,
                None,
                None,
                None,
            )
        output = buf.getvalue()
        self.assertIn("## LLVM-libc MC/DC Patch Coverage Report", output)
        self.assertIn("| **MC/DC Coverage** |", output)
        self.assertIn("View Annotated Patch Diff", output)

    def test_render_full_report_line_coverage_only(self):
        """Line-coverage-only reports must omit MC/DC headers and condition columns."""
        diff_text = _make_unified_diff("src/math/sin.cpp", ["int x = 1;"])
        diff_files = DiffParser.parse(diff_text)
        coverage_matrix = {
            "src/math/sin.cpp": {
                "covered": {2},
                "missed": set(),
                "mcdc_decisions": [],
            }
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_patch_report(
                diff_files,
                coverage_matrix,
                None,
                None,
                None,
                None,
                "libc-math-unit-tests",
            )
        output = buf.getvalue()
        self.assertIn("## LLVM-libc Patch Coverage Report", output)
        self.assertNotIn("MC/DC Coverage", output)
        self.assertIn("| **Line Coverage** | **100.00%** |", output)
        self.assertIn("`libc-math-unit-tests`", output)


class TestCommandLineInterfacePatch(unittest.TestCase):
    """Tests CLI execution and file validation safeguards."""

    def test_cli_minimal_arguments(self):
        """CLI must execute successfully when only required diff and JSON files are provided."""
        diff_text = _make_unified_diff("src/math/s.cpp", ["int x = 1;"])
        json_data = _make_cov_json("/workspace/src/math/s.cpp", [[2, 0, 1, 1, 1]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_diff = os.path.join(tmp_dir, "patch.diff")
            path_json = os.path.join(tmp_dir, "cov.json")
            with open(path_diff, "w") as f_diff:
                f_diff.write(diff_text)
            with open(path_json, "w") as f_json:
                json.dump(json_data, f_json)

            buf = io.StringIO()
            with patch.object(sys, "argv", ["diff_coverage.py", path_diff, path_json]):
                with redirect_stdout(buf):
                    main()
            self.assertIn("## LLVM-libc Patch Coverage Report", buf.getvalue())
            self.assertIn("[`src/math/s.cpp`]", buf.getvalue())

    def test_cli_execution_with_mcdc(self):
        """CLI must process MC/DC records and format tables when executed with MC/DC data."""
        diff_text = _make_unified_diff("src/math/sin.cpp", ["if (a && b) return 0;"])
        json_data = _make_cov_json(
            "/workspace/libc/src/math/sin.cpp",
            [[2, 0, 5, 1, 1]],
            mcdc_records=[[2, 5, 2, 20, 0, 0, 0, 0, 0, [True, True]]],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_diff = os.path.join(tmp_dir, "patch.diff")
            path_json = os.path.join(tmp_dir, "cov.json")
            with open(path_diff, "w") as f_diff:
                f_diff.write(diff_text)
            with open(path_json, "w") as f_json:
                json.dump(json_data, f_json)

            buf = io.StringIO()
            with patch.object(
                sys,
                "argv",
                ["diff_coverage.py", path_diff, path_json],
            ):
                with redirect_stdout(buf):
                    main()
            output = buf.getvalue()
            self.assertIn("## LLVM-libc MC/DC Patch Coverage Report", output)
            self.assertIn("blob/main/libc/src/math/sin.cpp", output)
            self.assertIn("MC/DC Coverage", output)
            self.assertIn("Condition Diagnostics", output)

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

    def test_cli_invalid_json_exits_with_error(self):
        """CLI must exit with code 1 when diff exists but coverage JSON is malformed."""
        diff_text = "diff --git a/a b/b\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_diff = os.path.join(tmp_dir, "patch.diff")
            path_json = os.path.join(tmp_dir, "cov.json")
            with open(path_diff, "w") as f_diff:
                f_diff.write(diff_text)
            with open(path_json, "w") as f_json:
                f_json.write("MALFORMED JSON")

            stderr_buf = io.StringIO()
            with patch.object(sys, "argv", ["diff_coverage.py", path_diff, path_json]):
                with patch("sys.stderr", stderr_buf):
                    with self.assertRaises(SystemExit):
                        main()
            self.assertIn("Error: Failed to parse coverage JSON", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
