# ===- Unit tests for full codebase coverage ----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

# To run these tests:
# python3 -m unittest full_coverage_test.py

"""Unit tests for codebase_coverage.py."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from typing import List, Optional
from unittest.mock import patch

# Ensure libc/utils/coverage is in sys.path when running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codebase_coverage import (
    DirectoryCoverageMetrics,
    FullCoverageSummary,
    extract_full_coverage_statistics,
    format_directory_breakdown_table,
    format_global_summary_table,
    format_overview_callout,
    main,
    render_full_report,
)


def _make_file_entry(
    filename: str,
    lines_total: int,
    lines_covered: int,
    func_total: int = 1,
    func_covered: int = 1,
    mcdc_total: int = 0,
    mcdc_covered: int = 0,
    mcdc_records: Optional[list] = None,
) -> dict:
    """Constructs a single file coverage record for llvm-cov JSON export."""
    entry = {
        "filename": filename,
        "summary": {
            "lines": {"count": lines_total, "covered": lines_covered},
            "functions": {"count": func_total, "covered": func_covered},
        },
    }
    if mcdc_total > 0 or mcdc_records:
        entry["summary"]["mcdc"] = {"count": mcdc_total, "covered": mcdc_covered}
    if mcdc_records is not None:
        entry["mcdc_records"] = mcdc_records
    return entry


def _make_codebase_payload(files: List[dict]) -> dict:
    """Wraps file coverage entries in the top-level llvm-cov JSON export structure."""
    return {"data": [{"files": files}]}


class TestDirectoryCoverageMetrics(unittest.TestCase):
    """Tests DirectoryCoverageMetrics mathematical operations and property safeguards."""

    def test_zero_totals_return_zero_percentages(self):
        """Zero totals must safely evaluate to 0.0 without ZeroDivisionError."""
        metrics = DirectoryCoverageMetrics(name="empty")
        self.assertEqual(metrics.line_pct, 0.0)
        self.assertEqual(metrics.func_pct, 0.0)
        self.assertEqual(metrics.mcdc_pct, 0.0)
        self.assertEqual(metrics.decisions_pct, 0.0)
        self.assertEqual(metrics.missed_lines, 0)

    def test_percentage_calculations(self):
        """Percentages must correctly compute ratios across lines, functions, and MC/DC."""
        metrics = DirectoryCoverageMetrics(
            name="src/math",
            lines_cov=75,
            lines_tot=100,
            func_cov=3,
            func_tot=4,
            mcdc_cov=7,
            mcdc_tot=10,
            decisions_tot=5,
            decisions_full=4,
        )
        self.assertAlmostEqual(metrics.line_pct, 75.0, places=2)
        self.assertAlmostEqual(metrics.func_pct, 75.0, places=2)
        self.assertAlmostEqual(metrics.mcdc_pct, 70.0, places=2)
        self.assertAlmostEqual(metrics.decisions_pct, 80.0, places=2)
        self.assertEqual(metrics.missed_lines, 25)

    def test_missed_lines_clamping(self):
        """Missed lines must clamp to 0 if covered lines exceed total lines."""
        metrics = DirectoryCoverageMetrics(
            name="src/clamped", lines_cov=120, lines_tot=100
        )
        self.assertEqual(metrics.missed_lines, 0)

    def test_all_zero_metrics(self):
        """Default initialized metrics object must have zero values and empty name."""
        metrics = DirectoryCoverageMetrics()
        self.assertEqual(metrics.name, "")
        self.assertEqual(metrics.lines_cov, 0)
        self.assertEqual(metrics.lines_tot, 0)
        self.assertEqual(metrics.decisions_pct, 0.0)


class TestFullCoverageSummary(unittest.TestCase):
    """Tests FullCoverageSummary attributes and condition indicators."""

    def test_has_mcdc_property(self):
        """has_mcdc must reflect whether global MC/DC conditions exist."""
        without_mcdc = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(mcdc_tot=0)
        )
        self.assertFalse(without_mcdc.has_mcdc)

        with_mcdc = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(mcdc_tot=12)
        )
        self.assertTrue(with_mcdc.has_mcdc)

    def test_default_initialization(self):
        """Default FullCoverageSummary must initialize empty directories dictionary."""
        summary = FullCoverageSummary()
        self.assertFalse(summary.has_mcdc)
        self.assertEqual(len(summary.directories), 0)


class TestExtractFullCoverageStatistics(unittest.TestCase):
    """Tests JSON extraction, file path filtering, directory grouping, and MC/DC aggregation."""

    def test_empty_or_invalid_payloads_return_none(self):
        """Missing or malformed payload structures must return None."""
        test_cases = [
            ({}, "Empty dictionary"),
            ({"data": []}, "Empty data array"),
            ({"data": [{}]}, "Data array without files key"),
            ({"data": [{"files": []}]}, "Empty files array"),
        ]
        for payload, description in test_cases:
            with self.subTest(msg=description):
                self.assertIsNone(extract_full_coverage_statistics(payload))

    def test_all_files_zero_lines_returns_none(self):
        """Payload containing only files with zero total lines must return None."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/empty.cpp",
                    lines_total=0,
                    lines_covered=0,
                    func_total=0,
                    func_covered=0,
                )
            ]
        )
        self.assertIsNone(extract_full_coverage_statistics(payload))

    def test_all_files_excluded_returns_none(self):
        """Payload containing only test or utility files must return None."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/test/src/math/sin_test.cpp",
                    lines_total=100,
                    lines_covered=100,
                ),
                _make_file_entry(
                    "/workspace/libc/utils/MPFRWrapper/MPFRUtils.cpp",
                    lines_total=200,
                    lines_covered=200,
                    func_total=2,
                    func_covered=2,
                ),
            ]
        )
        self.assertIsNone(extract_full_coverage_statistics(payload))

    def test_file_path_filtering(self):
        """Test and utility directories must be excluded from codebase coverage."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=100,
                    lines_covered=80,
                    func_total=2,
                    func_covered=2,
                ),
                _make_file_entry(
                    "/workspace/libc/test/src/math/sin_test.cpp",
                    lines_total=500,
                    lines_covered=500,
                    func_total=5,
                    func_covered=5,
                ),
                _make_file_entry(
                    "/workspace/libc/utils/MPFRWrapper/MPFRUtils.cpp",
                    lines_total=300,
                    lines_covered=300,
                    func_total=4,
                    func_covered=4,
                ),
                _make_file_entry(
                    "/workspace/libc/include/llvm-libc-types/size_t.h",
                    lines_total=50,
                    lines_covered=50,
                ),
                _make_file_entry(
                    "/workspace/libc/src/empty.cpp",
                    lines_total=0,
                    lines_covered=0,
                    func_total=0,
                    func_covered=0,
                ),
            ]
        )
        summary = extract_full_coverage_statistics(payload)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.global_stats.lines_tot, 100)
        self.assertEqual(summary.global_stats.lines_cov, 80)
        self.assertEqual(summary.global_stats.func_tot, 2)
        self.assertEqual(summary.global_stats.func_cov, 2)
        self.assertIn("src/math", summary.directories)
        self.assertEqual(len(summary.directories), 1)

    def test_directory_bucketing_and_nested_paths(self):
        """Files within identical top-level directories or deep subpaths must aggregate properly."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=100,
                    lines_covered=60,
                    func_total=2,
                    func_covered=1,
                ),
                _make_file_entry(
                    "/workspace/libc/src/math/cos.cpp",
                    lines_total=80,
                    lines_covered=80,
                    func_total=2,
                    func_covered=2,
                ),
                _make_file_entry(
                    "/workspace/libc/src/string/memory_utils/op_builtin.cpp",
                    lines_total=120,
                    lines_covered=100,
                    func_total=4,
                    func_covered=3,
                ),
            ]
        )
        summary = extract_full_coverage_statistics(payload)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.global_stats.lines_tot, 300)
        self.assertEqual(summary.global_stats.lines_cov, 240)

        self.assertIn("src/math", summary.directories)
        math_dir = summary.directories["src/math"]
        self.assertEqual(math_dir.lines_tot, 180)
        self.assertEqual(math_dir.lines_cov, 140)
        self.assertEqual(math_dir.func_tot, 4)
        self.assertEqual(math_dir.func_cov, 3)

        self.assertIn("src/string", summary.directories)
        str_dir = summary.directories["src/string"]
        self.assertEqual(str_dir.lines_tot, 120)
        self.assertEqual(str_dir.lines_cov, 100)

    def test_mcdc_records_aggregation_and_decision_tracking(self):
        """MC/DC records must be parsed for total conditions and full decision verification."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/fma.cpp",
                    lines_total=50,
                    lines_covered=50,
                    func_total=1,
                    func_covered=1,
                    mcdc_total=4,
                    mcdc_covered=3,
                    mcdc_records=[
                        [10, 5, 10, 20, 0, 0, 0, 0, 0, [True, True]],
                        [25, 5, 25, 25, 0, 0, 0, 0, 0, [True, False]],
                        [30, 5, 30, 20],  # Malformed: ignored
                    ],
                )
            ]
        )
        summary = extract_full_coverage_statistics(payload)
        self.assertIsNotNone(summary)
        self.assertTrue(summary.has_mcdc)
        self.assertEqual(summary.global_stats.mcdc_tot, 4)
        self.assertEqual(summary.global_stats.mcdc_cov, 3)
        self.assertEqual(summary.global_stats.decisions_tot, 2)
        self.assertEqual(summary.global_stats.decisions_full, 1)

    def test_malformed_and_empty_mcdc_records_ignored(self):
        """Malformed, non-list, or empty condition vectors must not count as valid decisions."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/exp.cpp",
                    lines_total=50,
                    lines_covered=50,
                    func_total=1,
                    func_covered=1,
                    mcdc_total=2,
                    mcdc_covered=2,
                    mcdc_records=[
                        [10, 5, 10, 20, 0, 0, 0, 0, 0, [True]],
                        [20, 5, 20, 20, 0, 0, 0, 0, 0, []],
                        [30, 5, 30, 20, 0, 0, 0, 0, 0, None],
                        [40, 5, 40, 20],
                    ],
                )
            ]
        )
        summary = extract_full_coverage_statistics(payload)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.global_stats.decisions_tot, 1)
        self.assertEqual(summary.global_stats.decisions_full, 1)


class TestCodebaseReportFormatting(unittest.TestCase):
    """Tests Markdown rendering of callouts, summary tables, and directory breakdowns."""

    def test_format_overview_callout_without_mcdc(self):
        """Callout without MC/DC must render line coverage and artifacts instructions."""
        metrics = DirectoryCoverageMetrics(name="global", lines_tot=1000, lines_cov=850)
        summary = FullCoverageSummary(global_stats=metrics)
        callout = format_overview_callout(summary)

        self.assertIn("### Overall Codebase Coverage: **85.00%**", callout)
        self.assertIn("Tested **850 / 1,000** executable lines", callout)
        self.assertIn("Artifacts", callout)
        self.assertIn("HTML Coverage Report", callout)
        self.assertNotIn("MC/DC", callout)

    def test_format_overview_callout_with_mcdc(self):
        """Callout with MC/DC must render both line and condition coverage metrics."""
        metrics = DirectoryCoverageMetrics(
            name="global",
            lines_tot=2000,
            lines_cov=1800,
            mcdc_tot=50,
            mcdc_cov=45,
            decisions_tot=20,
            decisions_full=18,
        )
        summary = FullCoverageSummary(global_stats=metrics)
        callout = format_overview_callout(summary)

        self.assertIn(
            "### Overall Codebase Coverage: **90.00% Line** | **90.00% MC/DC**",
            callout,
        )
        self.assertIn("Tested **1,800 / 2,000** executable lines", callout)
        self.assertIn("and **45 / 50** boolean conditions", callout)
        self.assertIn("across **20** decisions.", callout)
        self.assertIn("Artifacts", callout)

    def test_format_global_summary_table_without_mcdc(self):
        """Global table without MC/DC must display lines and functions only."""
        metrics = DirectoryCoverageMetrics(
            name="global",
            lines_tot=500,
            lines_cov=400,
            func_tot=50,
            func_cov=40,
        )
        summary = FullCoverageSummary(global_stats=metrics)
        table = format_global_summary_table(summary)

        self.assertIn("| **Executable Lines** | 400 | 500 | **80.00%** |", table)
        self.assertIn("| **Functions** | 40 | 50 | **80.00%** |", table)
        self.assertNotIn("MC/DC", table)

    def test_format_global_summary_table_with_mcdc(self):
        """Global table with MC/DC must include condition independence and decision verification."""
        metrics = DirectoryCoverageMetrics(
            name="global",
            lines_tot=1000,
            lines_cov=900,
            func_tot=100,
            func_cov=95,
            mcdc_tot=80,
            mcdc_cov=60,
            decisions_tot=30,
            decisions_full=25,
        )
        summary = FullCoverageSummary(global_stats=metrics)
        table = format_global_summary_table(summary)

        self.assertIn(
            "| **MC/DC Condition Independence** | 60 | 80 | **75.00%** |", table
        )
        self.assertIn("| **Fully Verified Decisions** | 25 | 30 | **83.33%** |", table)

    def test_format_directory_breakdown_table_alphabetical_sorting(self):
        """Directory breakdown table must sort directory names alphabetically."""
        dir_string = DirectoryCoverageMetrics(
            name="src/string", lines_tot=100, lines_cov=100
        )
        dir_math = DirectoryCoverageMetrics(
            name="src/math", lines_tot=100, lines_cov=80
        )
        dir_ctype = DirectoryCoverageMetrics(
            name="src/ctype", lines_tot=100, lines_cov=90
        )

        summary = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(lines_tot=300, lines_cov=270),
            directories={
                "src/string": dir_string,
                "src/math": dir_math,
                "src/ctype": dir_ctype,
            },
        )
        table = format_directory_breakdown_table(summary)

        idx_ctype = table.find("`libc/src/ctype`")
        idx_math = table.find("`libc/src/math`")
        idx_string = table.find("`libc/src/string`")

        self.assertTrue(0 <= idx_ctype < idx_math < idx_string)

    def test_format_directory_breakdown_table_without_mcdc(self):
        """Directory table without MC/DC must show line and function coverage columns."""
        dir_math = DirectoryCoverageMetrics(
            name="src/math", lines_tot=100, lines_cov=80, func_tot=2, func_cov=2
        )
        summary = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(lines_tot=100, lines_cov=80),
            directories={"src/math": dir_math},
        )
        table = format_directory_breakdown_table(summary)
        self.assertNotIn("MC/DC Conditions", table)
        self.assertIn("`libc/src/math` | **80.00%** | 100.00% | 100 | 20 |", table)

    def test_format_directory_breakdown_table_with_mcdc(self):
        """Directory table with MC/DC must show MC/DC condition columns."""
        dir_math = DirectoryCoverageMetrics(
            name="src/math",
            lines_tot=100,
            lines_cov=80,
            mcdc_tot=10,
            mcdc_cov=8,
            decisions_tot=4,
            decisions_full=3,
        )
        summary = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(
                lines_tot=100, lines_cov=80, mcdc_tot=10, mcdc_cov=8
            ),
            directories={"src/math": dir_math},
        )
        table = format_directory_breakdown_table(summary)

        self.assertIn("MC/DC Conditions", table)
        self.assertIn("**80.0%** (8/10)", table)

    def test_format_directory_breakdown_table_mixed_mcdc(self):
        """Directories without MC/DC records in an MC/DC run must display N/A."""
        dir_math = DirectoryCoverageMetrics(
            name="src/math",
            lines_tot=100,
            lines_cov=80,
            mcdc_tot=10,
            mcdc_cov=8,
            decisions_tot=4,
            decisions_full=3,
        )
        dir_ctype = DirectoryCoverageMetrics(
            name="src/ctype",
            lines_tot=50,
            lines_cov=50,
            mcdc_tot=0,
            mcdc_cov=0,
            decisions_tot=0,
            decisions_full=0,
        )
        summary = FullCoverageSummary(
            global_stats=DirectoryCoverageMetrics(
                lines_tot=150, lines_cov=130, mcdc_tot=10, mcdc_cov=8
            ),
            directories={"src/math": dir_math, "src/ctype": dir_ctype},
        )
        table = format_directory_breakdown_table(summary)
        self.assertIn("`libc/src/math` | **80.0%** (8/10) | 3 / 4", table)
        self.assertIn("`libc/src/ctype` | N/A | N/A", table)


class TestRenderFullReportEndToEnd(unittest.TestCase):
    """Tests full Markdown report composition from JSON payload to stdout."""

    def test_render_empty_payload_fallback(self):
        """Empty payload must output fallback message without crashing."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_full_report({})
        output = buf.getvalue()
        self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
        self.assertIn("### No Coverage Data Detected", output)
        self.assertIn(
            "The test execution completed but no coverage profiles were exported.",
            output,
        )

    def test_render_complete_report_without_mcdc(self):
        """Valid report without MC/DC must render line coverage and summary tables."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=100,
                    lines_covered=90,
                    func_total=2,
                    func_covered=2,
                )
            ]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_full_report(payload)
        output = buf.getvalue()

        self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
        self.assertIn("### Overall Codebase Coverage: **90.00%**", output)
        self.assertIn("`libc/src/math`", output)
        self.assertNotIn("MC/DC", output)

    def test_render_complete_report_with_mcdc(self):
        """Valid report with MC/DC must render full callouts, global table, and directory table."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=100,
                    lines_covered=90,
                    func_total=2,
                    func_covered=2,
                    mcdc_total=6,
                    mcdc_covered=6,
                    mcdc_records=[[10, 5, 10, 20, 0, 0, 0, 0, 0, [True, True]]],
                )
            ]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            render_full_report(payload)
        output = buf.getvalue()

        self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
        self.assertIn("### Overall Codebase Coverage:", output)
        self.assertIn("### Overall", output)
        self.assertIn("### Coverage Breakdown", output)
        self.assertIn("`libc/src/math`", output)


class TestCommandLineInterface(unittest.TestCase):
    """Tests CLI invocation, arguments parsing, and file handling."""

    def test_cli_execution(self):
        """CLI must read JSON coverage file from disk and write report to stdout."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=50,
                    lines_covered=40,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "cov.json")
            with open(tmp_path, "w") as tmp_file:
                json.dump(payload, tmp_file)

            buf = io.StringIO()
            with patch.object(sys, "argv", ["codebase_coverage.py", tmp_path]):
                with redirect_stdout(buf):
                    main()
            output = buf.getvalue()
            self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
            self.assertIn("`libc/src/math`", output)

    def test_cli_execution_with_mcdc(self):
        """CLI must render MC/DC breakdown tables when payload includes MC/DC records."""
        payload = _make_codebase_payload(
            [
                _make_file_entry(
                    "/workspace/libc/src/math/sin.cpp",
                    lines_total=60,
                    lines_covered=50,
                    func_total=2,
                    func_covered=2,
                    mcdc_total=4,
                    mcdc_covered=4,
                    mcdc_records=[[10, 5, 10, 20, 0, 0, 0, 0, 0, [True, True]]],
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "cov.json")
            with open(tmp_path, "w") as tmp_file:
                json.dump(payload, tmp_file)

            buf = io.StringIO()
            with patch.object(sys, "argv", ["codebase_coverage.py", tmp_path]):
                with redirect_stdout(buf):
                    main()
            output = buf.getvalue()
            self.assertIn("## LLVM-libc Full Codebase Coverage Report", output)
            self.assertIn("`libc/src/math`", output)
            self.assertIn("MC/DC Conditions", output)
            self.assertIn("Decisions (Verified / Total)", output)

    def test_cli_nonexistent_file_exits_with_error(self):
        """CLI must exit with code 1 when targeted file does not exist."""
        stderr_buf = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["codebase_coverage.py", "/nonexistent/path/coverage.json"],
        ):
            with patch("sys.stderr", stderr_buf):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
        self.assertIn("Error: Failed to parse coverage JSON", stderr_buf.getvalue())

    def test_cli_invalid_json_exits_with_error(self):
        """CLI must exit with code 1 when targeted file contains invalid JSON syntax."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "cov.json")
            with open(tmp_path, "w") as tmp_file:
                tmp_file.write("INVALID JSON CONTENT")

            stderr_buf = io.StringIO()
            with patch.object(sys, "argv", ["codebase_coverage.py", tmp_path]):
                with patch("sys.stderr", stderr_buf):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error: Failed to parse coverage JSON", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
