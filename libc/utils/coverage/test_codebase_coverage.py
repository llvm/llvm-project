#!/usr/bin/env python3
#
# ===- Unit tests for codebase coverage analyzer -------------*- python -*--==#
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
import unittest
from contextlib import redirect_stdout
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codebase_coverage import (
    DirectoryCoverageMetrics,
    FullCoverageSummary,
    extract_full_coverage_statistics,
    format_directory_breakdown_table,
    format_global_summary_table,
    format_overview_callout,
    render_full_report,
    resolve_dashboard_url,
)


class TestDirectoryCoverageMetrics(unittest.TestCase):
    """Tests for mathematical calculation and edge-case handling in metric models."""

    def test_metrics_percentages(self) -> None:
        metrics = DirectoryCoverageMetrics(
            name="math",
            lines_cov=850,
            lines_tot=1000,
            func_cov=45,
            func_tot=50,
            mcdc_cov=90,
            mcdc_tot=100,
            decisions_tot=40,
            decisions_full=36,
        )
        self.assertEqual(metrics.line_pct, 85.0)
        self.assertEqual(metrics.func_pct, 90.0)
        self.assertEqual(metrics.mcdc_pct, 90.0)
        self.assertEqual(metrics.decisions_pct, 90.0)
        self.assertEqual(metrics.missed_lines, 150)

    def test_zero_division_guard(self) -> None:
        empty = DirectoryCoverageMetrics(name="empty")
        self.assertEqual(empty.line_pct, 0.0)
        self.assertEqual(empty.func_pct, 0.0)
        self.assertEqual(empty.mcdc_pct, 0.0)
        self.assertEqual(empty.decisions_pct, 0.0)
        self.assertEqual(empty.missed_lines, 0)


class TestDashboardURLResolution(unittest.TestCase):
    """Tests for environment variable resolution and URL formatting."""

    def test_custom_environment_variable(self) -> None:
        os.environ["COVERAGE_DASHBOARD_URL"] = "https://custom-dashboard.internal/"
        url = resolve_dashboard_url(has_mcdc=False)
        self.assertEqual(url, "https://custom-dashboard.internal/")

        url_mcdc = resolve_dashboard_url(has_mcdc=True)
        self.assertEqual(url_mcdc, "https://custom-dashboard.internal/mcdc/")

    def test_github_repository_fallback(self) -> None:
        os.environ.pop("COVERAGE_DASHBOARD_URL", None)
        os.environ["GITHUB_REPOSITORY"] = "tapiwagonga/llvm-project"

        url = resolve_dashboard_url(has_mcdc=False)
        self.assertEqual(url, "https://tapiwagonga.github.io/llvm-project/")

        url_mcdc = resolve_dashboard_url(has_mcdc=True)
        self.assertEqual(url_mcdc, "https://tapiwagonga.github.io/llvm-project/mcdc/")


class TestDataExtraction(unittest.TestCase):
    """Tests for extracting and aggregating metrics across multi-directory codebase."""

    def test_multi_directory_aggregation(self) -> None:
        cov_data = {
            "data": [
                {
                    "files": [
                        # 1. ctype directory
                        {
                            "filename": "libc/src/ctype/isalpha.cpp",
                            "summary": {
                                "lines": {"count": 10, "covered": 10},
                                "functions": {"count": 1, "covered": 1},
                                "mcdc": {"count": 2, "covered": 2},
                            },
                            "mcdc_records": [
                                [10, 1, 10, 20, 2, 2, 2, 1, 1, [True, True]]
                            ],
                        },
                        # 2. math directory with nested directory
                        {
                            "filename": "libc/src/math/generic/sin.cpp",
                            "summary": {
                                "lines": {"count": 50, "covered": 40},
                                "functions": {"count": 2, "covered": 2},
                                "mcdc": {"count": 6, "covered": 4},
                            },
                            "mcdc_records": [
                                [15, 1, 15, 30, 2, 1, 2, 1, 1, [True, False]],
                                [25, 1, 25, 30, 2, 2, 2, 1, 1, [True, True]],
                            ],
                        },
                        # 3. support directory
                        {
                            "filename": "libc/src/__support/OSUtil/linux/syscall.cpp",
                            "summary": {
                                "lines": {"count": 100, "covered": 90},
                                "functions": {"count": 5, "covered": 5},
                                "mcdc": {"count": 0, "covered": 0},
                            },
                            "mcdc_records": [],
                        },
                        # 4. Ignored test and benchmark files
                        {
                            "filename": "libc/test/src/math/sin_test.cpp",
                            "summary": {"lines": {"count": 200, "covered": 200}},
                        },
                        {
                            "filename": "libc/utils/mathtools/ryu.py",
                            "summary": {"lines": {"count": 80, "covered": 80}},
                        },
                    ]
                }
            ]
        }

        summary = extract_full_coverage_statistics(cov_data)
        self.assertIsNotNone(summary)
        assert summary is not None

        # Verify only 3 libc/src directories are tracked
        self.assertEqual(len(summary.directories), 3)
        self.assertIn("src/ctype", summary.directories)
        self.assertIn("src/math", summary.directories)
        self.assertIn("src/__support", summary.directories)

        # Verify directory-specific metrics
        math_m = summary.directories["src/math"]
        self.assertEqual(math_m.lines_cov, 40)
        self.assertEqual(math_m.lines_tot, 50)
        self.assertEqual(math_m.line_pct, 80.0)
        self.assertEqual(math_m.mcdc_cov, 4)
        self.assertEqual(math_m.mcdc_tot, 6)
        self.assertEqual(math_m.decisions_tot, 2)
        self.assertEqual(math_m.decisions_full, 1)

        # Verify global aggregate sums
        self.assertEqual(summary.global_stats.lines_cov, 140)  # 10 + 40 + 90
        self.assertEqual(summary.global_stats.lines_tot, 160)  # 10 + 50 + 100
        self.assertEqual(summary.global_stats.func_cov, 8)     # 1 + 2 + 5
        self.assertEqual(summary.global_stats.mcdc_cov, 6)     # 2 + 4 + 0
        self.assertEqual(summary.global_stats.mcdc_tot, 8)     # 2 + 6 + 0
        self.assertEqual(summary.global_stats.decisions_tot, 3)# 1 + 2 + 0
        self.assertEqual(summary.global_stats.decisions_full, 2)# 1 + 1 + 0
        self.assertTrue(summary.has_mcdc)

    def test_empty_or_malformed_json(self) -> None:
        self.assertIsNone(extract_full_coverage_statistics({}))
        self.assertIsNone(extract_full_coverage_statistics({"data": []}))
        self.assertIsNone(extract_full_coverage_statistics({"data": [{"files": []}]}))


class TestReportFormatting(unittest.TestCase):
    """Tests for Markdown table generation, progress indicators, and banners."""

    def test_format_overview_callout(self) -> None:
        # Standard line mode
        g_std = DirectoryCoverageMetrics(lines_cov=950, lines_tot=1000)
        s_std = FullCoverageSummary(global_stats=g_std, dashboard_url="https://llvm.github.io/llvm-project/")
        callout_std = format_overview_callout(s_std)
        self.assertIn("95.00%", callout_std)
        self.assertNotIn("MC/DC", callout_std)

        # MC/DC mode
        g_mcdc = DirectoryCoverageMetrics(lines_cov=950, lines_tot=1000, mcdc_cov=90, mcdc_tot=100, decisions_tot=40)
        s_mcdc = FullCoverageSummary(global_stats=g_mcdc, dashboard_url="https://llvm.github.io/llvm-project/mcdc/")
        callout_mcdc = format_overview_callout(s_mcdc)
        self.assertIn("95.00% Line", callout_mcdc)
        self.assertIn("90.00% MC/DC", callout_mcdc)
        self.assertIn("https://llvm.github.io/llvm-project/mcdc/", callout_mcdc)

    def test_format_global_summary_table(self) -> None:
        g = DirectoryCoverageMetrics(
            lines_cov=950, lines_tot=1000, func_cov=98, func_tot=100,
            mcdc_cov=90, mcdc_tot=100, decisions_full=36, decisions_tot=40
        )
        summary = FullCoverageSummary(global_stats=g)
        table = format_global_summary_table(summary)

        self.assertIn("### Overall", table)
        self.assertIn("| **MC/DC Condition Independence** | 90 | 100 | **90.00%** |", table)
        self.assertIn("| **Fully Verified Decisions** | 36 | 40 | **90.00%** |", table)
        self.assertIn("| **Executable Lines** | 950 | 1,000 | **95.00%** |", table)
        self.assertIn("| **Functions** | 98 | 100 | **98.00%** |", table)

    def test_format_directory_breakdown_table_sorted(self) -> None:
        dir_metrics = {
            "src/string": DirectoryCoverageMetrics(name="src/string", lines_cov=20, lines_tot=20),
            "src/ctype": DirectoryCoverageMetrics(name="src/ctype", lines_cov=10, lines_tot=10),
            "src/math": DirectoryCoverageMetrics(name="src/math", lines_cov=40, lines_tot=50),
        }
        summary = FullCoverageSummary(directories=dir_metrics)
        table = format_directory_breakdown_table(summary)

        # Must be alphabetically sorted: ctype, math, string
        pos_ctype = table.find("`libc/src/ctype`")
        pos_math = table.find("`libc/src/math`")
        pos_string = table.find("`libc/src/string`")

        self.assertTrue(pos_ctype < pos_math < pos_string)

    def test_render_full_report_empty_data(self) -> None:
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            render_full_report({})
        output = stdout_buf.getvalue()
        self.assertIn("No Coverage Data Detected", output)


if __name__ == "__main__":
    unittest.main()
