# ====- Unit tests for codebase_coverage.py ------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import unittest
import json
from codebase_coverage import (
    DirectoryCoverageMetrics,
    FullCoverageSummary,
    extract_full_coverage_statistics,
    format_directory_breakdown_table,
    render_full_report
)

class TestCodebaseDirectoryAggregation(unittest.TestCase):
    def test_path_filtering_and_routing(self):
        mock_json = {
            "data": [{
                "files": [
                    {
                        "filename": "/workspace/libc/src/math/sin.cpp",
                        "summary": {
                            "lines": {"count": 100, "covered": 50, "percent": 50.0},
                            "functions": {"count": 2, "covered": 1, "percent": 50.0}
                        }
                    },
                    {
                        "filename": "/workspace/libc/src/string/strcpy.cpp",
                        "summary": {
                            "lines": {"count": 200, "covered": 200, "percent": 100.0},
                            "functions": {"count": 4, "covered": 4, "percent": 100.0}
                        }
                    },
                    {
                        "filename": "/workspace/libc/test/src/math/sin_test.cpp",
                        "summary": {
                            "lines": {"count": 500, "covered": 500, "percent": 100.0},
                            "functions": {"count": 10, "covered": 10, "percent": 100.0}
                        }
                    }
                ]
            }]
        }
        
        summary = extract_full_coverage_statistics(mock_json)
        
        # The test file should be ignored entirely
        self.assertEqual(summary.global_stats.lines_tot, 300)
        self.assertEqual(summary.global_stats.lines_cov, 250)
        self.assertEqual(summary.global_stats.func_tot, 6)
        self.assertEqual(summary.global_stats.func_cov, 5)
        
        # Ensure proper bucket routing
        self.assertIn("src/math", summary.directories)
        self.assertEqual(summary.directories["src/math"].lines_tot, 100)
        self.assertIn("src/string", summary.directories)
        self.assertEqual(summary.directories["src/string"].lines_tot, 200)
        self.assertNotIn("test/src/math", summary.directories)

class TestCodebaseReportRendering(unittest.TestCase):
    def test_format_directory_breakdown_table(self):
        metrics_math = DirectoryCoverageMetrics(
            name="src/math", lines_tot=100, lines_cov=50, func_tot=10, func_cov=5
        )
        metrics_string = DirectoryCoverageMetrics(
            name="src/string", lines_tot=200, lines_cov=200, func_tot=20, func_cov=20
        )
        summary = FullCoverageSummary(
            global_stats=metrics_math,
            directories={"src/math": metrics_math, "src/string": metrics_string}
        )
        
        table = format_directory_breakdown_table(summary)
        self.assertIn("`libc/src/math`", table)
        self.assertIn("`libc/src/string`", table)

if __name__ == '__main__':
    unittest.main()
