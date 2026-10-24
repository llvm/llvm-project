# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import tempfile
import unittest

from clang_ssaf_driver import report


def _location(uri, **region):
    return {"physicalLocation": {"artifactLocation": {"uri": uri}, "region": region}}


def _log(*results):
    return {"runs": [{"results": list(results)}]}


class Loading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, name, document):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            if isinstance(document, str):
                handle.write(document)
            else:
                json.dump(document, handle)
        return path

    def test_the_runs_of_both_reports_make_one_log(self):
        transformation = self._write(
            "t.sarif", {"runs": [{"tool": {"driver": {"name": "clang-ssaf"}}}]}
        )
        conflicts = self._write(
            "c.sarif",
            {"runs": [{"tool": {"driver": {"name": "clang-ssaf-src-edit-merge"}}}]},
        )
        log, problem = report.load([transformation, conflicts])
        self.assertIsNone(problem)
        self.assertEqual("2.1.0", log["version"])
        self.assertEqual(
            ["clang-ssaf", "clang-ssaf-src-edit-merge"],
            [run["tool"]["driver"]["name"] for run in log["runs"]],
        )

    def test_an_absent_report_adds_nothing(self):
        conflicts = self._write("c.sarif", {"runs": [{"results": []}]})
        log, problem = report.load([None, conflicts])
        self.assertIsNone(problem)
        self.assertEqual([{"results": []}], log["runs"])

    def test_what_is_not_a_sarif_log_is_a_problem(self):
        _, problem = report.load([self._write("list.sarif", "[]")])
        self.assertIn("not a SARIF log", problem.message)
        _, problem = report.load([self._write("bad.sarif", "{")])
        self.assertIn("cannot read the report", problem.message)

    def test_the_log_is_written_as_json(self):
        log, _ = report.load([])
        path = os.path.join(self.tmp, "out.sarif")
        report.write(path, log)
        with open(path) as handle:
            self.assertEqual(log, json.load(handle))


class Diagnostics(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        os.makedirs(os.path.join(self.root, "src"))
        with open(os.path.join(self.root, "src", "shared.h"), "wb") as handle:
            handle.write(b"ab\ncdef\n")
        self.shared = "file://" + os.path.join(self.root, "src", "shared.h")

    def test_a_finding_reads_like_a_compiler_diagnostic(self):
        log = _log(
            {
                "level": "note",
                "locations": [_location(self.shared, startLine=3, startColumn=7)],
                "message": {"text": "not yet rewritten"},
                "ruleId": "cpp-bounded-buffers-skipped",
            }
        )
        self.assertEqual(
            [
                "src/shared.h:3:7: note: not yet rewritten "
                "[cpp-bounded-buffers-skipped]",
                "1 cpp-bounded-buffers-skipped reported",
            ],
            report.diagnostics(log, self.root),
        )

    def test_a_conflict_is_placed_by_its_bytes_and_followed_by_its_candidates(self):
        log = _log(
            {
                "level": "error",
                "locations": [_location(self.shared, byteOffset=5)],
                "message": {"text": "dropped"},
                "relatedLocations": [
                    dict(
                        _location(self.shared, byteOffset=3, byteLength=2),
                        message={"text": 'candidate edit: "x"'},
                    )
                ],
                "ruleId": "conflict",
            }
        )
        self.assertEqual(
            [
                "src/shared.h:2:3: error: dropped [conflict]",
                'src/shared.h:2:1: note: candidate edit: "x"',
                "1 conflict reported",
            ],
            report.diagnostics(log, self.root),
        )

    def test_a_file_outside_the_project_keeps_its_whole_path(self):
        log = _log(
            {
                "level": "note",
                "locations": [_location("file:///usr/include/my%20lib.h", startLine=1)],
                "message": {"text": "m"},
                "ruleId": "r",
            }
        )
        self.assertEqual(
            "/usr/include/my lib.h:1:1: note: m [r]",
            report.diagnostics(log, self.root)[0],
        )

    def test_a_finding_without_a_location_is_the_drivers_own(self):
        self.assertEqual(
            "clang-ssaf-driver: warning: m [r]",
            report.diagnostics(_log({"message": {"text": "m"}, "ruleId": "r"}), "/")[0],
        )

    def test_the_findings_are_counted_per_rule(self):
        finding = {"locations": [_location(self.shared, startLine=1)]}
        log = {
            "runs": [
                {"results": [dict(finding, ruleId="b"), dict(finding, ruleId="a")]},
                {"results": [dict(finding, ruleId="b")]},
            ]
        }
        self.assertEqual("1 a, 2 b reported", report.diagnostics(log, self.root)[-1])

    def test_no_findings_print_nothing(self):
        self.assertEqual([], report.diagnostics(_log(), self.root))

    def test_a_merge_tools_path_is_read_as_written(self):
        path = os.path.join(self.root, "src", "a#b?c.h")
        with open(path, "wb") as handle:
            handle.write(b"x\ny\n")
        log = _log(
            {
                "locations": [_location("file://" + path, byteOffset=2)],
                "message": {"text": "m"},
                "ruleId": "r",
            }
        )
        self.assertEqual(
            "src/a#b?c.h:2:1: warning: m [r]", report.diagnostics(log, self.root)[0]
        )


class ConflictRanges(unittest.TestCase):
    def test_every_candidate_of_every_conflict_is_a_range(self):
        log = {
            "runs": [
                {"results": [{"locations": [_location("file:///p/a.h", startLine=1)]}]},
                {
                    "results": [
                        {
                            "locations": [_location("file:///p/h.h", byteOffset=4)],
                            "relatedLocations": [
                                _location("file:///p/h.h", byteOffset=4, byteLength=3),
                                _location("file:///p/h.h", byteOffset=6, byteLength=9),
                            ],
                        },
                        {
                            "relatedLocations": [
                                _location(
                                    "file:///p/my%20c.h", byteOffset=0, byteLength=1
                                )
                            ]
                        },
                    ]
                },
            ]
        }
        self.assertEqual(
            {"/p/h.h": [(4, 7), (6, 15)], "/p/my c.h": [(0, 1)]},
            report.conflict_ranges(log),
        )


if __name__ == "__main__":
    unittest.main()
