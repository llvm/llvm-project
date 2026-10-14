# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from clang_ssaf_driver import driver, make_orchestrator, source_edits
from clang_ssaf_driver.source_edits import Replacement


class Running(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.project = os.path.join(self.tmp, "project")
        os.makedirs(os.path.join(self.project, "src"))
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.project)
        with open("src/a.cpp", "wb") as handle:
            handle.write(b"int *p;\n")
        self.out = os.path.join(self.project, "out")
        os.makedirs(self.out)
        orchestrator = os.path.join(self.tmp, "clang-ssaf-orchestrator-make")
        open(orchestrator, "w").close()
        for name, value in (
            ("find_orchestrator", mock.Mock(return_value=orchestrator)),
            ("run_orchestrator", mock.Mock(return_value=(0, self._results()))),
        ):
            patcher = mock.patch.object(make_orchestrator, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _results(self):
        edits = os.path.join(self.tmp, "prg.edits.yaml")
        source_edits.write_source_edits(
            edits, [Replacement("src/a.cpp", 0, 5, "bounded_ptr<int> ")]
        )
        report = os.path.join(self.tmp, "prg.report.sarif")
        with open(report, "w") as handle:
            json.dump(
                {
                    "runs": [
                        {
                            "results": [
                                {
                                    "level": "note",
                                    "message": {"text": "m"},
                                    "ruleId": "r",
                                }
                            ]
                        }
                    ]
                },
                handle,
            )
        return {"source-edits": edits, "transformation-report": report}

    def _main(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            status = driver.main(
                ["--transformation=cpp-bounded-buffers", "-o", "out"]
                + list(extra)
                + ["--", "make"]
            )
        return status, stdout.getvalue(), stderr.getvalue()

    def test_a_run_writes_the_patch_and_the_report_and_names_them(self):
        status, stdout, stderr = self._main()
        self.assertEqual(0, status)
        patch_path = os.path.join(self.out, "cpp-bounded-buffers.patch")
        report_path = os.path.join(self.out, "cpp-bounded-buffers.sarif")
        self.assertEqual("patch: %s\nreport: %s\n" % (patch_path, report_path), stdout)
        self.assertEqual("clang-ssaf-driver: note: m [r]\n1 r reported\n", stderr)
        with open(patch_path, "rb") as handle:
            self.assertIn(b"+bounded_ptr<int> p;\n", handle.read())
        with open(report_path) as handle:
            self.assertEqual(1, len(json.load(handle)["runs"]))
        with open("src/a.cpp", "rb") as handle:
            self.assertEqual(b"int *p;\n", handle.read())

    def test_a_failed_orchestrator_run_writes_nothing(self):
        make_orchestrator.run_orchestrator.return_value = (2, {})
        status, stdout, stderr = self._main()
        self.assertEqual(2, status)
        self.assertEqual("", stdout)
        self.assertEqual(
            "clang-ssaf-driver: error: clang-ssaf-orchestrator-make failed (exit 2)\n",
            stderr,
        )
        self.assertEqual([], os.listdir(self.out))

    def test_a_run_without_source_edits_is_a_problem(self):
        make_orchestrator.run_orchestrator.return_value = (0, {})
        status, _, stderr = self._main()
        self.assertEqual(1, status)
        self.assertIn("reported no source edits", stderr)

    def test_a_missing_orchestrator_stops_the_run_before_it_starts(self):
        make_orchestrator.find_orchestrator.return_value = os.path.join(
            self.tmp, "nothing-here"
        )
        status, _, stderr = self._main()
        self.assertEqual(1, status)
        self.assertIn("cannot find clang-ssaf-orchestrator-make", stderr)
        make_orchestrator.run_orchestrator.assert_not_called()

    def test_a_missing_clang_apply_replacements_stops_the_run_before_it_starts(self):
        status, _, stderr = self._main(
            "--apply", "--clang-apply-replacements=" + os.path.join(self.tmp, "none")
        )
        self.assertEqual(1, status)
        self.assertIn("--apply needs clang-apply-replacements", stderr)
        make_orchestrator.run_orchestrator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
