# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from clang_ssaf_driver import make_orchestrator, options
from clang_ssaf_driver.transformations import TRANSFORMATIONS


class Command(unittest.TestCase):
    def test_the_orchestrator_runs_the_recipe_under_this_python(self):
        command_line_opts, _ = options.resolve(
            [
                "--transformation=cpp-bounded-buffers",
                "-o",
                "out",
                "--ssaf-clang=/x/clang",
                "make",
                "-j8",
            ]
        )
        self.assertEqual(
            [
                sys.executable,
                "/prefix/bin/clang-ssaf-orchestrator-make",
                "--extract-summaries=PointerFlow,TypeConstrainedPointers,"
                "UnsafeBufferUsage,VirtualMethod",
                "--analysis=UnsafeBufferReachableAnalysisResult",
                "--transformation=cpp-bounded-buffers",
                "--output-dir=out",
                "--ssaf-clang=/x/clang",
                "--",
                "make",
                "-j8",
            ],
            make_orchestrator.orchestrator_command(
                "/prefix/bin/clang-ssaf-orchestrator-make",
                TRANSFORMATIONS["cpp-bounded-buffers"],
                command_line_opts,
            ),
        )

    def test_the_orchestrator_is_the_one_installed_beside_the_driver(self):
        with mock.patch.object(
            make_orchestrator,
            "__file__",
            "/prefix/lib/clang_ssaf_driver/make_orchestrator.py",
        ):
            self.assertEqual(
                "/prefix/bin/clang-ssaf-orchestrator-make",
                make_orchestrator.find_orchestrator(),
            )


class Results(unittest.TestCase):
    def test_a_result_line_is_read(self):
        self.assertEqual(
            ("source-edits", "/out/prg.edits.yaml"),
            make_orchestrator.read_result(b"source-edits: /out/prg.edits.yaml\n"),
        )

    def test_nothing_else_is(self):
        for line in (
            b"clang++ -c -o build/a.o src/a.cpp\n",
            b"link-unit: /p/prg\n",
            b"source-edits:\n",
            b"make: source-edits: /x\n",
        ):
            self.assertIsNone(make_orchestrator.read_result(line))

    def test_a_path_that_is_not_utf8_survives(self):
        _, path = make_orchestrator.read_result(b"source-edits: /out/\xff.yaml\n")
        self.assertEqual(b"/out/\xff.yaml", os.fsencode(path))


class Running(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _run(self, body):
        script = os.path.join(self.tmp, "orchestrator")
        with open(script, "w") as handle:
            handle.write("import os, signal, sys\nout = sys.stdout.buffer\n" + body)
        relay = io.BytesIO()
        status, results = make_orchestrator.run_orchestrator(
            [sys.executable, script], False, relay
        )
        return status, results, relay.getvalue()

    def test_everything_is_relayed_and_the_last_results_are_kept(self):
        printed = (
            b"make: entering\n"
            b"source-edits: /old\n"
            b"\xff not utf8\n"
            b"source-edits: /out/prg.edits.yaml\n"
            b"transformation-report: /out/prg.report.sarif\n"
            b"conflict-report: /out/prg.conflict-report.sarif\n"
            b"output-dir: /out\n"
        )
        status, results, relayed = self._run("out.write(%r)\n" % printed)
        self.assertEqual(0, status)
        self.assertEqual(printed, relayed)
        self.assertEqual(
            {
                "source-edits": "/out/prg.edits.yaml",
                "transformation-report": "/out/prg.report.sarif",
                "conflict-report": "/out/prg.conflict-report.sarif",
            },
            results,
        )

    def test_the_exit_status_passes_through(self):
        status, _, _ = self._run("sys.exit(3)\n")
        self.assertEqual(3, status)

    def test_death_by_a_signal_is_the_shells_status_for_it(self):
        status, _, _ = self._run("os.kill(os.getpid(), signal.SIGTERM)\n")
        self.assertEqual(128 + 15, status)

    def test_a_result_glued_to_the_builds_last_output_is_not_read(self):
        printed = b"no newline" + b"source-edits: /x\n"
        status, results, relayed = self._run("out.write(%r)\n" % printed)
        self.assertEqual(0, status)
        self.assertEqual(printed, relayed)
        self.assertEqual({}, results)


if __name__ == "__main__":
    unittest.main()
