# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

from clang_ssaf_driver import apply, patch, source_edits
from clang_ssaf_driver.source_edits import Replacement


def _executable(path, body="pass\n"):
    with open(path, "w") as handle:
        handle.write("#!%s\nimport os, sys\n%s" % (sys.executable, body))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class ResolvingTheTool(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tool = _executable(os.path.join(self.tmp, "clang-apply-replacements"))

    def test_a_path_is_taken_as_given(self):
        self.assertEqual((self.tool, None), apply.resolve_tool(self.tool))

    def test_the_default_is_found_on_path(self):
        with mock.patch.dict(os.environ, {"PATH": self.tmp}):
            self.assertEqual((self.tool, None), apply.resolve_tool(None))

    def test_what_cannot_be_run_is_a_problem(self):
        text = os.path.join(self.tmp, "text")
        open(text, "w").close()
        cases = (
            ("no-such-tool", "no-such-tool: not found on PATH"),
            (os.path.join(self.tmp, "missing"), "not a file: %s/missing" % self.tmp),
            (self.tmp, "not a file: %s" % self.tmp),
            (text, "not executable: %s" % text),
        )
        for value, error in cases:
            with mock.patch.dict(os.environ, {"PATH": self.tmp}):
                tool, problem = apply.resolve_tool(value)
            self.assertIsNone(tool)
            self.assertEqual("--apply needs clang-apply-replacements:", problem.message)
            self.assertEqual(["--clang-apply-replacements: " + error], problem.details)


class Applying(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.out)
        self.source = os.path.join(self.tmp, "a.cpp")
        with open(self.source, "wb") as handle:
            handle.write(b"int *p;\n")
        edits = [(0, 5, 'bounded_ptr<int> "q"\\')]
        edited, _ = patch.apply_edits(b"int *p;\n", edits)
        self.files = [patch.FileEdits("a.cpp", self.source, edits, b"", edited)]

    def _tool(self, body):
        return _executable(os.path.join(self.tmp, "car"), body)

    def test_the_tool_gets_a_directory_holding_only_the_drivers_edits(self):
        log = os.path.join(self.tmp, "log")
        tool = self._tool(
            "open(%r, 'w').write(repr((sys.argv[1:], os.listdir(sys.argv[1]))))\n"
            "open(%r, 'wb').write(%r)\n" % (log, self.source, self.files[0].edited)
        )
        self.assertIsNone(apply.apply_source_edits(tool, self.files, self.out, False))
        staging_dir = os.path.join(self.out, "apply")
        with open(log) as handle:
            self.assertEqual(
                repr(([staging_dir], ["source-edits.yaml"])), handle.read()
            )
        self.assertEqual(
            ([Replacement(self.source, 0, 5, 'bounded_ptr<int> "q"\\')], None),
            source_edits.read_source_edits(
                os.path.join(staging_dir, "source-edits.yaml")
            ),
        )

    def test_an_earlier_runs_staging_is_replaced(self):
        stale = os.path.join(self.out, "apply", "stale.yaml")
        os.makedirs(os.path.dirname(stale))
        open(stale, "w").close()
        tool = self._tool(
            "open(%r, 'wb').write(%r)\n" % (self.source, self.files[0].edited)
        )
        self.assertIsNone(apply.apply_source_edits(tool, self.files, self.out, False))
        self.assertFalse(os.path.exists(stale))

    def test_a_failing_tool_is_a_problem_with_its_status(self):
        problem = apply.apply_source_edits(
            self._tool("sys.exit(3)\n"), self.files, self.out, False
        )
        self.assertEqual(3, problem.exit_status)
        self.assertEqual("clang-apply-replacements failed (exit 3)", problem.message)

    def test_a_tool_that_writes_something_else_is_a_problem(self):
        problem = apply.apply_source_edits(
            self._tool("pass\n"), self.files, self.out, False
        )
        self.assertEqual(
            "clang-apply-replacements did not write what the patch shows:",
            problem.message,
        )
        self.assertEqual(["a.cpp"], problem.details)


if __name__ == "__main__":
    unittest.main()
