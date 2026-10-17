# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import build_graph, config, failure_logs, utils
from clang_ssaf_orchestrator_make.failure_logs import SSAFFailureKind


class NamedForTheCommand(unittest.TestCase):
    def _path(self, command, cwd="/p"):
        with mock.patch("os.getcwd", return_value=cwd):
            return failure_logs.get_failure_log_path(
                "/out", SSAFFailureKind.FORWARD, command, None
            )

    def test_two_commands_get_two_logs(self):
        self.assertNotEqual(self._path(["ld", "a.o"]), self._path(["ld", "b.o"]))

    def test_one_command_in_two_directories_gets_two_logs(self):
        self.assertNotEqual(
            self._path(["ld", "a.o"], "/p/lib1"), self._path(["ld", "a.o"], "/p/lib2")
        )

    def test_the_same_command_in_the_same_directory_gets_the_same_log(self):
        self.assertEqual(self._path(["ld", "a.o"]), self._path(["ld", "a.o"]))


class HasFailureLogFor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, ssaf_failure_kind, build_artifact_path):
        path = failure_logs.get_failure_log_path(
            self.tmp,
            ssaf_failure_kind,
            None,
            build_graph.BuildArtifactId(build_artifact_path),
        )
        open(utils.make_parents(path), "w").close()

    def _has_log(self, build_artifact_path):
        return failure_logs.has_failure_log_for(
            self.tmp, build_graph.BuildArtifactId(build_artifact_path)
        )

    def test_a_log_of_any_kind_is_a_log_for_its_artifact(self):
        for each in SSAFFailureKind:
            self._write(each, "/p/%s.o" % each.value)
            self.assertTrue(self._has_log("/p/%s.o" % each.value), each)

    def test_a_log_for_a_longer_name_is_not_a_log_for_this_one(self):
        self._write(SSAFFailureKind.EXTRACT, "/p/a.o.x.o")
        self.assertFalse(self._has_log("/p/a.o"))


class CountFailureLogs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, directory, name):
        path = utils.make_parents(os.path.join(directory, name))
        open(path, "w").close()

    def test_an_output_directory_that_is_not_there_counts_nothing(self):
        self.assertEqual(
            0, failure_logs.count_failure_logs(os.path.join(self.tmp, "none"))
        )

    def test_logs_about_artifacts_and_about_commands_add_up(self):
        self._write(config.failure_log_dir(self.tmp), "p/a.o.extract.log")
        self._write(config.failure_log_dir(self.tmp), "p/0123456789ab.internal.log")
        self.assertEqual(2, failure_logs.count_failure_logs(self.tmp))

    def test_a_log_anywhere_under_the_output_directory_is_counted(self):
        self._write(config.ssaf_artifact_dir(self.tmp), "p/a.o.extract.log")
        self.assertEqual(1, failure_logs.count_failure_logs(self.tmp))

    def test_what_is_not_a_log_is_not_counted(self):
        self._write(config.ssaf_artifact_dir(self.tmp), "p/a.o.summary.json")
        self._write(self.tmp, "wpa.json")
        self.assertEqual(0, failure_logs.count_failure_logs(self.tmp))


if __name__ == "__main__":
    unittest.main()
