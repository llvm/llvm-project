# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import shutil
import tempfile
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import config, failure_logs, orchestrator
from clang_ssaf_orchestrator_make.shims import run
from clang_ssaf_orchestrator_make.shims import shim as shim_module

from . import configs


class RunShim(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env = {
            config.SSAF_CONFIG_ENV_VAR: orchestrator.write_pass_config(
                configs.workflow_config(output_dir=self.tmp), configs.analysis_config()
            )
        }
        self.binary_status = 0
        self.binary_runs = []
        self.ssaf_runs = []
        patch = mock.patch.object(
            shim_module.Shim, "run_bin_tool", lambda shim, path, cwd: self._binary(shim)
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _binary(self, shim):
        self.binary_runs.append(shim.argv)
        return self.binary_status

    def _run_shim(self, argv, work=lambda: None):
        def invoke_ssaf_tools(shim, cwd):
            self.ssaf_runs.append(shim.argv)
            work()

        return run.run_shim(invoke_ssaf_tools, argv, self.env)

    def test_a_shim_reached_through_its_link_runs_the_tool_then_the_ssaf_half(self):
        self.assertEqual(0, self._run_shim(["/out/shims/ld", "-o", "prg", "a.o"]))
        self.assertEqual([["/out/shims/ld", "-o", "prg", "a.o"]], self.binary_runs)
        self.assertEqual([["/out/shims/ld", "-o", "prg", "a.o"]], self.ssaf_runs)

    def test_a_forwarded_job_does_not_run_the_tool_again(self):
        argv = ["/out/shims/ld", "-o", "prg", "a.o"]
        self.assertEqual(
            0, self._run_shim(argv + [shim_module.SKIP_BINARY_TOOL_INVOCATION])
        )
        self.assertEqual([], self.binary_runs)
        self.assertEqual([argv], self.ssaf_runs)

    def test_a_tool_that_fails_is_the_shims_status_and_no_ssaf_step_runs(self):
        self.binary_status = 2
        self.assertEqual(2, self._run_shim(["ld", "-o", "prg"]))
        self.assertEqual([], self.ssaf_runs)

    def test_an_internal_error_lets_the_build_go_on_and_is_counted(self):
        def explode():
            raise ValueError("deliberate")

        with mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(0, self._run_shim(["ld", "-o", "prg"], explode))
        self.assertEqual(1, failure_logs.count_failure_logs(self.tmp))

    def test_outside_a_run_nothing_runs(self):
        with mock.patch("sys.stderr", io.StringIO()):
            status = run.run_shim(
                lambda shim, cwd: self.ssaf_runs.append(shim.argv), ["ld"], {}
            )
        self.assertEqual(1, status)
        self.assertEqual([], self.binary_runs)
        self.assertEqual([], self.ssaf_runs)


if __name__ == "__main__":
    unittest.main()
