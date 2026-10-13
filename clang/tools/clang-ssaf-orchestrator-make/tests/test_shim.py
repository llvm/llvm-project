# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import os
import shutil
import signal
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import config
from clang_ssaf_orchestrator_make.shims import shim as shim_utils
from clang_ssaf_orchestrator_make import build_graph
from clang_ssaf_orchestrator_make import failure_logs
from clang_ssaf_orchestrator_make.failure_logs import SSAFFailureKind

from . import configs


class BinaryTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim_link_dir_path = os.path.join(self.tmp, "shims")
        self.binary_tools = os.path.join(self.tmp, "binary")
        for directory in (self.shim_link_dir_path, self.binary_tools):
            os.makedirs(directory)
        for name in ("cc", "clang", "c++", "clang++", "ld", "ar"):
            self._executable(self.binary_tools, name)
        self.workflow_config = configs.workflow_config(
            output_dir=os.path.join(self.tmp, "out")
        )
        self.workflow_config.restored_path_env_var = self.binary_tools

    def _executable(self, directory, name):
        path = os.path.join(directory, name)
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def _shim(self, name, *extra):
        argv = [os.path.join(self.shim_link_dir_path, name), "-c", "a.c"]
        return shim_utils.Shim(
            self.workflow_config,
            config.SSAFAnalysisPassConfig(),
            argv + list(extra),
        )

    def test_a_name_on_path_runs_that_same_name(self):
        shim = self._shim("clang")
        self.assertFalse(shim.skip_binary_tool_invocation)
        self.assertEqual(os.path.join(self.binary_tools, "clang"), shim.bin_tool_path())

    def test_a_name_on_path_is_resolved_off_our_own_path(self):
        self.assertEqual(
            os.path.join(self.binary_tools, "cc"),
            self._shim("cc").bin_tool_path(),
        )

    def test_the_marker_says_the_binary_tool_has_already_run(self):
        shim = self._shim("ld", shim_utils.SKIP_BINARY_TOOL_INVOCATION)
        self.assertTrue(shim.skip_binary_tool_invocation)

    def test_without_the_marker_the_shim_owns_the_binary_tool(self):
        shim = self._shim("ld")
        self.assertFalse(shim.skip_binary_tool_invocation)
        self.assertEqual(os.path.join(self.binary_tools, "ld"), shim.bin_tool_path())

    def test_the_marker_never_reaches_the_binary_tool(self):
        shim = self._shim("ld", shim_utils.SKIP_BINARY_TOOL_INVOCATION)
        self.assertEqual(["-c", "a.c"], shim.argv[1:])
        self.assertNotIn(shim_utils.SKIP_BINARY_TOOL_INVOCATION, shim.argv)

    def test_a_name_that_no_longer_resolves_yields_no_words(self):
        os.remove(os.path.join(self.binary_tools, "ar"))
        self.assertIsNone(self._shim("ar").bin_tool_path())


class ShimEnvironment(unittest.TestCase):
    def setUp(self):
        self.workflow_config = config.SSAFWorkflowConfig(
            installed_shim_dir_path="/libexec",
            restored_path_env_var="/usr/bin",
        )
        self.shim = shim_utils.Shim(
            self.workflow_config, config.SSAFAnalysisPassConfig(), ["cc"]
        )

    def test_the_binary_tool_sees_the_build_s_own_path(self):
        env = {
            config.SSAF_CONFIG_ENV_VAR: "/out/config-analysis.json",
            "PATH": "/out/shims:/usr/bin",
            "HOME": "/home/jan",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cleaned = self.shim.get_unmodified_env()
        self.assertNotIn(config.SSAF_CONFIG_ENV_VAR, cleaned)
        self.assertEqual("/usr/bin", cleaned["PATH"])
        self.assertEqual("/home/jan", cleaned["HOME"])


class MissingSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "project")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.root, "build"))
        workflow_config = config.SSAFWorkflowConfig(output_dir=self.out)
        self.shim = shim_utils.Shim(
            workflow_config,
            config.SSAFAnalysisPassConfig(),
            ["ld", "-o", "build/prg"],
        )

    def _id(self, relative):
        return build_graph.BuildArtifactId(os.path.join(self.root, relative))

    def _report(self, path):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.shim.log_missing_summary(
                self._id("build/prg"), build_graph.BuildArtifactId(path)
            )
        return stream.getvalue()

    def _logs(self):
        logs = config.failure_log_dir(self.out)
        found = []
        for directory, _, filenames in os.walk(logs):
            found.extend(
                os.path.relpath(os.path.join(directory, name), logs)
                for name in filenames
                if name.endswith(".log")
            )
        return sorted(found)

    def _log(self, ssaf_failure_kind, build_artifact_id):
        return os.path.relpath(
            failure_logs.get_failure_log_path(
                self.out, ssaf_failure_kind, None, build_artifact_id
            ),
            config.failure_log_dir(self.out),
        )

    def test_an_artifact_with_no_summary_is_a_counted_failure(self):
        message = self._report(os.path.join(self.root, "build", "libx.a"))
        self.assertEqual(
            "clang-ssaf-orchestrator-make: error: no summary for %s, consumed "
            "by %s; nothing it was assembled from was analysed\n"
            % (
                self._id("build/libx.a").get_path(),
                self._id("build/prg").get_path(),
            ),
            message,
        )
        self.assertEqual(
            [self._log(SSAFFailureKind.MISSING, self._id("build/libx.a"))],
            self._logs(),
        )

    def test_where_the_artifact_lives_does_not_change_that(self):
        slice_object = os.path.join(self.tmp, "slices", "a-arm64.o")
        build_artifact_id = build_graph.BuildArtifactId(slice_object)
        self.assertIn("no summary for", self._report(slice_object))
        self.assertEqual(
            [self._log(SSAFFailureKind.MISSING, build_artifact_id)],
            self._logs(),
        )

    def test_an_already_reported_artifact_is_not_reported_twice(self):
        obj = os.path.join(self.root, "build", "a.o")
        existing = failure_logs.get_failure_log_path(
            self.out, SSAFFailureKind.EXTRACT, None, self._id("build/a.o")
        )
        os.makedirs(os.path.dirname(existing))
        with open(existing, "w") as handle:
            handle.write("the SSAF compilation failed\n")
        self.assertEqual("", self._report(obj))
        self.assertEqual(
            [self._log(SSAFFailureKind.EXTRACT, self._id("build/a.o"))],
            self._logs(),
        )


class WithoutConfiguration(unittest.TestCase):
    def test_a_shim_run_outside_a_run_says_so(self):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            status = shim_utils.error_no_config()
        self.assertEqual(1, status)
        self.assertEqual(
            "clang-ssaf-orchestrator-make: error: missing configuration file\n"
            "  %s is not set, so this is not an orchestrator run\n"
            % config.SSAF_CONFIG_ENV_VAR,
            stream.getvalue(),
        )


class Argv(unittest.TestCase):
    def _shim(self, argv):
        return shim_utils.Shim(configs.workflow_config(), None, argv)

    def test_the_marker_is_taken_only_off_the_end(self):
        shim = self._shim(["ld", shim_utils.SKIP_BINARY_TOOL_INVOCATION, "a.o"])
        self.assertFalse(shim.skip_binary_tool_invocation)
        self.assertIn(shim_utils.SKIP_BINARY_TOOL_INVOCATION, shim.argv)

    def test_a_command_of_nothing_but_the_marker_keeps_it(self):
        shim = self._shim([shim_utils.SKIP_BINARY_TOOL_INVOCATION])
        self.assertFalse(shim.skip_binary_tool_invocation)

    def test_the_tool_is_named_by_the_link_it_was_reached_through(self):
        shim = self._shim(["/out/shims/c++", "-c", "a.c"])
        self.assertEqual("c++", shim.invoked_as)

    def test_the_artifact_repository_is_built_once(self):
        shim = self._shim(["ld", "-o", "prg"])
        self.assertIs(shim.ssaf_artifact_repo, shim.ssaf_artifact_repo)


class RunBinaryTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = shim_utils.Shim(
            configs.workflow_config(output_dir=self.tmp),
            config.SSAFAnalysisPassConfig(),
            ["ld", "-o", "prg"],
        )

    def _stderr(self, work):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.status = work()
        return stream.getvalue()

    def test_a_tool_that_cannot_be_found_is_a_missing_command(self):
        message = self._stderr(lambda: self.shim.run_bin_tool(None, self.tmp))
        self.assertEqual(127, self.status)
        self.assertIn("cannot find the binary ld", message)

    def test_a_tool_that_cannot_be_run_is_a_missing_command(self):
        message = self._stderr(
            lambda: self.shim.run_bin_tool(os.path.join(self.tmp, "nope"), self.tmp)
        )
        self.assertEqual(127, self.status)
        self.assertIn("cannot run", message)

    def test_the_tool_runs_as_the_build_asked_in_the_builds_environment(self):
        self.shim.workflow_config.restored_path_env_var = "/usr/bin"
        live = {config.SSAF_CONFIG_ENV_VAR: "/out/c.json", "PATH": "/out/shims"}
        with mock.patch.dict(os.environ, live), mock.patch(
            "subprocess.call", return_value=3
        ) as called:
            status = self.shim.run_bin_tool("/usr/bin/ld", self.tmp)
        self.assertEqual(3, status)
        self.assertEqual(["/usr/bin/ld", "-o", "prg"], called.call_args[0][0])
        self.assertEqual(self.tmp, called.call_args[1]["cwd"])
        env = called.call_args[1]["env"]
        self.assertNotIn(config.SSAF_CONFIG_ENV_VAR, env)
        self.assertEqual("/usr/bin", env["PATH"])

    def test_a_signal_the_tool_died_of_is_re_raised(self):
        killed = []
        with mock.patch("signal.signal") as signalled, mock.patch(
            "os.kill", side_effect=lambda pid, number: killed.append(number)
        ):
            status = self.shim._propagate_process_return_status(-signal.SIGTERM)
        self.assertEqual([signal.SIGTERM], killed)
        self.assertEqual((signal.SIGTERM, signal.SIG_DFL), signalled.call_args[0])
        self.assertEqual(128 + signal.SIGTERM, status)

    def test_a_signal_that_cannot_be_reset_is_not_raised_on_ourselves(self):
        with mock.patch("signal.signal", side_effect=OSError(22, "bad")), mock.patch(
            "os.kill"
        ) as killed:
            self.assertEqual(137, self.shim._propagate_process_return_status(-9))
        killed.assert_not_called()


class Protect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = shim_utils.Shim(
            configs.workflow_config(output_dir=self.tmp),
            config.SSAFAnalysisPassConfig(),
            ["ld", "-o", "prg"],
        )

    def test_work_that_succeeds_is_not_reported(self):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.assertEqual(0, self.shim.protect(lambda: None))
        self.assertEqual("", stream.getvalue())

    def test_an_internal_error_is_reported_logged_and_swallowed(self):
        def explode():
            raise ValueError("deliberate")

        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.assertEqual(0, self.shim.protect(explode))
        self.assertIn("internal error in ld", stream.getvalue())
        self.assertIn("deliberate", stream.getvalue())
        log = failure_logs.get_failure_log_path(
            self.tmp, SSAFFailureKind.INTERNAL, self.shim.argv, None
        )
        self.assertTrue(os.path.isfile(log))
        with open(log) as handle:
            self.assertIn("deliberate", handle.read())

    def test_a_log_that_cannot_be_written_is_itself_reported(self):
        def explode():
            raise ValueError("deliberate")

        stream = io.StringIO()
        with mock.patch("sys.stderr", stream), mock.patch(
            "clang_ssaf_orchestrator_make.utils.make_parents", side_effect=OSError
        ):
            self.assertEqual(0, self.shim.protect(explode))
        self.assertIn("could not be written", stream.getvalue())


class RunningAnSsafStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = shim_utils.Shim(
            configs.workflow_config(output_dir=self.tmp),
            config.SSAFAnalysisPassConfig(),
            ["ld", "-o", "prg"],
        )
        self.build_artifact_id = build_graph.BuildArtifactId("/p/prg")

    def _run(self, command):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            succeeded = self.shim.run(
                command, SSAFFailureKind.LINK, self.build_artifact_id
            )
        return succeeded, stream.getvalue()

    def test_what_a_step_says_is_kept_from_the_build_unless_relayed(self):
        command = [sys.executable, "-c", "print('from the step')"]
        _, quiet = self._run(command)
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.shim.run(
                command, SSAFFailureKind.FORWARD, self.build_artifact_id, True
            )
        self.assertNotIn("from the step", quiet)
        self.assertIn("from the step", stream.getvalue())


class FailureLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = shim_utils.Shim(
            configs.workflow_config(output_dir=self.tmp),
            config.SSAFAnalysisPassConfig(),
            ["ld", "-o", "prg"],
        )
        self.build_artifact_id = build_graph.BuildArtifactId("/p/build/a.o")

    def _log(self):
        return failure_logs.get_failure_log_path(
            self.tmp, SSAFFailureKind.EXTRACT, None, self.build_artifact_id
        )

    def _log_failure(self, command, completed):
        self.shim.log_failure(
            command, completed, SSAFFailureKind.EXTRACT, self.build_artifact_id
        )

    def test_the_command_and_its_output_are_recorded(self):
        completed = types.SimpleNamespace(
            returncode=2, stdout=b"it broke\n", stderr=None
        )
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self._log_failure(["ssaf-clang", "a.c"], completed)
        with open(self._log()) as handle:
            self.assertEqual("$ ssaf-clang a.c\nit broke\n", handle.read())
        self.assertIn("exit 2", stream.getvalue())
        self.assertIn(self._log(), stream.getvalue())

    def test_a_tool_that_said_nothing_still_gets_a_log(self):
        completed = types.SimpleNamespace(returncode=1, stdout=None, stderr=None)
        with mock.patch("sys.stderr", io.StringIO()):
            self._log_failure(["ssaf-clang"], completed)
        with open(self._log()) as handle:
            self.assertEqual("$ ssaf-clang\n", handle.read())

    def test_what_a_tool_wrote_on_its_own_stderr_is_recorded_too(self):
        completed = types.SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"error: unknown argument\n"
        )
        with mock.patch("sys.stderr", io.StringIO()):
            self._log_failure(["clang", "-###"], completed)
        with open(self._log()) as handle:
            self.assertEqual("$ clang -###\nerror: unknown argument\n", handle.read())


if __name__ == "__main__":
    unittest.main()
