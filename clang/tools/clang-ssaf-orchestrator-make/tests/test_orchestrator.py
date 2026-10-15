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

from clang_ssaf_orchestrator_make import build_graph, config, config_orchestrator
from clang_ssaf_orchestrator_make import config_shims, options, orchestrator, utils
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository

from . import configs


class MakeWarning(unittest.TestCase):
    def _warning(self, build_command):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            orchestrator.warn_if_not_make(build_command)
        return stream.getvalue()

    def test_make_is_silent(self):
        self.assertEqual("", self._warning(["make"]))

    def test_gmake_is_silent(self):
        self.assertEqual("", self._warning(["gmake", "-j8"]))

    def test_an_absolute_make_is_silent(self):
        self.assertEqual("", self._warning(["/usr/bin/make", "-C", "sub"]))

    def test_anything_else_warns_but_does_not_refuse(self):
        warning = self._warning(["./build.sh"])
        self.assertIn("./build.sh", warning)
        self.assertIn("not make", warning)
        self.assertIn("-B", warning)

    def test_an_env_prefix_is_looked_past_not_objected_to(self):
        self.assertEqual("", self._warning(["env", "V=1", "make"]))
        self.assertEqual("", self._warning(["V=1", "make", "-j8"]))

    def test_the_warning_names_the_word_that_is_not_make(self):
        self.assertIn("'ninja' is not make", self._warning(["env", "V=1", "ninja"]))
        self.assertIn("'cmake' is not make", self._warning(["cmake", "--build", "."]))

    def test_a_command_of_nothing_but_assignments_says_nothing(self):
        for build_command in ([], ["V=1"], ["env", "V=1"]):
            self.assertEqual("", self._warning(build_command), build_command)

    def test_an_env_prefix_around_something_else_still_warns(self):
        self.assertIn("not make", self._warning(["env", "V=1", "ninja"]))

    def test_the_warning_names_what_will_be_appended(self):
        self.assertIn("-B", self._warning(["ninja"]))


class Main(unittest.TestCase):
    def _main(self, argv):
        self.run_calls = []
        with mock.patch("sys.stderr", io.StringIO()) as stream, mock.patch.object(
            orchestrator,
            "run_ssaf_workflow",
            lambda *args: self.run_calls.append(args),
        ):
            return orchestrator.main(argv, {}), stream.getvalue()

    def test_a_usage_error_stops_before_the_build(self):
        status, message = self._main(["--extract-summaries=PointerFlow"])
        self.assertEqual(options.USAGE_ERROR, status)
        self.assertIn("no build command", message)
        self.assertEqual([], self.run_calls)

    def test_an_environment_error_stops_before_the_build(self):
        status, message = self._main(
            [
                "--extract-summaries=PointerFlow",
                "--ssaf-clang=/no/such/clang",
                "--",
                "make",
            ]
        )
        self.assertEqual(options.ENVIRONMENT_ERROR, status)
        self.assertIn("--ssaf-clang", message)
        self.assertEqual([], self.run_calls)

    def test_a_configured_run_is_handed_its_three_configurations(self):
        configured = ("workflow", "analysis", "transformation")
        with mock.patch.object(
            config_orchestrator, "configure", return_value=(configured, None)
        ):
            status, _ = self._main(["--extract-summaries=X", "--", "make"])
        self.assertEqual(0, status)
        self.assertEqual([configured], self.run_calls)


class ClearOutputDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_what_a_previous_run_left_is_removed(self):
        os.makedirs(os.path.join(self.tmp, "shims"))
        open(os.path.join(self.tmp, "wpa.json"), "w").close()
        orchestrator.clear_output_dir(self.tmp)
        self.assertEqual([config.OUTPUT_DIR_MARKER], os.listdir(self.tmp))

    def test_an_output_directory_that_is_not_there_yet_is_made(self):
        output_dir = os.path.join(self.tmp, "out")
        orchestrator.clear_output_dir(output_dir)
        self.assertEqual([config.OUTPUT_DIR_MARKER], os.listdir(output_dir))


class PassConfigFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.workflow_config = config.SSAFWorkflowConfig(
            output_dir=self.tmp,
            installed_shim_dir_path="/libexec",
            build_command=["make"],
        )
        self.analysis_config = config.SSAFAnalysisPassConfig(
            ssaf_clang="/bin/clang",
            ssaf_extra_clang_args=["-DX"],
            extractors="PointerFlow",
            ssaf_linker="/bin/clang-ssaf-linker",
            analyses=["A"],
        )
        self.transformation_config = config.SSAFTransformationPassConfig(
            ssaf_clang="/bin/clang",
            ssaf_extra_clang_args=["-DX"],
            ssaf_src_edit_merge="/bin/clang-ssaf-src-edit-merge",
            transformation="x",
        )
        self.transformation_config.name_link_unit("build/prg")

    def test_each_pass_is_written_under_the_output_directory(self):
        for pass_config in (self.analysis_config, self.transformation_config):
            path = orchestrator.write_pass_config(self.workflow_config, pass_config)
            self.assertEqual(os.path.join(self.tmp, pass_config.file_name), path)
            self.assertTrue(os.path.isfile(path))

    def test_nothing_is_left_behind_by_the_write(self):
        orchestrator.write_pass_config(self.workflow_config, self.analysis_config)
        self.assertEqual(["config-analysis.json"], os.listdir(self.tmp))

    def test_both_passes_can_be_written_and_coexist(self):
        for pass_config in (self.analysis_config, self.transformation_config):
            orchestrator.write_pass_config(self.workflow_config, pass_config)
        self.assertEqual(
            ["config-analysis.json", "config-transformation.json"],
            sorted(os.listdir(self.tmp)),
        )

    def test_the_document_does_not_carry_its_own_path(self):
        path = orchestrator.write_pass_config(
            self.workflow_config, self.analysis_config
        )
        with open(path) as handle:
            written = json.load(handle)
        self.assertNotIn("serialized_config_path", written)
        self.assertNotIn(path, written.values())

    def test_an_output_directory_that_does_not_exist_is_not_silently_skipped(self):
        workflow_config = configs.workflow_config(
            output_dir=os.path.join(self.tmp, "absent")
        )
        with self.assertRaises(OSError):
            orchestrator.write_pass_config(workflow_config, self.analysis_config)

    def test_a_shim_reads_back_the_run_and_its_own_pass_whole(self):
        for written in (self.analysis_config, self.transformation_config):
            path = orchestrator.write_pass_config(self.workflow_config, written)
            workflow_config, pass_config = config_shims.bootstrap_from_env(
                {config.SSAF_CONFIG_ENV_VAR: path}
            )
            self.assertEqual(self.workflow_config.to_json(), workflow_config.to_json())
            self.assertIs(type(written), type(pass_config))
            self.assertEqual(written.to_json(), pass_config.to_json())


class RunBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.workflow_config = configs.workflow_config(output_dir=self.tmp)
        self.workflow_config.build_command = ["make", "-j8"]

    def _run(self, **kwargs):
        with mock.patch("subprocess.call", **kwargs) as called:
            self.called = called
            return orchestrator.run_build_with_toolchain_interposition(
                self.workflow_config, "/out/config.json", "/out/shims", "build"
            )

    def test_a_build_that_succeeds_is_no_problem(self):
        self.assertIsNone(self._run(return_value=0))

    def test_the_rebuild_flag_is_appended_to_the_build_command(self):
        self._run(return_value=0)
        self.assertEqual(["make", "-j8", "-B"], self.called.call_args[0][0])

    def test_the_build_runs_where_we_were_invoked_and_sees_the_shims(self):
        self._run(return_value=0)
        self.assertNotIn("cwd", self.called.call_args[1])
        env = self.called.call_args[1]["env"]
        self.assertTrue(env["PATH"].startswith("/out/shims"))
        self.assertEqual("/out/config.json", env[config.SSAF_CONFIG_ENV_VAR])

    def test_a_build_that_fails_carries_its_own_status(self):
        problem = self._run(return_value=2)
        self.assertEqual(2, problem.exit_status)
        self.assertEqual("build failed (exit 2)", problem.message)

    def test_a_build_killed_by_a_signal_exits_as_a_shell_would(self):
        problem = self._run(return_value=-15)
        self.assertEqual(143, problem.exit_status)

    def test_a_build_command_that_cannot_be_run_at_all(self):
        problem = self._run(side_effect=OSError(2, "No such file"))
        self.assertEqual(127, problem.exit_status)
        self.assertIn("cannot run 'make'", problem.message)

    def test_the_description_names_which_build_failed(self):
        with mock.patch("subprocess.call", return_value=1):
            problem = orchestrator.run_build_with_toolchain_interposition(
                self.workflow_config, "/c.json", "/s", "rebuild for something"
            )
        self.assertEqual("rebuild for something failed (exit 1)", problem.message)


class TheAnalysisPass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.workflow_config = configs.workflow_config(output_dir=self.tmp)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)
        prg = build_graph.BuildArtifactId("/p/prg")
        configs.build_graph_for(self.tmp).record_build_step(prg, [])
        self.summary = utils.make_parents(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    prg, repository.SSAFArtifactKind.SUMMARY
                )
            )
        )
        with open(self.summary, "w") as handle:
            json.dump({"lu_namespace": [{"kind": "LinkUnit", "name": "prg"}]}, handle)

    def _run(self, analyses, status=0):
        with mock.patch.object(
            orchestrator, "run_build_with_toolchain_interposition", return_value=None
        ), mock.patch("subprocess.call", return_value=status) as called, mock.patch(
            "sys.stdout", io.StringIO()
        ):
            problem = orchestrator.run_ssaf_analysis_pass(
                self.workflow_config,
                configs.analysis_config(analyses=analyses),
                self.ssaf_artifact_repo,
                "/out/shims",
            )
        return problem, called

    def test_without_an_analysis_no_analyzer_runs(self):
        problem, called = self._run([])
        self.assertIsNone(problem)
        called.assert_not_called()

    def test_the_analyzer_runs_every_analysis_over_the_link_unit_summary(self):
        problem, called = self._run(["A", "B"])
        self.assertIsNone(problem)
        self.assertEqual(
            [
                "/bin/clang-ssaf-analyzer",
                self.summary,
                "-o",
                config.wpa_result_path(self.tmp),
                "-a",
                "A",
                "-a",
                "B",
            ],
            called.call_args[0][0],
        )

    def test_a_failed_analysis_fails_the_run_with_its_own_status(self):
        problem, _ = self._run(["A"], status=3)
        self.assertEqual(3, problem.exit_status)
        self.assertEqual("whole-program analysis failed (exit 3)", problem.message)

    def test_an_analyzer_killed_by_a_signal_exits_as_a_shell_would(self):
        problem, _ = self._run(["A"], status=-9)
        self.assertEqual(137, problem.exit_status)


class TopMostLinkUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)
        self.graph = configs.build_graph_for(self.tmp)

    def _record(self, output_path, inputs):
        inputs = [build_graph.BuildArtifactId(each) for each in inputs]
        self.graph.record_build_step(
            build_graph.BuildArtifactId(output_path),
            inputs,
        )

    def _summary(self, build_artifact_path, document=None):
        path = utils.make_parents(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId(build_artifact_path),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            )
        )
        if document is None:
            document = {
                "lu_namespace": [
                    {"kind": "LinkUnit", "name": os.path.basename(build_artifact_path)}
                ]
            }
        with open(path, "w") as handle:
            json.dump(document, handle)

    def test_no_link_unit_at_all(self):
        link_unit, _, _, problem = orchestrator.get_topmost_link_unit(
            self.ssaf_artifact_repo, self.tmp
        )
        self.assertIsNone(link_unit)
        self.assertIn("produced no link unit", problem.message)

    def test_several_that_consume_none_of_each_other(self):
        self._record("/p/one", ["a.o"])
        self._record("/p/two", ["b.o"])
        link_unit, _, _, problem = orchestrator.get_topmost_link_unit(
            self.ssaf_artifact_repo, self.tmp
        )
        self.assertIsNone(link_unit)
        self.assertIn("produced 2 unconsumed outputs", problem.message)
        self.assertIn("p/one", problem.message)
        self.assertIn("p/two", problem.message)

    def test_one_whose_summary_was_never_written(self):
        self._record("/p/prg", ["a.o"])
        link_unit, _, _, problem = orchestrator.get_topmost_link_unit(
            self.ssaf_artifact_repo, self.tmp
        )
        self.assertIsNone(link_unit)
        self.assertIn("no summary", problem.message)
        self.assertIn("p/prg", problem.message)

    def test_one_whose_summary_names_no_link_unit(self):
        self._record("/p/libx.a", ["a.o"])
        self._summary(
            "/p/libx.a", {"namespace": {"kind": "StaticLibrary", "name": "libx.a"}}
        )
        link_unit, _, _, problem = orchestrator.get_topmost_link_unit(
            self.ssaf_artifact_repo, self.tmp
        )
        self.assertIsNone(link_unit)
        self.assertIn("is not a link unit", problem.message)
        self.assertIn("p/libx.a", problem.message)

    def test_one_with_its_summary_is_the_answer(self):
        self._record("/p/prg", ["a.o"])
        self._summary("/p/prg")
        link_unit, _, unit_id, problem = orchestrator.get_topmost_link_unit(
            self.ssaf_artifact_repo, self.tmp
        )
        self.assertIsNone(problem)
        self.assertEqual("/p/prg", link_unit.get_path())
        self.assertEqual("prg", unit_id)


if __name__ == "__main__":
    unittest.main()
