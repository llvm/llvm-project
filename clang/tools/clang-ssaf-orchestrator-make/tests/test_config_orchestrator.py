# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import config, config_orchestrator, options


class Executables:
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.elsewhere = os.path.join(self.tmp, "nowhere")

    def _executable(self, name):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path


class ResolveSsafTool(Executables, unittest.TestCase):
    def _resolve(self, option_value, installed_shim_dir_path=None, name="sh"):
        errors = []
        path = config_orchestrator._resolve_ssaf_tool(
            "--ssaf-clang",
            name,
            option_value,
            installed_shim_dir_path or self.elsewhere,
            errors,
        )
        return path, errors

    def test_no_value_takes_the_default_name_from_path(self):
        planted = self._executable("default-tool")
        with mock.patch.dict(os.environ, {"PATH": self.tmp}):
            path, errors = self._resolve(None, name="default-tool")
        self.assertEqual([], errors)
        self.assertEqual(planted, path)

    def test_the_tool_is_made_absolute(self):
        self._executable("cc")
        with mock.patch.dict(os.environ, {"PATH": self.tmp}):
            path, errors = self._resolve("cc")
        self.assertEqual([], errors)
        self.assertEqual(os.path.join(self.tmp, "cc"), path)

    def test_anything_but_one_executable_is_refused(self):
        tool = self._executable("cc")
        path, errors = self._resolve("%s -std=gnu11" % tool)
        self.assertIsNone(path)
        self.assertEqual(
            ["--ssaf-clang: expected one executable, got '%s -std=gnu11'" % tool],
            errors,
        )

    def test_a_missing_tool_is_reported(self):
        path, errors = self._resolve("no-such-tool-xyz")
        self.assertIsNone(path)
        self.assertIn("not found on PATH", errors[0])

    def test_a_directory_is_not_a_tool(self):
        path, errors = self._resolve(self.tmp)
        self.assertIsNone(path)
        self.assertIn("not a file", errors[0])

    def test_a_non_executable_is_reported(self):
        plain = os.path.join(self.tmp, "plain")
        with open(plain, "w") as handle:
            handle.write("")
        path, errors = self._resolve(plain)
        self.assertIsNone(path)
        self.assertIn("not executable", errors[0])

    def test_an_empty_value_is_reported_rather_than_run(self):
        path, errors = self._resolve("   ")
        self.assertIsNone(path)
        self.assertIn("nothing to run", errors[0])

    def test_one_of_our_own_shims_is_refused(self):
        shim = self._executable("cc")
        path, errors = self._resolve(shim, os.path.realpath(self.tmp))
        self.assertIsNone(path)
        self.assertIn("one of our own shims", errors[0])

    def test_refusals_accumulate_in_the_order_they_were_asked(self):
        errors = []
        for option_name in ("--ssaf-clang", "--ssaf-linker"):
            config_orchestrator._resolve_ssaf_tool(
                option_name, "sh", "/no/such/tool", self.elsewhere, errors
            )
        self.assertEqual(
            [
                "--ssaf-clang: not a file: /no/such/tool",
                "--ssaf-linker: not a file: /no/such/tool",
            ],
            errors,
        )


class Configure(Executables, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.clang = self._executable("clang")
        self.linker = self._executable("clang-ssaf-linker")
        self.analyzer = self._executable("clang-ssaf-analyzer")
        self.project = os.path.join(self.tmp, "project")
        os.makedirs(self.project)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.project)
        patch = mock.patch.object(
            config_orchestrator, "find_installed_shim_dir", return_value=self.elsewhere
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _configure(self, *extra, env=None):
        argv = [
            "--extract-summaries=PointerFlow",
            "--ssaf-clang=" + self.clang,
            "--ssaf-linker=" + self.linker,
        ]
        argv.extend(extra)
        return self._configure_argv(argv, env)

    def _configure_argv(self, argv, env=None):
        command_line_opts, problem = options.resolve(argv)
        self.assertIsNone(problem)
        configs, problem = config_orchestrator.configure(
            command_line_opts, {} if env is None else env
        )
        return (configs or (None, None)) + (problem,)

    def test_a_valid_command_line_yields_the_configuration(self):
        workflow_config, analysis_config, problem = self._configure("--", "make", "-j8")
        self.assertIsNone(problem)
        self.assertEqual(["make", "-j8"], workflow_config.build_command)
        self.assertEqual("PointerFlow", analysis_config.extractors)
        self.assertEqual(self.clang, analysis_config.ssaf_clang)

    def test_the_configuration_holds_the_resolved_tool_not_what_was_typed(self):
        argv = [
            "--extract-summaries=PointerFlow",
            "--ssaf-clang=clang",
            "--ssaf-linker=clang-ssaf-linker",
            "--ssaf-analyzer=clang-ssaf-analyzer",
            "--analysis=a",
            "--",
            "make",
        ]
        with mock.patch.dict(os.environ, {"PATH": self.tmp}):
            workflow_config, analysis_config, problem = self._configure_argv(argv)
        self.assertIsNone(problem)
        self.assertEqual(self.clang, analysis_config.ssaf_clang)
        self.assertEqual(self.linker, analysis_config.ssaf_linker)
        self.assertEqual(self.analyzer, workflow_config.ssaf_analyzer)

    def test_the_environment_reaches_the_config_but_not_the_file(self):
        workflow_config, _, problem = self._configure(
            "--", "make", env={"PATH": "/usr/bin", "SECRET": "x"}
        )
        self.assertIsNone(problem)
        self.assertEqual("x", workflow_config.build_environment["SECRET"])
        self.assertEqual("/usr/bin", workflow_config.restored_path_env_var)
        self.assertNotIn("SECRET", json.dumps(workflow_config.to_json()))

    def test_an_output_dir_holding_the_project_is_refused(self):
        for value in (".", "..", os.getcwd()):
            workflow_config, _, problem = self._configure("-o", value, "--", "make")
            self.assertIsNone(workflow_config, value)
            self.assertEqual(options.ENVIRONMENT_ERROR, problem.exit_status)
            self.assertIn("would contain the project", problem.message)

    def test_a_directory_no_run_made_is_not_deleted(self):
        other = os.path.join(self.project, "src")
        os.makedirs(other)
        open(os.path.join(other, "main.c"), "w").close()
        workflow_config, _, problem = self._configure("-o", other, "--", "make")
        self.assertIsNone(workflow_config)
        self.assertEqual(options.ENVIRONMENT_ERROR, problem.exit_status)
        self.assertIn("no earlier run made it", problem.message)
        self.assertEqual(["main.c"], os.listdir(other))

    def test_an_empty_directory_is_taken(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        workflow_config, _, problem = self._configure("-o", empty, "--", "make")
        self.assertIsNone(problem)
        self.assertEqual(os.path.realpath(empty), workflow_config.output_dir)

    def test_a_directory_an_earlier_run_made_is_taken(self):
        earlier = os.path.join(self.tmp, "earlier")
        os.makedirs(os.path.join(earlier, "ssaf-artifacts"))
        open(config.output_dir_marker_path(earlier), "w").close()
        _, _, problem = self._configure("-o", earlier, "--", "make")
        self.assertIsNone(problem)

    def test_a_relative_output_dir_is_made_absolute(self):
        workflow_config, _, problem = self._configure("-o", "somewhere", "--", "make")
        self.assertIsNone(problem)
        self.assertEqual(
            os.path.join(os.getcwd(), "somewhere"), workflow_config.output_dir
        )

    def test_an_unresolvable_ssaf_tool(self):
        workflow_config, _, problem = self._configure_argv(
            [
                "--extract-summaries=PointerFlow",
                "--ssaf-clang=/no/such/clang",
                "--ssaf-linker=" + self.linker,
                "--",
                "make",
            ]
        )
        self.assertIsNone(workflow_config)
        self.assertEqual(options.ENVIRONMENT_ERROR, problem.exit_status)
        self.assertEqual(1, len(problem.details))
        self.assertIn("/no/such/clang", problem.details[0])

    def test_the_ssaf_tools_take_no_flags(self):
        workflow_config, _, problem = self._configure_argv(
            [
                "--extract-summaries=PointerFlow",
                "--ssaf-clang=%s -O2" % self.clang,
                "--ssaf-linker=" + self.linker,
                "--",
                "make",
            ]
        )
        self.assertIsNone(workflow_config)
        self.assertEqual(
            ["--ssaf-clang: expected one executable, got '%s -O2'" % self.clang],
            problem.details,
        )

    def test_a_detail_names_the_option_and_the_path(self):
        *_, problem = self._configure_argv(
            [
                "--extract-summaries=PointerFlow",
                "--ssaf-clang=/no/such/clang",
                "--ssaf-linker=" + self.linker,
                "--",
                "make",
            ]
        )
        self.assertEqual(["--ssaf-clang: not a file: /no/such/clang"], problem.details)

    def test_a_tool_absent_from_path_names_what_was_looked_for(self):
        with mock.patch.dict(os.environ, {"PATH": self.elsewhere}):
            workflow_config, _, problem = self._configure_argv(
                [
                    "--extract-summaries=PointerFlow",
                    "--ssaf-clang=" + self.clang,
                    "--",
                    "make",
                ]
            )
        self.assertIsNone(workflow_config)
        self.assertEqual(
            ["--ssaf-linker: clang-ssaf-linker: not found on PATH"],
            problem.details,
        )

    def test_every_unresolvable_tool_is_reported_at_once(self):
        workflow_config, _, problem = self._configure_argv(
            [
                "--extract-summaries=PointerFlow",
                "--ssaf-clang=/no/such/clang",
                "--ssaf-linker=/no/such/linker",
                "--ssaf-analyzer=/no/such/analyzer",
                "--analysis=a",
                "--",
                "make",
            ]
        )
        self.assertIsNone(workflow_config)
        self.assertEqual(
            [
                "--ssaf-clang: not a file: /no/such/clang",
                "--ssaf-linker: not a file: /no/such/linker",
                "--ssaf-analyzer: not a file: /no/such/analyzer",
            ],
            problem.details,
        )

    def test_the_analyzer_is_only_required_when_an_analysis_is_asked_for(self):
        workflow_config, _, problem = self._configure("--", "make")
        self.assertIsNone(problem)
        self.assertIsNone(workflow_config.ssaf_analyzer)

    def test_a_tool_no_pass_will_run_is_not_resolved_at_all(self):
        workflow_config, _, problem = self._configure(
            "--ssaf-analyzer=/no/such/analyzer", "--", "make"
        )
        self.assertIsNone(problem)
        self.assertIsNone(workflow_config.ssaf_analyzer)


class InstalledShims(Executables, unittest.TestCase):
    def _found_from(self, script):
        entry = types.SimpleNamespace(__file__=script)
        with mock.patch.dict(sys.modules, {"__main__": entry}):
            return config_orchestrator.find_installed_shim_dir()

    def test_an_installed_layout_finds_libexec(self):
        prefix = os.path.join(self.tmp, "prefix")
        libexec = os.path.join(prefix, "libexec", "ssaf-orchestrator-shims")
        os.makedirs(libexec)
        os.makedirs(os.path.join(prefix, "bin"))
        found = self._found_from(
            os.path.join(prefix, "bin", "clang-ssaf-orchestrator-make")
        )
        self.assertEqual(os.path.realpath(libexec), found)

    def test_a_source_tree_falls_back_to_shims(self):
        root = os.path.join(self.tmp, "tool")
        os.makedirs(os.path.join(root, "bin"))
        found = self._found_from(
            os.path.join(root, "bin", "clang-ssaf-orchestrator-make")
        )
        self.assertEqual(os.path.realpath(os.path.join(root, "shims")), found)

    def test_the_result_is_resolved_so_the_own_shim_guard_can_compare_it(self):
        prefix = os.path.join(self.tmp, "real")
        libexec = os.path.join(prefix, "libexec", "ssaf-orchestrator-shims")
        os.makedirs(libexec)
        os.makedirs(os.path.join(prefix, "bin"))
        through = os.path.join(self.tmp, "through")
        os.symlink(prefix, through)
        found = self._found_from(
            os.path.join(through, "bin", "clang-ssaf-orchestrator-make")
        )
        self.assertEqual(os.path.realpath(found), found)
        self.assertEqual(os.path.realpath(libexec), found)


if __name__ == "__main__":
    unittest.main()
