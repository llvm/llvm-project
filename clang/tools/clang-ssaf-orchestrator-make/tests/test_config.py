# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import unittest

from clang_ssaf_orchestrator_make import config

from . import configs


class Passes(unittest.TestCase):
    def test_a_pass_names_its_own_configuration_file(self):
        self.assertEqual("config-analysis.json", configs.analysis_config().file_name)

    def test_a_pass_kind_is_stored_as_its_plain_string(self):
        written = json.loads(json.dumps(configs.analysis_config().to_json()))
        self.assertEqual("analysis", written["pass_kind"])

    def test_a_pass_carries_only_the_tools_it_runs(self):
        self.assertEqual(
            {
                "pass_kind",
                "ssaf_clang",
                "ssaf_extra_clang_args",
                "extractors",
                "ssaf_linker",
                "analyses",
            },
            set(configs.analysis_config().to_json()),
        )

    def test_a_pass_knows_nothing_of_the_run_it_belongs_to(self):
        run_only = config.SSAFWorkflowConfig.written_field_names()
        self.assertEqual(set(), run_only & set(configs.analysis_config().to_json()))


class OutputDirPaths(unittest.TestCase):
    def test_the_shim_links_and_the_analysis_result_sit_at_the_root(self):
        self.assertEqual("/out/shims", config.shim_link_dir("/out"))
        self.assertEqual("/out/wpa.json", config.wpa_result_path("/out"))

    def test_each_owner_has_a_directory_of_its_own(self):
        self.assertEqual("/out/build-graph", config.build_graph_dir("/out"))
        self.assertEqual("/out/ssaf-artifacts", config.ssaf_artifact_dir("/out"))
        self.assertEqual("/out/failure-logs", config.failure_log_dir("/out"))


class WorkflowConfig(unittest.TestCase):
    def test_it_carries_what_is_true_of_the_run_whichever_pass_is_running(self):
        self.assertEqual(
            {
                "output_dir",
                "installed_shim_dir_path",
                "restored_path_env_var",
                "verbose",
                "build_command",
                "ssaf_analyzer",
                "keep_going",
            },
            set(configs.workflow_config().to_json()),
        )

    def test_the_build_environment_is_never_written(self):
        workflow_config = configs.workflow_config(
            env={"SECRET": "x", "PATH": "/usr/bin"}
        )
        self.assertEqual("x", workflow_config.build_environment["SECRET"])
        self.assertNotIn("build_environment", workflow_config.to_json())
        self.assertNotIn("SECRET", json.dumps(workflow_config.to_json()))

    def test_every_written_key_is_also_reachable_as_an_attribute(self):
        workflow_config = configs.workflow_config()
        written = workflow_config.to_json()
        self.assertEqual(
            written,
            {name: getattr(workflow_config, name) for name in written},
        )

    def test_an_absent_path_is_empty_so_the_live_path_is_never_searched(self):
        self.assertEqual("", config.SSAFWorkflowConfig().restored_path_env_var)


if __name__ == "__main__":
    unittest.main()
