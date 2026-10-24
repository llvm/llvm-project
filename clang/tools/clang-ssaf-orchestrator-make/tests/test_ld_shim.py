# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest

from clang_ssaf_orchestrator_make import utils
from clang_ssaf_orchestrator_make.shims import ld
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository
from clang_ssaf_orchestrator_make import build_graph

from . import configs


class Inputs(unittest.TestCase):
    def test_objects(self):
        files = ld.identify_inputs_output(["ld", "-o", "prg", "a.o", "b.o", "-lm"])
        self.assertEqual(["a.o", "b.o"], files.inputs)
        self.assertEqual("prg", files.output)

    def test_any_object_is_named_whatever_precedes_it(self):
        for option in ("-force_load", "-weak_library", "-filelist", "-bundle"):
            files = ld.identify_inputs_output(["ld", "-o", "prg", "a.o", option, "b.o"])
            self.assertEqual(["a.o", "b.o"], files.inputs, option)

    def test_a_value_with_no_object_extension_is_not_named(self):
        files = ld.identify_inputs_output(["ld", "-o", "prg", "a.o", "-l", "x"])
        self.assertEqual(["a.o"], files.inputs)

    def test_a_link_with_no_objects_reads_as_nothing(self):
        self.assertIsNone(ld.identify_inputs_output(["ld", "-v"]))

    def test_bare_tool_reads_as_nothing(self):
        self.assertIsNone(ld.identify_inputs_output(["ld"]))


class Output(unittest.TestCase):
    def test_partial_link_output_is_not_read_back_as_an_input(self):
        files = ld.identify_inputs_output(
            ["ld", "-r", "-o", "combined.o", "a.o", "b.o"]
        )
        self.assertEqual("combined.o", files.output)
        self.assertEqual(["a.o", "b.o"], files.inputs)

    def test_only_a_whole_argument_names_the_output(self):
        for option in (
            "-object_path_lto",
            "-objc_abi_version",
            "-order_file",
            "-order_file_statistics",
            "-oso_prefix",
            "-object-file-name=x.o",
        ):
            files = ld.identify_inputs_output(["ld", "-o", "prg", "a.o", option])
            self.assertEqual("prg", files.output, option)

    def test_a_joined_output_is_not_read(self):
        files = ld.identify_inputs_output(["ld", "-obuild/prg", "a.o"])
        self.assertIsNone(files.output)


class SsafStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "project")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.root, "build"))
        self.ssaf_artifact_repo = configs.repository_for(self.out)
        self.graph = configs.build_graph_for(self.out)

    def _id(self, relative):
        return build_graph.BuildArtifactId(os.path.join(self.root, relative))

    def _summary(self, relative):
        open(
            utils.make_parents(
                self.ssaf_artifact_repo.get_path_for_artifact(
                    repository.get_ssaf_artifact_id(
                        build_graph.BuildArtifactId(os.path.join(self.root, relative)),
                        repository.SSAFArtifactKind.SUMMARY,
                    )
                )
            ),
            "w",
        ).close()

    def _run(self, arguments, pass_config=None):
        shim = configs.RecordingShim(
            ["ld", "-o", "build/prg"] + arguments, self.out, pass_config
        )
        with configs.recorded_notes() as notes:
            ld.invoke_ssaf_tools(shim, self.root)
        self.notes = notes
        return shim

    def _recorded_inputs(self):
        return self.graph.get_recorded_build_steps()[self._id("build/prg").get_path()]

    def test_only_what_the_run_produced_is_linked(self):
        self._summary("build/a.o")
        shim = self._run(["build/a.o", "third_party/prebuilt.o", "/usr/lib/crt1.o"])
        self.assertEqual([self._id("build/a.o").get_path()], self._recorded_inputs())
        self.assertEqual([], shim.missing)
        _, command = shim.commands[0]
        self.assertIn(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId(os.path.join(self.root, "build/a.o")),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            ),
            command,
        )

    def test_an_lto_scratch_object_is_not_an_input(self):
        self._summary("build/a.o")
        self._run(["build/a.o", "-object_path_lto", "/tmp/lto.o"])
        self.assertEqual([self._id("build/a.o").get_path()], self._recorded_inputs())

    def test_a_link_of_nothing_we_know_runs_no_step(self):
        shim = self._run(["build/a.o"])
        self.assertEqual([], shim.commands)
        self.assertEqual([], shim.missing)
        self.assertIn("nothing to link", self.notes[-1])

    def test_in_the_transformation_pass_the_edits_are_merged_instead(self):
        shim = self._run(["build/a.o"], configs.transformation_config())
        self.assertEqual(
            ["no edits feed %s, nothing to merge" % self._id("build/prg").get_path()],
            self.notes,
        )


if __name__ == "__main__":
    unittest.main()
