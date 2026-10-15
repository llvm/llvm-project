# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest

from clang_ssaf_orchestrator_make import utils
from clang_ssaf_orchestrator_make.failure_logs import SSAFFailureKind
from clang_ssaf_orchestrator_make.shims import ar
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository
from clang_ssaf_orchestrator_make import build_graph

from . import configs


class SsafStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "project")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.root, "build"))

    def _shim(self, argv, pass_config=None):
        return configs.RecordingShim(argv, self.out, pass_config)

    def _id(self, relative):
        return build_graph.BuildArtifactId(os.path.join(self.root, relative))

    def _summary(self, relative):
        records = configs.repository_for(self.out)
        open(
            utils.make_parents(
                records.get_path_for_artifact(
                    repository.get_ssaf_artifact_id(
                        build_graph.BuildArtifactId(os.path.join(self.root, relative)),
                        repository.SSAFArtifactKind.SUMMARY,
                    )
                )
            ),
            "w",
        ).close()

    def test_a_command_that_creates_no_archive_runs_no_step(self):
        shim = self._shim(["ar", "t", "build/libx.a"])
        with configs.recorded_notes() as notes:
            ar.invoke_ssaf_tools(shim, self.root)
        self.assertEqual([], shim.commands)
        self.assertIn("not archive creation", notes[0])

    def test_a_member_the_run_never_touched_is_not_an_input(self):
        shim = self._shim(["ar", "rcs", "build/libx.a", "build/a.o"])
        ar.invoke_ssaf_tools(shim, self.root)
        self.assertEqual([], shim.commands)
        self.assertEqual([], shim.missing)
        libx = self._id("build/libx.a")
        self.assertTrue(configs.build_graph_for(self.out).was_built(libx))
        self.assertEqual(
            [],
            configs.build_graph_for(self.out).get_recorded_build_steps()[
                libx.get_path()
            ],
        )

    def test_the_summaries_that_exist_are_handed_to_the_linker(self):
        self._summary("build/a.o")
        shim = self._shim(["ar", "rcs", "build/libx.a", "build/a.o"])
        ar.invoke_ssaf_tools(shim, self.root)
        self.assertEqual(1, len(shim.commands))
        failure, command = shim.commands[0]
        self.assertEqual((SSAFFailureKind.ARCHIVE, self._id("build/libx.a")), failure)
        self.assertEqual(
            ["/bin/clang-ssaf-linker", "static-library", "create"], command[:3]
        )
        self.assertIn("--namespace=libx.a", command)

    def test_the_namespace_is_the_archives_file_name_not_its_path(self):
        self._summary("build/a.o")
        shim = self._shim(["ar", "rcs", "build/libx.a", "build/a.o"])
        ar.invoke_ssaf_tools(shim, self.root)
        _, command = shim.commands[0]
        self.assertNotIn("--namespace=build/libx.a", command)

    def test_a_member_that_is_not_an_object_is_not_an_input(self):
        self._summary("build/a.o")
        shim = self._shim(["ar", "rcs", "build/libx.a", "build/a.o", "README"])
        ar.invoke_ssaf_tools(shim, self.root)
        self.assertEqual([], shim.missing)
        self.assertEqual(
            [self._id("build/a.o").get_path()],
            configs.build_graph_for(self.out).get_recorded_build_steps()[
                self._id("build/libx.a").get_path()
            ],
        )

    def test_in_the_transformation_pass_the_edits_are_merged_instead(self):
        shim = self._shim(
            ["ar", "rcs", "build/libx.a", "build/a.o"],
            configs.transformation_config(),
        )
        with configs.recorded_notes() as notes:
            ar.invoke_ssaf_tools(shim, self.root)
        self.assertEqual(
            [
                "no edits feed %s, nothing to merge"
                % self._id("build/libx.a").get_path()
            ],
            notes,
        )


class Archive(unittest.TestCase):
    def test_archive(self):
        files = ar.identify_inputs_output(["ar", "rcs", "libx.a", "a.o", "b.o"])
        self.assertEqual("libx.a", files.output)
        self.assertEqual(["a.o", "b.o"], files.inputs)

    def test_leading_dash_operation(self):
        files = ar.identify_inputs_output(["ar", "-rcs", "libx.a", "a.o"])
        self.assertEqual(["a.o"], files.inputs)

    def test_plugin_option_is_not_the_operation(self):
        files = ar.identify_inputs_output(
            ["ar", "--plugin", "/usr/lib/liblto.so", "rcs", "libx.a", "a.o"]
        )
        self.assertEqual(["a.o"], files.inputs)

    def test_non_creating_operations_read_as_nothing(self):
        self.assertIsNone(ar.identify_inputs_output(["ar", "t", "libx.a"]))

    def test_every_creating_operation_is_recognised(self):
        for operation in ("r", "q", "m"):
            files = ar.identify_inputs_output(["ar", operation + "cs", "libx.a", "a.o"])
            self.assertEqual(["a.o"], files.inputs, operation)
            self.assertEqual("libx.a", files.output, operation)

    def test_non_archive_output_reads_as_nothing(self):
        self.assertIsNone(ar.identify_inputs_output(["ar", "rcs", "notlib", "a.o"]))

    def test_too_few_arguments(self):
        self.assertIsNone(ar.identify_inputs_output(["ar", "rcs"]))


if __name__ == "__main__":
    unittest.main()
