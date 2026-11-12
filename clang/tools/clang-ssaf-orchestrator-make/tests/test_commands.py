# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest

from clang_ssaf_orchestrator_make import build_graph, utils
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository

from . import configs
from clang_ssaf_orchestrator_make.shims import commands


class Extensions(unittest.TestCase):
    def test_object_extensions(self):
        for name in ("a.o", "build/a.o", "a.obj", "A.O".lower()):
            self.assertTrue(commands.is_object(name), name)
        for name in ("a.a", "a.c", "a", "a.o.tmp", "a.O"):
            self.assertFalse(commands.is_object(name), name)

    def test_archive_extensions(self):
        for name in ("libx.a", "build/libx.a"):
            self.assertTrue(commands.is_archive(name), name)
        for name in ("libx.so", "libx.dylib", "libx", "libx.a.tmp"):
            self.assertFalse(commands.is_archive(name), name)


class Unique(unittest.TestCase):
    def test_the_first_of_each_is_kept_in_order(self):
        self.assertEqual(["b", "a", "c"], commands.unique(["b", "a", "b", "c", "a"]))

    def test_nothing_stays_nothing(self):
        self.assertEqual([], commands.unique([]))

    def test_the_input_is_not_modified(self):
        items = ["a", "a"]
        commands.unique(items)
        self.assertEqual(["a", "a"], items)


class InputsAndOutput(unittest.TestCase):
    def test_it_carries_what_went_in_and_what_came_out(self):
        files = commands.Files(["a.o"], "prg")
        self.assertEqual(["a.o"], files.inputs)
        self.assertEqual("prg", files.output)


class _OverABuild(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out, True)
        self.ssaf_artifact_repo = configs.repository_for(self.out)
        self.build_graph = configs.build_graph_for(self.out)

    def _path(self, relative):
        return os.path.join(self.root, relative)

    def _id(self, relative):
        return build_graph.BuildArtifactId(self._path(relative))

    def _write(self, relative, ssaf_artifact_kind):
        path = self.ssaf_artifact_repo.get_path_for_artifact(
            repository.get_ssaf_artifact_id(self._id(relative), ssaf_artifact_kind)
        )
        open(utils.make_parents(path), "w").close()
        return path


class SummaryOf(_OverABuild):
    def test_an_object_resolves_to_its_own_summary(self):
        want = self._write("build/a.o", repository.SSAFArtifactKind.SUMMARY)
        self.assertEqual(
            want, commands.summary_of(self.ssaf_artifact_repo, self._id("build/a.o"))
        )

    def test_an_artifact_the_build_did_not_produce_resolves_to_nothing(self):
        self.assertIsNone(
            commands.summary_of(
                self.ssaf_artifact_repo,
                build_graph.BuildArtifactId("/usr/lib/libSystem.dylib"),
            )
        )

    def test_an_assembled_artifact_does_not_reach_past_its_own(self):
        self._write("build/a.o", repository.SSAFArtifactKind.SUMMARY)
        self.build_graph.record_build_step(
            self._id("build/lib.a"), [self._id("build/a.o")]
        )
        self.assertIsNone(
            commands.summary_of(self.ssaf_artifact_repo, self._id("build/lib.a"))
        )


class FilterBuildArtifacts(_OverABuild):
    def _kept(self, *files):
        return commands.filter_build_artifacts(
            self.build_graph, self.ssaf_artifact_repo, self.root, list(files)
        )

    def test_a_compiled_object_is_kept(self):
        self._write("build/a.o", repository.SSAFArtifactKind.SUMMARY)
        self.assertEqual([self._id("build/a.o")], self._kept("build/a.o"))

    def test_an_object_with_only_source_edits_is_kept(self):
        self._write("build/a.o", repository.SSAFArtifactKind.EDITS)
        self.assertEqual([self._id("build/a.o")], self._kept("build/a.o"))

    def test_an_archive_with_only_a_record_is_kept(self):
        self.build_graph.record_build_step(
            self._id("build/x.a"), [self._id("build/a.o")]
        )
        self.assertEqual([self._id("build/x.a")], self._kept("build/x.a"))

    def test_what_the_build_never_touched_is_dropped(self):
        self.assertEqual([], self._kept("third_party/prebuilt.a", "/usr/lib/libz.a"))


if __name__ == "__main__":
    unittest.main()
