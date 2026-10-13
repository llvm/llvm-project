# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import build_graph, utils
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository

from . import configs


def _id(build_artifact_path):
    return build_graph.BuildArtifactId(build_artifact_path)


class BuildArtifactIds(unittest.TestCase):
    def test_an_id_is_the_path_without_the_leading_separator(self):
        self.assertEqual(
            "p/build/foo.o",
            build_graph.BuildArtifactId("/p/build/foo.o").get_path().lstrip(os.sep),
        )

    def test_the_id_is_independent_of_how_the_path_was_spelled(self):
        self.assertEqual(
            build_graph.BuildArtifactId("/p/build/foo.o"),
            build_graph.BuildArtifactId("/p/./build/../build/foo.o"),
        )

    def test_nothing_is_left_to_climb_out_of_the_output_directory_with(self):
        build_artifact_id = build_graph.BuildArtifactId("/elsewhere/../deep/foo.o")
        self.assertEqual("/deep/foo.o", build_artifact_id.get_path())
        self.assertNotIn("..", build_artifact_id.get_path())

    def test_two_paths_are_two_ids(self):
        first = build_graph.BuildArtifactId("/a/foo.o")
        second = build_graph.BuildArtifactId("/b/foo.o")
        self.assertNotEqual(first, second)
        self.assertEqual(first, build_graph.BuildArtifactId("/a/foo.o"))

    def test_same_basename_in_different_directories_do_not_collide(self):
        self.assertNotEqual(
            build_graph.BuildArtifactId("/p/build/arm64/parser.o"),
            build_graph.BuildArtifactId("/p/build/x86_64/parser.o"),
        )
        self.assertEqual(
            "p/build/arm64/parser.o",
            build_graph.BuildArtifactId("/p/build/arm64/parser.o")
            .get_path()
            .lstrip(os.sep),
        )

    def test_no_part_of_the_path_is_elided(self):
        self.assertEqual(
            "p/a/very/deep/nested/dir/parser.o",
            build_graph.BuildArtifactId("/p/a/very/deep/nested/dir/parser.o")
            .get_path()
            .lstrip(os.sep),
        )


class SymlinkedPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.real = os.path.join(self.tmp, "real")
        os.makedirs(os.path.join(self.real, "build"))
        self.link = os.path.join(self.tmp, "link")
        os.symlink(self.real, self.link)

    def test_an_artifact_reached_through_a_symlink_gets_the_real_path_id(self):
        directly = os.path.join(self.real, "build", "a.o")
        self.assertEqual(
            directly,
            build_graph.BuildArtifactId(
                os.path.join(self.link, "build", "a.o")
            ).get_path(),
        )


class BuildstepRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)
        self.graph = configs.build_graph_for(self.tmp)

    def test_round_trip(self):
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/build/prg"),
            [_id("/a.o"), _id("/b.o")],
        )
        self.assertEqual(
            {"/p/build/prg": ["/a.o", "/b.o"]}, self.graph.get_recorded_build_steps()
        )

    def test_several_build_steps_in_nested_directories(self):
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/libx.a"),
            [_id("/a.o")],
        )
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/build/sub/prg"),
            [_id("/p/libx.a")],
        )
        self.assertEqual(
            {"/p/libx.a", "/p/build/sub/prg"},
            set(self.graph.get_recorded_build_steps()),
        )

    def test_one_output_has_one_record_and_the_last_call_wins(self):
        first = self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/libx.a"),
            [_id("/a.o")],
        )
        second = self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/libx.a"),
            [_id("/b.o")],
        )
        self.assertEqual(first, second)
        self.assertEqual(["/b.o"], self.graph.get_recorded_build_steps()["/p/libx.a"])

    def test_a_record_whose_output_extends_another_is_not_read_back(self):
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/prg.o"),
            [_id("/a.o")],
        )
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/prg"),
            [_id("/p/prg.o")],
        )
        self.assertEqual(["/p/prg.o"], self.graph.get_recorded_build_steps()["/p/prg"])

    def test_a_summary_beside_a_record_is_not_read_as_one(self):
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/prg"),
            [_id("/a.o")],
        )
        open(
            utils.make_parents(
                self.ssaf_artifact_repo.get_path_for_artifact(
                    repository.get_ssaf_artifact_id(
                        build_graph.BuildArtifactId("/p/prg"),
                        repository.SSAFArtifactKind.SUMMARY,
                    )
                )
            ),
            "w",
        ).close()
        self.assertEqual(["/a.o"], self.graph.get_recorded_build_steps()["/p/prg"])

    def test_a_truncated_record_is_skipped_and_reported(self):
        self.graph.record_build_step(
            build_graph.BuildArtifactId("/p/good"),
            [_id("/a.o")],
        )
        broken = utils.make_parents(
            self.graph.get_buildstep_record_path(
                build_graph.BuildArtifactId("/p/broken")
            )
        )
        with open(broken, "w") as handle:
            handle.write('{"kind": "li')
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.assertEqual(["/p/good"], list(self.graph.get_recorded_build_steps()))
        self.assertIn("is not valid JSON", stream.getvalue())
        self.assertIn(broken, stream.getvalue())

    def test_reading_an_absent_directory_yields_nothing(self):
        self.assertEqual(
            {},
            configs.build_graph_for(self.tmp + "/none").get_recorded_build_steps(),
        )


class TopMostOutputs(unittest.TestCase):
    def _topmost(self, steps):
        return build_graph.BuildGraph.find_topmost_outputs(steps)

    def test_single_binary(self):
        self.assertEqual(["/p/prg"], self._topmost({"/p/prg": ["/a.o", "/b.o"]}))

    def test_anything_another_step_consumed_is_not_top_most(self):
        self.assertEqual(
            ["/p/prg"],
            self._topmost(
                {"/p/libx.a": [_id("/a.o")], "/p/prg": ["main.o", "/p/libx.a"]}
            ),
        )

    def test_intermediate_binary_is_not_top_most(self):
        self.assertEqual(
            ["/p/prg"],
            self._topmost({"part.o": [_id("/a.o"), _id("/b.o")], "/p/prg": ["part.o"]}),
        )

    def test_an_unconsumed_output_is_a_candidate_whatever_produced_it(self):
        self.assertEqual(["/p/libx.a"], self._topmost({"/p/libx.a": ["/a.o"]}))

    def test_a_library_nothing_links_is_a_candidate_beside_a_binary(self):
        self.assertEqual(
            ["/p/libx.a", "/p/prg"],
            self._topmost({"/p/libx.a": [_id("/a.o")], "/p/prg": ["main.o"]}),
        )

    def test_several_unrelated_binaries(self):
        self.assertEqual(
            ["tests", "tool"], self._topmost({"tool": [_id("/a.o")], "tests": ["b.o"]})
        )

    def test_nothing_recorded(self):
        self.assertEqual([], self._topmost({}))


if __name__ == "__main__":
    unittest.main()
