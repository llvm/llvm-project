# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import tempfile
import unittest

from clang_ssaf_orchestrator_make import build_graph, utils
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository

from . import configs


class OutputDirIsNeverLeft(unittest.TestCase):
    def test_an_artifact_anywhere_stays_inside_the_output_directory(self):
        ssaf_artifact_repo = configs.repository_for("/out")
        build_artifact_id = build_graph.BuildArtifactId("/elsewhere/a.o")
        self.assertTrue(
            ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_artifact_id, repository.SSAFArtifactKind.SUMMARY
                )
            ).startswith("/out/ssaf-artifacts/")
        )


class Paths(unittest.TestCase):
    def setUp(self):
        self.ssaf_artifact_repo = configs.repository_for("/out")
        self.graph = configs.build_graph_for("/out")

    def test_artifact_paths(self):
        self.assertEqual(
            "/out/ssaf-artifacts/p/build/a.o.summary.json",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/build/a.o"),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            ),
        )
        self.assertEqual(
            "/out/ssaf-artifacts/p/prg.summary.json",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/prg"),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            ),
        )
        self.assertEqual(
            "/out/ssaf-artifacts/p/build/a.o.edits.yaml",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/build/a.o"),
                    repository.SSAFArtifactKind.EDITS,
                )
            ),
        )
        self.assertEqual(
            "/out/ssaf-artifacts/p/build/a.o.report.sarif",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/build/a.o"),
                    repository.SSAFArtifactKind.REPORT,
                )
            ),
        )
        self.assertEqual(
            "/out/ssaf-artifacts/p/prg.edits.yaml",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/prg"),
                    repository.SSAFArtifactKind.EDITS,
                )
            ),
        )
        self.assertEqual(
            "/out/ssaf-artifacts/p/prg.conflicts.sarif",
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/prg"),
                    repository.SSAFArtifactKind.CONFLICTS,
                )
            ),
        )


class Construction(unittest.TestCase):
    def test_constructing_it_creates_no_directory(self):
        output_dir = os.path.join(tempfile.mkdtemp(), "out")
        self.addCleanup(shutil.rmtree, os.path.dirname(output_dir), True)
        paths = repository.SSAFArtifactRepository(output_dir)
        self.assertEqual(output_dir, paths.ssaf_artifact_dir)
        self.assertFalse(os.path.exists(output_dir))


class LinkUnitName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)
        self.graph = configs.build_graph_for(self.tmp)

    def _summary_path(self):
        return self.ssaf_artifact_repo.get_path_for_artifact(
            repository.get_ssaf_artifact_id(
                build_graph.BuildArtifactId("/p/prg"),
                repository.SSAFArtifactKind.SUMMARY,
            )
        )

    def _write(self, document):
        path = utils.make_parents(self._summary_path())
        with open(path, "w") as handle:
            json.dump(document, handle)

    def _name(self):
        return self.ssaf_artifact_repo.get_link_unit_name(self._summary_path())

    def test_reads_the_link_unit_name(self):
        self._write({"lu_namespace": [{"kind": "LinkUnit", "name": "prg"}]})
        self.assertEqual("prg", self._name())

    def test_a_library_names_no_link_unit(self):
        self._write({"namespace": {"kind": "StaticLibrary", "name": "libx"}})
        self.assertIsNone(self._name())

    def test_skips_other_namespace_kinds(self):
        self._write(
            {
                "lu_namespace": [
                    {"kind": "StaticLibrary", "name": "libx"},
                    {"kind": "LinkUnit", "name": "prg"},
                ]
            }
        )
        self.assertEqual("prg", self._name())

    def test_absent_namespace(self):
        self._write({})
        self.assertIsNone(self._name())

    def test_absent_summary(self):
        self.assertIsNone(self._name())


class Conflicts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)

    def _write(self, build_artifact_path, results):
        path = utils.make_parents(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId(build_artifact_path),
                    repository.SSAFArtifactKind.CONFLICTS,
                )
            )
        )
        with open(path, "w") as handle:
            json.dump({"version": "2.1.0", "runs": [{"results": results}]}, handle)
        return path

    def test_a_merge_that_dropped_nothing_reports_nothing(self):
        self._write("/p/prg", [])
        self.assertEqual([], self.ssaf_artifact_repo.get_conflicts())

    def test_every_merge_that_dropped_an_edit_is_reported(self):
        library = self._write("/p/libx.a", [{"ruleId": "conflict"}])
        program = self._write("/p/prg", [{"ruleId": "conflict"}])
        self._write("/p/other", [])
        self.assertEqual(
            sorted([library, program]),
            self.ssaf_artifact_repo.get_conflicts(),
        )


class Counts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ssaf_artifact_repo = configs.repository_for(self.tmp)
        self.graph = configs.build_graph_for(self.tmp)

    def _write(self, path):
        open(utils.make_parents(path), "w").close()

    def test_an_output_directory_that_does_not_exist_counts_nothing(self):
        absent = configs.repository_for(os.path.join(self.tmp, "none"))
        self.assertEqual(0, absent.count_summaries())

    def test_each_kind_is_counted_apart_at_any_depth(self):
        self._write(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/a.o"),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            )
        )
        self._write(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/deep/b.o"),
                    repository.SSAFArtifactKind.SUMMARY,
                )
            )
        )
        self._write(
            self.ssaf_artifact_repo.get_path_for_artifact(
                repository.get_ssaf_artifact_id(
                    build_graph.BuildArtifactId("/p/a.o"),
                    repository.SSAFArtifactKind.EDITS,
                )
            )
        )
        self._write(
            self.graph.get_buildstep_record_path(build_graph.BuildArtifactId("/p/prg"))
        )
        self.assertEqual(2, self.ssaf_artifact_repo.count_summaries())


if __name__ == "__main__":
    unittest.main()
