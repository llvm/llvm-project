# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make.shims import clang, shim as shim_module
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository
from clang_ssaf_orchestrator_make import build_graph, config
from clang_ssaf_orchestrator_make.failure_logs import SSAFFailureKind

from . import configs


class ForwardingAJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")
        self.links = os.path.join(self.out, "shims")
        os.makedirs(self.links)
        for name in config.SHIM_FOR_LINK_NAME:
            path = os.path.join(self.links, name)
            open(path, "w").close()
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        self.shim = configs.RecordingShim(["c++", "-o", "prg"], self.out)

    def _job(self, argv):
        return clang.Job(argv)

    def test_a_job_whose_tool_has_a_link_is_forwarded_with_the_marker(self):
        clang._invoke_shim_for_job(
            self.shim, self._job(["/toolchain/bin/ld", "-o", "prg", "a.o"]), self.tmp
        )
        failure, command = self.shim.commands[0]
        self.assertEqual(
            (
                SSAFFailureKind.FORWARD,
                build_graph.BuildArtifactId(os.path.join(self.tmp, "prg")),
            ),
            failure,
        )
        self.assertEqual(os.path.join(self.links, "ld"), command[0])
        self.assertEqual(["-o", "prg", "a.o"], command[1:-1])
        self.assertEqual(shim_module.SKIP_BINARY_TOOL_INVOCATION, command[-1])
        self.assertEqual([True], self.shim.relayed)

    def test_a_versioned_tool_still_finds_its_link(self):
        clang._invoke_shim_for_job(
            self.shim, self._job(["/build/bin/clang-23", "-cc1", "-o", "a.o"]), self.tmp
        )
        self.assertEqual(os.path.join(self.links, "clang"), self.shim.commands[0][1][0])

    def test_a_link_run_by_ld_lld_reaches_the_ld_shim(self):
        clang._invoke_shim_for_job(
            self.shim, self._job(["/build/bin/ld.lld", "-o", "prg", "a.o"]), self.tmp
        )
        link = self.shim.commands[0][1][0]
        self.assertEqual(os.path.join(self.links, "ld.lld"), link)
        self.assertEqual("ld", config.SHIM_FOR_LINK_NAME[os.path.basename(link)])

    def test_a_job_with_no_link_is_noted_and_skipped(self):
        with configs.recorded_notes() as notes:
            clang._invoke_shim_for_job(
                self.shim, self._job(["/usr/bin/dsymutil", "prg"]), self.tmp
            )
        self.assertEqual([], self.shim.commands)
        self.assertIn("no shim stands in for dsymutil", notes[0])

    def test_a_job_with_no_program_at_all_is_skipped(self):
        clang._invoke_shim_for_job(self.shim, self._job([]), self.tmp)
        self.assertEqual([], self.shim.commands)

    def test_a_job_with_no_output_is_logged_under_its_command(self):
        clang._invoke_shim_for_job(
            self.shim, self._job(["/usr/bin/ld", "a.o"]), self.tmp
        )
        self.assertEqual((SSAFFailureKind.FORWARD, None), self.shim.commands[0][0])


class ListingTheJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = configs.RecordingShim(
            ["c++", "-c", "-o", "build/a.o", "a.cpp"],
            os.path.join(self.tmp, "out"),
        )

    def _completed(self, stderr, returncode=0):
        return mock.Mock(stderr=stderr, stdout=b"", returncode=returncode)

    def test_a_listing_with_no_jobs_is_noted(self):
        with mock.patch(
            "subprocess.run", return_value=self._completed(b"")
        ), configs.recorded_notes() as notes:
            clang._invoke_all_jobs(self.shim, "/bin/clang", self.tmp)
        self.assertEqual([], self.shim.commands)
        self.assertEqual("binary: /bin/clang -### -c -o build/a.o a.cpp", notes[0])
        self.assertIn("no jobs for", notes[1])

    def test_a_driver_that_failed_is_logged_rather_than_noted(self):
        logged = []
        self.shim._shim.log_failure = lambda *args: logged.append(args)
        with mock.patch(
            "subprocess.run", return_value=self._completed(b"", returncode=1)
        ):
            clang._invoke_all_jobs(self.shim, "/bin/clang", self.tmp)
        command, _, failure_kind, build_artifact_id = logged[0]
        self.assertEqual(
            ["/bin/clang", "-###", "-c", "-o", "build/a.o", "a.cpp"], command
        )
        self.assertEqual(SSAFFailureKind.DRIVER_JOBS, failure_kind)
        self.assertEqual(
            build_graph.BuildArtifactId(os.path.join(self.tmp, "build", "a.o")),
            build_artifact_id,
        )

    def test_the_driver_is_asked_with_the_builds_own_arguments(self):
        with mock.patch("subprocess.run", return_value=self._completed(b"")) as ran:
            clang._invoke_all_jobs(self.shim, "/bin/clang", self.tmp)
        self.assertEqual(
            ["/bin/clang", "-###", "-c", "-o", "build/a.o", "a.cpp"],
            ran.call_args[0][0],
        )


class TheSsafArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")

    def _id(self, relative):
        return build_graph.BuildArtifactId(os.path.join(self.tmp, relative))

    def _shim(self, pass_config=None):
        return configs.RecordingShim(
            ["clang", "-cc1", "-o", "build/a.o", "a.cpp"],
            self.out,
            pass_config,
        )

    def test_extraction_names_the_unit_and_where_its_summary_goes(self):
        shim = self._shim()
        (summary,), extra = clang._summary_extraction_args(shim, self._id("build/a.o"))
        self.assertIn("--ssaf-extract-summaries=PointerFlow", extra)
        self.assertIn(
            "--ssaf-compilation-unit-id=" + self._id("build/a.o").get_path(), extra
        )
        self.assertIn("--ssaf-tu-summary-file=" + summary, extra)
        self.assertTrue(os.path.isdir(os.path.dirname(summary)))

    def test_transformation_names_the_link_unit_and_the_analysis_result(self):
        pass_config = configs.transformation_config()
        pass_config.name_link_unit(self._id("build/prg").get_path())
        shim = self._shim(pass_config)
        (edits, report), extra = clang._transformation_args(shim, self._id("build/a.o"))
        self.assertIn("--ssaf-link-unit-id=" + self._id("build/prg").get_path(), extra)
        self.assertIn("--ssaf-src-edit-file=" + edits, extra)
        self.assertTrue(
            any(a.startswith("--ssaf-global-scope-analysis-result=") for a in extra)
        )
        self.assertIn("--ssaf-transformation-report-file=" + report, extra)

    def test_the_extra_clang_arguments_come_last(self):
        shim = self._shim(configs.analysis_config(ssaf_extra_clang_args=["-DX"]))
        _, extra = clang._summary_extraction_args(shim, self._id("build/a.o"))
        self.assertEqual("-DX", extra[-1])


class RunningTheFrontend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")

    def _shim(self, argv):
        return configs.RecordingShim(argv, self.out)

    def _id(self, relative):
        return build_graph.BuildArtifactId(os.path.join(self.tmp, relative))

    def test_a_job_with_no_output_runs_nothing(self):
        shim = self._shim(["clang", "-cc1", "-E", "a.cpp"])
        with configs.recorded_notes() as notes:
            clang._run_clang_driven_tool(shim, self.tmp)
        self.assertEqual([], shim.commands)
        self.assertIn("no output", notes[0])

    def test_the_ssaf_clang_gets_the_jobs_own_arguments_then_the_options(self):
        shim = self._shim(["clang", "-cc1", "-o", "build/a.o", "a.cpp"])
        clang._run_clang_driven_tool(shim, self.tmp)
        failure, command = shim.commands[0]
        self.assertEqual((SSAFFailureKind.EXTRACT, self._id("build/a.o")), failure)
        self.assertEqual("/bin/clang", command[0])
        self.assertEqual(["-cc1", "-o", "build/a.o", "a.cpp"], command[1:5])
        self.assertTrue(command[5].startswith("--ssaf-"))

    def test_an_incomplete_artifact_is_removed_and_reported(self):
        shim = self._shim(["clang", "-cc1", "-o", "build/a.o", "a.cpp"])
        shim._shim.run = lambda *args: False
        stream = mock.MagicMock()
        with mock.patch("sys.stderr", stream):
            clang._run_clang_driven_tool(shim, self.tmp)
        records = configs.repository_for(self.out)
        summary = records.get_path_for_artifact(
            repository.get_ssaf_artifact_id(
                build_graph.BuildArtifactId(os.path.join(self.tmp, "build", "a.o")),
                repository.SSAFArtifactKind.SUMMARY,
            )
        )
        self.assertFalse(os.path.exists(summary))

    def test_a_failed_transformation_leaves_neither_edits_nor_report(self):
        shim = configs.RecordingShim(
            ["clang", "-cc1", "-o", "build/a.o", "a.cpp"],
            self.out,
            configs.transformation_config(link_unit_id="build/prg"),
        )
        records = configs.repository_for(self.out)
        artifacts = [
            records.get_path_for_artifact(
                repository.get_ssaf_artifact_id(self._id("build/a.o"), kind)
            )
            for kind in (
                repository.SSAFArtifactKind.EDITS,
                repository.SSAFArtifactKind.REPORT,
            )
        ]

        def run(*args):
            for artifact in artifacts:
                open(artifact, "w").close()
            return False

        shim._shim.run = run
        with mock.patch("sys.stderr", mock.MagicMock()):
            clang._run_clang_driven_tool(shim, self.tmp)
        self.assertEqual(
            [False, False], [os.path.exists(artifact) for artifact in artifacts]
        )

    def test_the_marker_chooses_the_frontend_half(self):
        shim = self._shim(
            [
                "clang",
                "-cc1",
                "-o",
                "build/a.o",
                "a.cpp",
                shim_module.SKIP_BINARY_TOOL_INVOCATION,
            ]
        )
        self.assertTrue(shim.skip_binary_tool_invocation)
        clang.invoke_ssaf_tools(shim, self.tmp)
        self.assertEqual(
            (SSAFFailureKind.EXTRACT, self._id("build/a.o")), shim.commands[0][0]
        )


class ShimNameForTool(unittest.TestCase):
    def test_the_directory_is_dropped(self):
        self.assertEqual("ld", clang._shim_name_for("/usr/bin/ld"))

    def test_a_version_suffix_is_dropped(self):
        self.assertEqual("clang", clang._shim_name_for("/opt/llvm/bin/clang-23"))

    def test_an_unversioned_name_is_left_alone(self):
        self.assertEqual(
            "clang",
            clang._shim_name_for(
                "/Applications/Xcode.app/Contents/Developer/Toolchains/"
                "XcodeDefault.xctoolchain/usr/bin/clang"
            ),
        )

    def test_only_a_trailing_numeric_suffix_is_dropped(self):
        self.assertEqual("clang-cl", clang._shim_name_for("/x/clang-cl"))
        self.assertEqual("ld64.lld", clang._shim_name_for("/x/ld64.lld"))

    def test_a_suffixed_link_name_still_reaches_its_link(self):
        for name in ("cc", "clang", "c++", "clang++"):
            self.assertEqual(name, clang._shim_name_for("/x/%s-14" % name))


if __name__ == "__main__":
    unittest.main()
