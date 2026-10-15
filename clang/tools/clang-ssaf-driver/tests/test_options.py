# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import os
import shutil
import unittest
from unittest import mock

from clang_ssaf_driver import options


class Resolve(unittest.TestCase):
    def _resolve(self, *extra):
        return options.resolve(["--transformation=cpp-bounded-buffers"] + list(extra))

    def test_a_valid_command_line_yields_the_options_as_typed(self):
        command_line_opts, problem = self._resolve("-o", "out", "--", "make", "-j8")
        self.assertIsNone(problem)
        self.assertEqual(["make", "-j8"], command_line_opts.build_command)
        self.assertEqual("cpp-bounded-buffers", command_line_opts.transformation)
        self.assertEqual("out", command_line_opts.output_dir)
        self.assertFalse(command_line_opts.apply)
        self.assertEqual([], command_line_opts.orchestrator_args)

    def test_the_separator_before_the_build_command_is_optional(self):
        command_line_opts, problem = self._resolve("make", "-j8")
        self.assertIsNone(problem)
        self.assertEqual(["make", "-j8"], command_line_opts.build_command)

    def test_an_option_after_the_build_command_is_the_builds(self):
        command_line_opts, _ = self._resolve("make", "--apply")
        self.assertEqual(["make", "--apply"], command_line_opts.build_command)
        self.assertFalse(command_line_opts.apply)

    def test_no_build_command_is_a_usage_error(self):
        _, problem = self._resolve("--")
        self.assertEqual(options.USAGE_ERROR, problem.exit_status)
        self.assertEqual("no build command given", problem.message)

    def test_the_transformation_is_required_and_must_be_known(self):
        for argv in (["--", "make"], ["--transformation=nonexistent", "--", "make"]):
            with mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    options.resolve(argv)
            self.assertEqual(options.USAGE_ERROR, raised.exception.code)

    def test_the_orchestrator_gets_each_forwarded_option_with_its_value(self):
        command_line_opts, _ = self._resolve(
            "--ssaf-clang=/x/clang",
            "--ssaf-merge-conflict-report",
            "/x/merge",
            "--ssaf-extra-clang-arg=-DX",
            "--ssaf-extra-clang-arg=-DY",
            "-k",
            "-v",
            "--",
            "make",
        )
        self.assertEqual(
            [
                "--ssaf-clang=/x/clang",
                "--ssaf-merge-conflict-report=/x/merge",
                "--ssaf-extra-clang-arg=-DX",
                "--ssaf-extra-clang-arg=-DY",
                "--keep-going",
                "--verbose",
            ],
            command_line_opts.orchestrator_args,
        )

    def test_apply_and_its_tool_are_kept_for_the_driver(self):
        command_line_opts, _ = self._resolve(
            "--apply", "--clang-apply-replacements=/x/car", "--", "make"
        )
        self.assertTrue(command_line_opts.apply)
        self.assertEqual("/x/car", command_line_opts.clang_apply_replacements)
        self.assertEqual([], command_line_opts.orchestrator_args)

    def test_resolving_makes_none_of_the_calls_the_driver_makes(self):
        watched = (
            (os, "getcwd"),
            (os, "access"),
            (os.path, "isfile"),
            (os.path, "realpath"),
            (shutil, "which"),
        )
        with contextlib.ExitStack() as stack:
            for module, name in watched:
                stack.enter_context(
                    mock.patch.object(
                        module,
                        name,
                        side_effect=AssertionError("%s called" % name),
                    )
                )
            command_line_opts, problem = self._resolve(
                "--apply", "--clang-apply-replacements=car", "--", "make"
            )
        self.assertIsNone(problem)
        self.assertEqual("car", command_line_opts.clang_apply_replacements)


if __name__ == "__main__":
    unittest.main()
