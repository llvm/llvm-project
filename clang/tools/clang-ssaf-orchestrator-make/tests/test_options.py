# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import os
import shutil
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import options


class Resolve(unittest.TestCase):
    def _resolve(self, *extra):
        return options.resolve(["--extract-summaries=PointerFlow"] + list(extra))

    def test_a_valid_command_line_yields_the_options_as_typed(self):
        command_line_opts, problem = self._resolve("--", "make", "-j8")
        self.assertIsNone(problem)
        self.assertEqual(["make", "-j8"], command_line_opts.build_command)
        self.assertEqual("PointerFlow", command_line_opts.extract_summaries)

    def test_nothing_on_disk_is_consulted(self):
        command_line_opts, problem = self._resolve(
            "--ssaf-clang=/no/such/clang",
            "-o",
            "/no/such/output",
            "--",
            "make",
        )
        self.assertIsNone(problem)
        self.assertEqual("/no/such/clang", command_line_opts.ssaf_clang)
        self.assertEqual("/no/such/output", command_line_opts.output_dir)

    def test_the_separator_is_not_part_of_the_build_command(self):
        command_line_opts, problem = self._resolve("--", "make")
        self.assertIsNone(problem)
        self.assertEqual(["make"], command_line_opts.build_command)

    def test_no_build_command_is_a_usage_error(self):
        command_line_opts, problem = self._resolve()
        self.assertIsNone(command_line_opts)
        self.assertEqual(options.USAGE_ERROR, problem.exit_status)
        self.assertIn("no build command", problem.message)

    def test_a_separator_with_nothing_after_it_is_the_same(self):
        command_line_opts, problem = self._resolve("--")
        self.assertIsNone(command_line_opts)
        self.assertEqual(options.USAGE_ERROR, problem.exit_status)

    def test_a_transformation_needs_an_analysis(self):
        command_line_opts, problem = self._resolve("--transformation=x", "--", "make")
        self.assertIsNone(command_line_opts)
        self.assertEqual(options.USAGE_ERROR, problem.exit_status)
        self.assertIn("--analysis", problem.message)

    def test_resolving_makes_none_of_the_calls_configure_makes(self):
        # Not every filesystem call: argparse reads the locale through
        # gettext. These are the calls the configuration layer makes, and the
        # ones that would mean resolution had started doing its job.
        watched = (
            (os, "getcwd"),
            (os, "listdir"),
            (os, "access"),
            (os.path, "isdir"),
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
                "--ssaf-clang=/no/such/clang", "-o", "nowhere", "--", "make"
            )
        self.assertIsNone(problem)
        self.assertEqual("/no/such/clang", command_line_opts.ssaf_clang)
        self.assertEqual("nowhere", command_line_opts.output_dir)


if __name__ == "__main__":
    unittest.main()
