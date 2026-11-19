# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
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
        self.assertEqual([], command_line_opts.orchestrator_args)

    def test_the_separator_before_the_build_command_is_optional(self):
        command_line_opts, problem = self._resolve("make", "-j8")
        self.assertIsNone(problem)
        self.assertEqual(["make", "-j8"], command_line_opts.build_command)

    def test_an_option_after_the_build_command_is_the_builds(self):
        command_line_opts, _ = self._resolve("make", "--verbose")
        self.assertEqual(["make", "--verbose"], command_line_opts.build_command)
        self.assertFalse(command_line_opts.verbose)

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
            "--ssaf-conflict-merge",
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
                "--ssaf-conflict-merge=/x/merge",
                "--ssaf-extra-clang-arg=-DX",
                "--ssaf-extra-clang-arg=-DY",
                "--keep-going",
                "--verbose",
            ],
            command_line_opts.orchestrator_args,
        )


if __name__ == "__main__":
    unittest.main()
