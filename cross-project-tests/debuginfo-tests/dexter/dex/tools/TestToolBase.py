# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base class for subtools that run tests."""

import abc
from datetime import datetime
import os
import sys

from dex.debugger.Debuggers import add_debugger_tool_arguments
from dex.debugger.Debuggers import handle_debugger_tool_options
from dex.heuristic.Heuristic import add_heuristic_tool_arguments
from dex.tools.ToolBase import ToolBase
from dex.utils import get_root_directory
from dex.utils.Exceptions import Error, ToolArgumentError
from dex.utils.ReturnCode import ReturnCode


def add_executable_arguments(parser):
    executable_group = parser.add_mutually_exclusive_group(required=True)
    executable_group.add_argument(
        "--binary", metavar="<file>", help="provide binary file to debug"
    )
    executable_group.add_argument(
        "--vs-solution",
        metavar="<file>",
        help="provide a path to an already existing visual studio solution.",
    )


class TestToolBase(ToolBase):
    def __init__(self, *args, **kwargs):
        super(TestToolBase, self).__init__(*args, **kwargs)

    def add_tool_arguments(self, parser, defaults):
        parser.description = self.__doc__
        add_debugger_tool_arguments(parser, self.context, defaults)
        add_executable_arguments(parser)
        add_heuristic_tool_arguments(parser)

        parser.add_argument(
            "test_path",
            type=str,
            metavar="<test-path>",
            nargs="?",
            default=os.path.abspath(os.path.join(get_root_directory(), "..", "tests")),
            help="directory containing test(s)",
        )

        parser.add_argument(
            "--results-directory",
            type=str,
            metavar="<directory>",
            default=None,
            help="directory to save results (default: none)",
        )

    def handle_options(self, defaults):
        options = self.context.options

        if options.vs_solution:
            options.vs_solution = os.path.abspath(options.vs_solution)
            if not os.path.isfile(options.vs_solution):
                raise Error(
                    '<d>could not find VS solution file</> <r>"{}"</>'.format(
                        options.vs_solution
                    )
                )
        elif options.binary:
            options.binary = os.path.abspath(options.binary)
            if not os.path.isfile(options.binary):
                raise Error(
                    '<d>could not find binary file</> <r>"{}"</>'.format(options.binary)
                )

        try:
            handle_debugger_tool_options(self.context, defaults)
        except ToolArgumentError as e:
            raise Error(e)

        options.test_path = os.path.abspath(options.test_path)
        options.test_path = os.path.normcase(options.test_path)
        if not os.path.isfile(options.test_path) and not os.path.isdir(
            options.test_path
        ):
            raise Error(
                '<d>could not find test path</> <r>"{}"</>'.format(options.test_path)
            )

        if options.results_directory:
            options.results_directory = os.path.abspath(options.results_directory)
            if not os.path.isdir(options.results_directory):
                try:
                    os.makedirs(options.results_directory, exist_ok=True)
                except OSError as e:
                    raise Error(
                        '<d>could not create directory</> <r>"{}"</> <y>({})</>'.format(
                            options.results_directory, e.strerror
                        )
                    )

    def go(self) -> ReturnCode:  # noqa
        options = self.context.options

        options.executable = os.path.join(
            self.context.working_directory.path, "tmp.exe"
        )

        # Test files contain dexter commands.
        options.test_files = [options.test_path]
        # Source files are the files that the program was built from, and are
        # used to determine whether a breakpoint is external to the program
        # (e.g. into a system header) or not.
        options.source_files = []
        if not options.test_path.endswith(".dex"):
            options.source_files = [options.test_path]
        self._run_test(self._get_test_name(options.test_path))

        return self._handle_results()

    @staticmethod
    def _is_current_directory(test_directory):
        return test_directory == "."

    def _get_test_name(self, test_path):
        """Get the test name from either the test file, or the sub directory
        path it's stored in.
        """
        # test names are distinguished by their relative path from the
        # specified test path.
        test_name = os.path.relpath(test_path, self.context.options.test_path)
        if self._is_current_directory(test_name):
            test_name = os.path.basename(test_path)
        return test_name

    @abc.abstractmethod
    def _run_test(self, test_dir):
        pass

    @abc.abstractmethod
    def _handle_results(self) -> ReturnCode:
        pass
