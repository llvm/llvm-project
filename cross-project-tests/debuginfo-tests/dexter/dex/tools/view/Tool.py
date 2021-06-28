# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""View tool."""

import os

import pickle
from dex.heuristic import Heuristic
from dex.heuristic.Heuristic import add_heuristic_tool_arguments
from dex.tools import ToolBase
from dex.utils.Exceptions import Error, HeuristicException
from dex.utils.ReturnCode import ReturnCode


class Tool(ToolBase):
    """Given a dextIR file, display the information in a human-readable form.
    """

    @property
    def name(self):
        return 'DExTer view'

    def add_tool_arguments(self, parser, defaults):
        add_heuristic_tool_arguments(parser)
        parser.add_argument(
            'input_path',
            metavar='dextIR-file',
            type=str,
            default=None,
            help='dexter dextIR file to view')
        parser.description = Tool.__doc__

    def handle_options(self, defaults):
        options = self.context.options

        options.input_path = os.path.abspath(options.input_path)
        if not os.path.isfile(options.input_path):
            raise Error('<d>could not find dextIR file</> <r>"{}"</>'.format(
                options.input_path))

    def go(self) -> ReturnCode:
        options = self.context.options

        with open(options.input_path, 'rb') as fp:
            steps = pickle.load(fp)

        try:
            heuristic = Heuristic(self.context, steps)
        except HeuristicException as e:
            raise Error('could not apply heuristic: {}'.format(e))

        self.context.o.auto('{}\n\n{}\n\n{}\n\n'.format(
            heuristic.summary_string, steps, heuristic.verbose_output))

        return ReturnCode.OK
