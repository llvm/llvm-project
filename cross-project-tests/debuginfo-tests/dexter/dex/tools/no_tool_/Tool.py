# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This is a special subtool that is run when no subtool is specified.
It just provides a welcome message and simple usage instructions.
"""

from dex.tools import ToolBase, get_tool_names
from dex.utils.Exceptions import Error
from dex.utils.ReturnCode import ReturnCode


# This is a special "tool" that is run when no subtool has been specified on
# the command line. Its only job is to provide useful usage info.
class Tool(ToolBase):
    """Welcome to DExTer (Debugging Experience Tester).
    Please choose a subtool from the list below.  Use 'dexter.py help' for more
    information.
    """

    @property
    def name(self):
        return 'DExTer'

    def add_tool_arguments(self, parser, defaults):
        parser.description = Tool.__doc__
        parser.add_argument(
            'subtool',
            choices=[t for t in get_tool_names() if not t.endswith('-')],
            nargs='?',
            help='name of subtool')
        parser.add_argument(
            'subtool_options',
            metavar='subtool-options',
            nargs='*',
            help='subtool specific options')

    def handle_options(self, defaults):
        if not self.context.options.subtool:
            raise Error('<d>no subtool specified</>\n\n{}\n'.format(
                self.parser.format_help()))

    def go(self) -> ReturnCode:
        # This fn is never called because not specifying a subtool raises an
        # exception.
        return ReturnCode._ERROR
