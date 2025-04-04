# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""List debuggers tool."""

from dex.debugger.Debuggers import add_debugger_tool_base_arguments
from dex.debugger.Debuggers import handle_debugger_tool_base_options
from dex.debugger.Debuggers import Debuggers
from dex.tools import ToolBase
from dex.utils import Timer
from dex.utils.Exceptions import DebuggerException, Error
from dex.utils.ReturnCode import ReturnCode


class Tool(ToolBase):
    """List all of the potential debuggers that DExTer knows about and whether
    there is currently a valid interface available for them.
    """

    @property
    def name(self):
        return "DExTer list debuggers"

    def add_tool_arguments(self, parser, defaults):
        parser.description = Tool.__doc__
        add_debugger_tool_base_arguments(parser, defaults)

    def handle_options(self, defaults):
        handle_debugger_tool_base_options(self.context, defaults)

    def go(self) -> ReturnCode:
        with Timer("list debuggers"):
            try:
                Debuggers(self.context).list()
            except DebuggerException as e:
                raise Error(e)
        return ReturnCode.OK
