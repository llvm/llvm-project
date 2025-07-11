# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dex.tools.Main import Context, main, tool_main
from dex.tools.TestToolBase import TestToolBase
from dex.tools.ToolBase import ToolBase

def get_tool_names():
    """Returns a list of expected DExTer Tools"""
    return ["help", "list-debuggers", "no-tool-", "run-debugger-internal-", "test", "view"]

def get_tools():
    """Returns a dictionary of expected DExTer Tools"""
    return _the_tools


from .help import Tool as help_tool
from .list_debuggers import Tool as list_debuggers_tool
from .no_tool_ import Tool as no_tool_tool
from .run_debugger_internal_ import Tool as run_debugger_internal_tool
from .test import Tool as test_tool
from .view import Tool as view_tool

_the_tools = {
      "help" : help_tool,
      "list-debuggers" : list_debuggers_tool,
      "no_tool_" : no_tool_tool,
      "run-debugger-internal-" : run_debugger_internal_tool,
      "test" : test_tool,
      "view" : view_tool
}

