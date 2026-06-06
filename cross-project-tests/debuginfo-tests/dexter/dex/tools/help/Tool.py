# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Help tool."""

import textwrap

from dex.tools import ToolBase, get_tool_names, get_tools_directory, tool_main
from dex.utils.Imports import load_module
from dex.utils.ReturnCode import ReturnCode


class Tool(ToolBase):
    """Provides help info on subtools."""

    @property
    def name(self):
        return "DExTer help"

    @property
    def _visible_tool_names(self):
        return [t for t in get_tool_names() if not t.endswith("-")]

    def add_tool_arguments(self, parser, defaults):
        parser.description = Tool.__doc__
        parser.add_argument(
            "tool", choices=self._visible_tool_names, nargs="?", help="name of subtool"
        )

    def handle_options(self, defaults):
        pass

    @property
    def _default_text(self):
        s = "\n<b>The following subtools are available:</>\n\n"
        tools_directory = get_tools_directory()
        for tool_name in sorted(self._visible_tool_names):
            internal_name = tool_name.replace("-", "_")
            tool_doc = load_module(internal_name, tools_directory).Tool.__doc__
            tool_doc = tool_doc.strip() if tool_doc else ""
            tool_doc = textwrap.fill(" ".join(tool_doc.split()), 80)
            s += "<g>{}</>\n{}\n\n".format(tool_name, tool_doc)
        return s

    def go(self) -> ReturnCode:
        if self.context.options.tool is None:
            self.context.o.auto(self._default_text)
            return ReturnCode.OK

        tool_name = self.context.options.tool.replace("-", "_")
        tools_directory = get_tools_directory()
        module = load_module(tool_name, tools_directory)
        return tool_main(self.context, module.Tool(self.context), ["--help"])
