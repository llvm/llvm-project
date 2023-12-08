# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This is an internal subtool used to sandbox the communication with a
debugger into a separate process so that any crashes inside the debugger will
not bring down the entire DExTer tool.
"""

import pickle

from dex.debugger import Debuggers
from dex.tools import ToolBase
from dex.utils import Timer
from dex.utils.Exceptions import DebuggerException, Error
from dex.utils.ReturnCode import ReturnCode


class Tool(ToolBase):
    def __init__(self, *args, **kwargs):
        self.controller_path = None
        self.debugger_controller = None
        self.options = None
        super(Tool, self).__init__(*args, **kwargs)

    @property
    def name(self):
        return "DExTer run debugger internal"

    def add_tool_arguments(self, parser, defaults):
        parser.add_argument(
            "controller_path", type=str, help="pickled debugger controller file"
        )

    def handle_options(self, defaults):
        with open(self.context.options.controller_path, "rb") as fp:
            self.debugger_controller = pickle.load(fp)
        self.controller_path = self.context.options.controller_path
        self.context = self.debugger_controller.context
        self.options = self.context.options
        Timer.display = self.options.time_report

    def raise_debugger_error(self, action, debugger):
        msg = "<d>could not {} {}</> ({})\n".format(
            action, debugger.name, debugger.loading_error
        )
        if self.options.verbose:
            msg = "{}\n    {}".format(msg, "    ".join(debugger.loading_error_trace))
        raise Error(msg)

    def go(self) -> ReturnCode:
        with Timer("loading debugger"):
            debugger = Debuggers(self.context).load(self.options.debugger)

        with Timer("running debugger"):
            if not debugger.is_available:
                self.raise_debugger_error("load", debugger)

            self.debugger_controller.run_debugger(debugger)

            if debugger.loading_error:
                self.raise_debugger_error("run", debugger)

        with open(self.controller_path, "wb") as fp:
            pickle.dump(self.debugger_controller, fp)
        return ReturnCode.OK
