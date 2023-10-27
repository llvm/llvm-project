# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Default class for controlling debuggers."""

from itertools import chain
import os
import time

from dex.debugger.DebuggerControllers.DebuggerControllerBase import (
    DebuggerControllerBase,
)
from dex.debugger.DebuggerControllers.ControllerHelpers import (
    in_source_file,
    update_step_watches,
)
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException
from dex.utils.Timeout import Timeout


class EarlyExitCondition(object):
    def __init__(self, on_line, hit_count, expression, values):
        self.on_line = on_line
        self.hit_count = hit_count
        self.expression = expression
        self.values = values


class DefaultController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
        self.source_files = context.options.source_files
        self.watches = set()
        self.step_index = 0
        super(DefaultController, self).__init__(context, step_collection)

    def _break_point_all_lines(self):
        for s in self.context.options.source_files:
            with open(s, "r") as fp:
                num_lines = len(fp.readlines())
            for line in range(1, num_lines + 1):
                try:
                    self.debugger.add_breakpoint(s, line)
                except DebuggerException:
                    raise LoadDebuggerException(DebuggerException.msg)

    def _get_early_exit_conditions(self):
        commands = self.step_collection.commands
        early_exit_conditions = []
        if "DexFinishTest" in commands:
            finish_commands = commands["DexFinishTest"]
            for fc in finish_commands:
                condition = EarlyExitCondition(
                    on_line=fc.on_line,
                    hit_count=fc.hit_count,
                    expression=fc.expression,
                    values=fc.values,
                )
                early_exit_conditions.append(condition)
        return early_exit_conditions

    def _should_exit(self, early_exit_conditions, line_no):
        for condition in early_exit_conditions:
            if condition.on_line == line_no:
                exit_condition_hit = condition.expression is None
                if condition.expression is not None:
                    # For the purposes of consistent behaviour with the
                    # Conditional Controller, check equality in the debugger
                    # rather than in python (as the two can differ).
                    for value in condition.values:
                        expr_val = self.debugger.evaluate_expression(
                            f"({condition.expression}) == ({value})"
                        )
                        if expr_val.value == "true":
                            exit_condition_hit = True
                            break
                if exit_condition_hit:
                    if condition.hit_count <= 0:
                        return True
                    else:
                        condition.hit_count -= 1
        return False

    def _run_debugger_custom(self, cmdline):
        self.step_collection.debugger = self.debugger.debugger_info
        self._break_point_all_lines()
        self.debugger.launch(cmdline)
        for command_obj in chain.from_iterable(self.step_collection.commands.values()):
            self.watches.update(command_obj.get_watches())
        early_exit_conditions = self._get_early_exit_conditions()
        timed_out = False
        total_timeout = Timeout(self.context.options.timeout_total)
        max_steps = self.context.options.max_steps
        for _ in range(max_steps):
            breakpoint_timeout = Timeout(self.context.options.timeout_breakpoint)
            while self.debugger.is_running and not timed_out:
                # Check to see whether we've timed out while we're waiting.
                if total_timeout.timed_out():
                    self.context.logger.error(
                        "Debugger session has been "
                        f"running for {total_timeout.elapsed}s, timeout reached!"
                    )
                    timed_out = True
                if breakpoint_timeout.timed_out():
                    self.context.logger.error(
                        f"Debugger session has not "
                        f"hit a breakpoint for {breakpoint_timeout.elapsed}s, timeout "
                        "reached!"
                    )
                    timed_out = True

            if timed_out or self.debugger.is_finished:
                break

            self.step_index += 1
            step_info = self.debugger.get_step_info(self.watches, self.step_index)

            if step_info.current_frame:
                update_step_watches(
                    step_info, self.watches, self.step_collection.commands
                )
                self.step_collection.new_step(self.context, step_info)
                if self._should_exit(
                    early_exit_conditions, step_info.current_frame.loc.lineno
                ):
                    break

            if in_source_file(self.source_files, step_info):
                self.debugger.step()
            else:
                self.debugger.go()

            time.sleep(self.context.options.pause_between_steps)
        else:
            raise DebuggerException(
                "maximum number of steps reached ({})".format(max_steps)
            )
