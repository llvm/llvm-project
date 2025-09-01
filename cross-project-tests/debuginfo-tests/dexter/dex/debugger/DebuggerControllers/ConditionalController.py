# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Conditional Controller Class for DExTer.-"""


import os
import time
from collections import defaultdict
from itertools import chain

from dex.debugger.DebuggerControllers.ControllerHelpers import (
    in_source_file,
    update_step_watches,
)
from dex.debugger.DebuggerControllers.DebuggerControllerBase import (
    DebuggerControllerBase,
)
from dex.debugger.DebuggerBase import DebuggerBase
from dex.utils.Exceptions import DebuggerException
from dex.utils.Timeout import Timeout
from dex.dextIR import LocIR

class BreakpointRange:
    """A range of breakpoints and a set of conditions.

    The leading breakpoint (on line `range_from`) is always active.

    When the leading breakpoint is hit the trailing range should be activated
    when `expression` evaluates to any value in `values`. If there are no
    conditions (`expression` is None) then the trailing breakpoint range should
    always be activated upon hitting the leading breakpoint.

    Args:
       expression: None for no conditions, or a str expression to compare
       against `values`.

       hit_count: None for no limit, or int to set the number of times the
                  leading breakpoint is triggered before it is removed.
    """

    def __init__(
        self,
        expression: str,
        path: str,
        range_from: int,
        range_to: int,
        values: list,
        hit_count: int,
        finish_on_remove: bool,
        is_continue: bool = False,
        function: str = None,
        addr: str = None,
    ):
        self.expression = expression
        self.path = path
        self.range_from = range_from
        self.range_to = range_to
        self.conditional_values = values
        self.max_hit_count = hit_count
        self.current_hit_count = 0
        self.finish_on_remove = finish_on_remove
        self.is_continue = is_continue
        self.function = function
        self.addr = addr

    def limit_steps(
        expression: str,
        path: str,
        range_from: int,
        range_to: int,
        values: list,
        hit_count: int,
    ):
        return BreakpointRange(
            expression,
            path,
            range_from,
            range_to,
            values,
            hit_count,
            False,
        )

    def finish_test(
        expression: str, path: str, on_line: int, values: list, hit_count: int
    ):
        return BreakpointRange(
            expression,
            path,
            on_line,
            on_line,
            values,
            hit_count,
            True,
        )

    def continue_from_to(
        expression: str,
        path: str,
        from_line: int,
        to_line: int,
        values: list,
        hit_count: int,
    ):
        return BreakpointRange(
            expression,
            path,
            from_line,
            to_line,
            values,
            hit_count,
            finish_on_remove=False,
            is_continue=True,
        )

    def step_function(function: str, path: str, hit_count: int):
        return BreakpointRange(
            None,
            path,
            None,
            None,
            None,
            hit_count,
            finish_on_remove=False,
            is_continue=False,
            function=function,
        )

    def has_conditions(self):
        return self.expression is not None

    def get_conditional_expression_list(self):
        conditional_list = []
        for value in self.conditional_values:
            # (<expression>) == (<value>)
            conditional_expression = "({}) == ({})".format(self.expression, value)
            conditional_list.append(conditional_expression)
        return conditional_list

    def add_hit(self):
        self.current_hit_count += 1

    def should_be_removed(self):
        if self.max_hit_count is None:
            return False
        return self.current_hit_count >= self.max_hit_count


class ConditionalController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
        self._bp_ranges = None
        self._watches = set()
        self._step_index = 0
        self._pause_between_steps = context.options.pause_between_steps
        self._max_steps = context.options.max_steps
        # Map {id: BreakpointRange}
        self._leading_bp_handles = {}
        super(ConditionalController, self).__init__(context, step_collection)
        self._build_bp_ranges()

    def _build_bp_ranges(self):
        commands = self.step_collection.commands
        self._bp_ranges = []

        cond_controller_cmds = ["DexLimitSteps", "DexStepFunction", "DexContinue"]
        if not any(c in commands for c in cond_controller_cmds):
            raise DebuggerException(
                f"No conditional commands {cond_controller_cmds}, cannot conditionally step."
            )

        if "DexLimitSteps" in commands:
            for c in commands["DexLimitSteps"]:
                bpr = BreakpointRange.limit_steps(
                    c.expression,
                    c.path,
                    c.from_line,
                    c.to_line,
                    c.values,
                    c.hit_count,
                )
                self._bp_ranges.append(bpr)
        if "DexFinishTest" in commands:
            for c in commands["DexFinishTest"]:
                bpr = BreakpointRange.finish_test(
                    c.expression, c.path, c.on_line, c.values, c.hit_count + 1
                )
                self._bp_ranges.append(bpr)
        if "DexContinue" in commands:
            for c in commands["DexContinue"]:
                bpr = BreakpointRange.continue_from_to(
                    c.expression, c.path, c.from_line, c.to_line, c.values, c.hit_count
                )
                self._bp_ranges.append(bpr)
        if "DexStepFunction" in commands:
            for c in commands["DexStepFunction"]:
                bpr = BreakpointRange.step_function(
                    c.get_function(), c.path, c.hit_count
                )
                self._bp_ranges.append(bpr)

    def _set_leading_bps(self):
        # Set a leading breakpoint for each BreakpointRange, building a
        # map of {leading bp id: BreakpointRange}.
        for bpr in self._bp_ranges:
            if bpr.has_conditions():
                # Add a conditional breakpoint for each condition.
                for cond_expr in bpr.get_conditional_expression_list():
                    id = self.debugger.add_conditional_breakpoint(
                        bpr.path, bpr.range_from, cond_expr
                    )
                    self._leading_bp_handles[id] = bpr
            elif bpr.function is not None:
                id = self.debugger.add_function_breakpoint(bpr.function)
                self._leading_bp_handles[id] = bpr
            else:
                # Add an unconditional breakpoint.
                id = self.debugger.add_breakpoint(bpr.path, bpr.range_from)
                self._leading_bp_handles[id] = bpr

    def _run_debugger_custom(self, cmdline):
        # TODO: Add conditional and unconditional breakpoint support to dbgeng.
        if self.debugger.get_name() == "dbgeng":
            raise DebuggerException(
                "DexLimitSteps commands are not supported by dbgeng"
            )

        self.step_collection.clear_steps()
        self._set_leading_bps()

        for command_obj in chain.from_iterable(self.step_collection.commands.values()):
            self._watches.update(command_obj.get_watches())

        self.debugger.launch(cmdline)
        time.sleep(self._pause_between_steps)

        exit_desired = False
        timed_out = False
        total_timeout = Timeout(self.context.options.timeout_total)

        step_function_backtraces: list[list[str]] = []
        self.instr_bp_ids = set()

        while not self.debugger.is_finished:
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

            step_info = self.debugger.get_step_info(self._watches, self._step_index)
            backtrace = None
            if step_info.current_frame:
                self._step_index += 1
                backtrace = [f.function for f in step_info.frames]

            record_step = False
            debugger_continue = False
            bp_to_delete = []
            for bp_id in self.debugger.get_triggered_breakpoint_ids():
                try:
                    # See if this is one of our leading breakpoints.
                    bpr = self._leading_bp_handles[bp_id]
                    record_step = True
                except KeyError:
                    # This is a trailing bp. Mark it for removal.
                    bp_to_delete.append(bp_id)
                    if bp_id in self.instr_bp_ids:
                        self.instr_bp_ids.remove(bp_id)
                    else:
                        record_step = True
                    continue

                bpr.add_hit()
                if bpr.should_be_removed():
                    if bpr.finish_on_remove:
                        exit_desired = True
                    bp_to_delete.append(bp_id)
                    del self._leading_bp_handles[bp_id]

                if bpr.function is not None:
                    if step_info.frames:
                        # Add this backtrace to the stack. While the current
                        # backtrace matches the top of the stack we'll step,
                        # and while there's a backtrace in the stack that
                        # is a subset of the current backtrace we'll step-out.
                        if (
                            len(step_function_backtraces) == 0
                            or backtrace != step_function_backtraces[-1]
                        ):
                            step_function_backtraces.append(backtrace)

                            # Add an address breakpoint so we don't fall out
                            # the end of nested DexStepFunctions with a DexContinue.
                            addr = self.debugger.get_pc(frame_idx=1)
                            instr_id = self.debugger.add_instruction_breakpoint(addr)
                            # Note the breakpoint so we don't log the source location
                            # it in the trace later.
                            self.instr_bp_ids.add(instr_id)

                elif bpr.is_continue:
                    debugger_continue = True
                    if bpr.range_to is not None:
                        self.debugger.add_breakpoint(bpr.path, bpr.range_to)

                else:
                    # Add a range of trailing breakpoints covering the lines
                    # requested in the DexLimitSteps command. Ignore first line as
                    # that's covered by the leading bp we just hit and include the
                    # final line.
                    for line in range(bpr.range_from + 1, bpr.range_to + 1):
                        id = self.debugger.add_breakpoint(bpr.path, line)

            # Remove any trailing or expired leading breakpoints we just hit.
            self.debugger.delete_breakpoints(bp_to_delete)

            debugger_next = False
            debugger_out = False
            if not debugger_continue and step_info.current_frame and step_info.frames:
                while len(step_function_backtraces) > 0:
                    match_subtrace = False  # Backtrace contains a target trace.
                    match_trace = False  # Backtrace matches top of target stack.

                    # The top of the step_function_backtraces stack contains a
                    # backtrace that we want to step through. Check if the
                    # current backtrace ("backtrace") either matches that trace
                    # or otherwise contains it.
                    target_backtrace = step_function_backtraces[-1]
                    if len(backtrace) >= len(target_backtrace):
                        match_trace = len(backtrace) == len(target_backtrace)
                        # Check if backtrace contains target_backtrace, matching
                        # from the end (bottom of call stack) backwards.
                        match_subtrace = (
                            backtrace[-len(target_backtrace) :] == target_backtrace
                        )

                    if match_trace:
                        # We want to step through this function; do so and
                        # log the steps in the step trace.
                        debugger_next = True
                        record_step = True
                        break
                    elif match_subtrace:
                        # There's a function we care about buried in the
                        # current backtrace. Step-out until we get to it.
                        debugger_out = True
                        break
                    else:
                        # Drop backtraces that are not match_subtraces of the current
                        # backtrace; the functions we wanted to step through
                        # there are no longer reachable.
                        step_function_backtraces.pop()

            if record_step and step_info.current_frame:
                # Record the step.
                update_step_watches(
                    step_info, self._watches, self.step_collection.commands
                )
                self.step_collection.new_step(self.context, step_info)

            if exit_desired:
                break
            elif debugger_next:
                self.debugger.step_next()
            elif debugger_out:
                self.debugger.step_out()
            else:
                self.debugger.go()
            time.sleep(self._pause_between_steps)
