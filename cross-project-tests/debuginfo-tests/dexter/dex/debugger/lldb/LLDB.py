# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interface for communicating with the LLDB debugger via its python interface.
"""

import imp
import os
import shlex
from subprocess import CalledProcessError, check_output, STDOUT
import sys

from dex.debugger.DebuggerBase import DebuggerBase, watch_is_active
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import StackFrame, SourceLocation, ProgramState
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode


class LLDB(DebuggerBase):
    def __init__(self, context, *args):
        self.lldb_executable = context.options.lldb_executable
        self._debugger = None
        self._target = None
        self._process = None
        self._thread = None
        # Map {id (int): condition (str)} for breakpoints which have a
        # condition. See get_triggered_breakpoint_ids usage for more info.
        self._breakpoint_conditions = {}
        super(LLDB, self).__init__(context, *args)

    def _custom_init(self):
        self._debugger = self._interface.SBDebugger.Create()
        self._debugger.SetAsync(False)
        self._target = self._debugger.CreateTargetWithFileAndArch(
            self.context.options.executable, self.context.options.arch
        )
        if not self._target:
            raise LoadDebuggerException(
                'could not create target for executable "{}" with arch:{}'.format(
                    self.context.options.executable, self.context.options.arch
                )
            )

    def _custom_exit(self):
        if getattr(self, "_process", None):
            self._process.Kill()
        if getattr(self, "_debugger", None) and getattr(self, "_target", None):
            self._debugger.DeleteTarget(self._target)

    def _translate_stop_reason(self, reason):
        if reason == self._interface.eStopReasonNone:
            return None
        if reason == self._interface.eStopReasonBreakpoint:
            return StopReason.BREAKPOINT
        if reason == self._interface.eStopReasonPlanComplete:
            return StopReason.STEP
        if reason == self._interface.eStopReasonThreadExiting:
            return StopReason.PROGRAM_EXIT
        if reason == self._interface.eStopReasonException:
            return StopReason.ERROR
        return StopReason.OTHER

    def _load_interface(self):
        try:
            args = [self.lldb_executable, "-P"]
            pythonpath = check_output(args, stderr=STDOUT).rstrip().decode("utf-8")
        except CalledProcessError as e:
            raise LoadDebuggerException(str(e), sys.exc_info())
        except OSError as e:
            raise LoadDebuggerException(
                '{} ["{}"]'.format(e.strerror, self.lldb_executable), sys.exc_info()
            )

        if not os.path.isdir(pythonpath):
            raise LoadDebuggerException(
                'path "{}" does not exist [result of {}]'.format(pythonpath, args),
                sys.exc_info(),
            )

        try:
            module_info = imp.find_module("lldb", [pythonpath])
            return imp.load_module("lldb", *module_info)
        except ImportError as e:
            msg = str(e)
            if msg.endswith("not a valid Win32 application."):
                msg = "{} [Are you mixing 32-bit and 64-bit binaries?]".format(msg)
            raise LoadDebuggerException(msg, sys.exc_info())

    @classmethod
    def get_name(cls):
        return "lldb"

    @classmethod
    def get_option_name(cls):
        return "lldb"

    @property
    def version(self):
        try:
            return self._interface.SBDebugger_GetVersionString()
        except AttributeError:
            return None

    def clear_breakpoints(self):
        self._target.DeleteAllBreakpoints()

    def _add_breakpoint(self, file_, line):
        return self._add_conditional_breakpoint(file_, line, None)

    def _add_conditional_breakpoint(self, file_, line, condition):
        bp = self._target.BreakpointCreateByLocation(file_, line)
        if not bp:
            raise DebuggerException(
                "could not add breakpoint [{}:{}]".format(file_, line)
            )
        id = bp.GetID()
        if condition:
            bp.SetCondition(condition)
            assert id not in self._breakpoint_conditions
            self._breakpoint_conditions[id] = condition
        return id

    def _evaulate_breakpoint_condition(self, id):
        """Evaluate the breakpoint condition and return the result.

        Returns True if a conditional breakpoint with the specified id cannot
        be found (i.e. assume it is an unconditional breakpoint).
        """
        try:
            condition = self._breakpoint_conditions[id]
        except KeyError:
            # This must be an unconditional breakpoint.
            return True
        valueIR = self.evaluate_expression(condition)
        return valueIR.type_name == "bool" and valueIR.value == "true"

    def get_triggered_breakpoint_ids(self):
        # Breakpoints can only have been triggered if we've hit one.
        stop_reason = self._translate_stop_reason(self._thread.GetStopReason())
        if stop_reason != StopReason.BREAKPOINT:
            return []
        breakpoint_ids = set()
        # When the stop reason is eStopReasonBreakpoint, GetStopReasonDataCount
        # counts all breakpoints associated with the location that lldb has
        # stopped at, regardless of their condition. I.e. Even if we have two
        # breakpoints at the same source location that have mutually exclusive
        # conditions, both will be counted by GetStopReasonDataCount when
        # either condition is true. Check each breakpoint condition manually to
        # filter the list down to breakpoints that have caused this stop.
        #
        # Breakpoints have two data parts: Breakpoint ID, Location ID. We're
        # only interested in the Breakpoint ID so we skip every other item.
        for i in range(0, self._thread.GetStopReasonDataCount(), 2):
            id = self._thread.GetStopReasonDataAtIndex(i)
            if self._evaulate_breakpoint_condition(id):
                breakpoint_ids.add(id)
        return breakpoint_ids

    def delete_breakpoints(self, ids):
        for id in ids:
            bp = self._target.FindBreakpointByID(id)
            if not bp:
                # The ID is not valid.
                raise KeyError
            try:
                del self._breakpoint_conditions[id]
            except KeyError:
                # This must be an unconditional breakpoint.
                pass
            self._target.BreakpointDelete(id)

    def launch(self, cmdline):
        if self.context.options.target_run_args:
            cmdline += shlex.split(self.context.options.target_run_args)
        self._process = self._target.LaunchSimple(cmdline, None, os.getcwd())
        if not self._process or self._process.GetNumThreads() == 0:
            raise DebuggerException("could not launch process")
        if self._process.GetNumThreads() != 1:
            raise DebuggerException("multiple threads not supported")
        self._thread = self._process.GetThreadAtIndex(0)
        assert self._thread, (self._process, self._thread)

    def step(self):
        self._thread.StepInto()

    def go(self) -> ReturnCode:
        self._process.Continue()
        return ReturnCode.OK

    def _get_step_info(self, watches, step_index):
        frames = []
        state_frames = []

        for i in range(0, self._thread.GetNumFrames()):
            sb_frame = self._thread.GetFrameAtIndex(i)
            sb_line = sb_frame.GetLineEntry()
            sb_filespec = sb_line.GetFileSpec()

            try:
                path = os.path.join(
                    sb_filespec.GetDirectory(), sb_filespec.GetFilename()
                )
            except (AttributeError, TypeError):
                path = None

            function = self._sanitize_function_name(sb_frame.GetFunctionName())

            loc_dict = {
                "path": path,
                "lineno": sb_line.GetLine(),
                "column": sb_line.GetColumn(),
            }
            loc = LocIR(**loc_dict)
            valid_loc_for_watch = loc.path and os.path.exists(loc.path)

            frame = FrameIR(function=function, is_inlined=sb_frame.IsInlined(), loc=loc)

            if any(
                name in (frame.function or "")  # pylint: disable=no-member
                for name in self.frames_below_main
            ):
                break

            frames.append(frame)

            state_frame = StackFrame(
                function=frame.function,
                is_inlined=frame.is_inlined,
                location=SourceLocation(**loc_dict),
                watches={},
            )
            if valid_loc_for_watch:
                for expr in map(
                    # Filter out watches that are not active in the current frame,
                    # and then evaluate all the active watches.
                    lambda watch_info, idx=i: self.evaluate_expression(
                        watch_info.expression, idx
                    ),
                    filter(
                        lambda watch_info, idx=i, line_no=loc.lineno, loc_path=loc.path: watch_is_active(
                            watch_info, loc_path, idx, line_no
                        ),
                        watches,
                    ),
                ):
                    state_frame.watches[expr.expression] = expr
            state_frames.append(state_frame)

        if len(frames) == 1 and frames[0].function is None:
            frames = []
            state_frames = []

        reason = self._translate_stop_reason(self._thread.GetStopReason())

        return StepIR(
            step_index=step_index,
            frames=frames,
            stop_reason=reason,
            program_state=ProgramState(state_frames),
        )

    @property
    def is_running(self):
        # We're not running in async mode so this is always False.
        return False

    @property
    def is_finished(self):
        return not self._thread.GetFrameAtIndex(0)

    @property
    def frames_below_main(self):
        return ["__scrt_common_main_seh", "__libc_start_main", "__libc_start_call_main"]

    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        result = self._thread.GetFrameAtIndex(frame_idx).EvaluateExpression(expression)
        error_string = str(result.error)

        value = result.value
        could_evaluate = not any(
            s in error_string
            for s in [
                "Can't run the expression locally",
                "use of undeclared identifier",
                "no member named",
                "Couldn't lookup symbols",
                "reference to local variable",
                "invalid use of 'this' outside of a non-static member function",
            ]
        )

        is_optimized_away = any(
            s in error_string
            for s in [
                "value may have been optimized out",
            ]
        )

        is_irretrievable = any(
            s in error_string
            for s in [
                "couldn't get the value of variable",
                "couldn't read its memory",
                "couldn't read from memory",
                "Cannot access memory at address",
                "invalid address (fault address:",
            ]
        )

        if could_evaluate and not is_irretrievable and not is_optimized_away:
            assert error_string == "success", (error_string, expression, value)
            # assert result.value is not None, (result.value, expression)

        if error_string == "success":
            error_string = None

        # attempt to find expression as a variable, if found, take the variable
        # obj's type information as it's 'usually' more accurate.
        var_result = self._thread.GetFrameAtIndex(frame_idx).FindVariable(expression)
        if str(var_result.error) == "success":
            type_name = var_result.type.GetDisplayTypeName()
        else:
            type_name = result.type.GetDisplayTypeName()

        return ValueIR(
            expression=expression,
            value=value,
            type_name=type_name,
            error_string=error_string,
            could_evaluate=could_evaluate,
            is_optimized_away=is_optimized_away,
            is_irretrievable=is_irretrievable,
        )
