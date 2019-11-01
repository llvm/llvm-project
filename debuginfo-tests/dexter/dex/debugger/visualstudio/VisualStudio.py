# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interface for communicating with the Visual Studio debugger via DTE."""

import abc
import imp
import os
import sys

from dex.debugger.DebuggerBase import DebuggerBase
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import StackFrame, SourceLocation, ProgramState
from dex.utils.Exceptions import Error, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode


def _load_com_module():
    try:
        module_info = imp.find_module(
            'ComInterface',
            [os.path.join(os.path.dirname(__file__), 'windows')])
        return imp.load_module('ComInterface', *module_info)
    except ImportError as e:
        raise LoadDebuggerException(e, sys.exc_info())


class VisualStudio(DebuggerBase, metaclass=abc.ABCMeta):  # pylint: disable=abstract-method

    # Constants for results of Debugger.CurrentMode
    # (https://msdn.microsoft.com/en-us/library/envdte.debugger.currentmode.aspx)
    dbgDesignMode = 1
    dbgBreakMode = 2
    dbgRunMode = 3

    def __init__(self, *args):
        self.com_module = None
        self._debugger = None
        self._solution = None
        self._fn_step = None
        self._fn_go = None
        super(VisualStudio, self).__init__(*args)

    def _custom_init(self):
        try:
            self._debugger = self._interface.Debugger
            self._debugger.HexDisplayMode = False

            self._interface.MainWindow.Visible = (
                self.context.options.show_debugger)

            self._solution = self._interface.Solution
            self._solution.Create(self.context.working_directory.path,
                                  'DexterSolution')

            try:
                self._solution.AddFromFile(self._project_file)
            except OSError:
                raise LoadDebuggerException(
                    'could not debug the specified executable', sys.exc_info())

            self._fn_step = self._debugger.StepInto
            self._fn_go = self._debugger.Go

        except AttributeError as e:
            raise LoadDebuggerException(str(e), sys.exc_info())

    def _custom_exit(self):
        if self._interface:
            self._interface.Quit()

    @property
    def _project_file(self):
        return self.context.options.executable

    @abc.abstractproperty
    def _dte_version(self):
        pass

    @property
    def _location(self):
        bp = self._debugger.BreakpointLastHit
        return {
            'path': getattr(bp, 'File', None),
            'lineno': getattr(bp, 'FileLine', None),
            'column': getattr(bp, 'FileColumn', None)
        }

    @property
    def _mode(self):
        return self._debugger.CurrentMode

    def _load_interface(self):
        self.com_module = _load_com_module()
        return self.com_module.DTE(self._dte_version)

    @property
    def version(self):
        try:
            return self._interface.Version
        except AttributeError:
            return None

    def clear_breakpoints(self):
        for bp in self._debugger.Breakpoints:
            bp.Delete()

    def add_breakpoint(self, file_, line):
        self._debugger.Breakpoints.Add('', file_, line)

    def launch(self):
        self.step()

    def step(self):
        self._fn_step()

    def go(self) -> ReturnCode:
        self._fn_go()
        return ReturnCode.OK

    def set_current_stack_frame(self, idx: int = 0):
        thread = self._debugger.CurrentThread
        stack_frames = thread.StackFrames
        try:
            stack_frame = stack_frames[idx]
            self._debugger.CurrentStackFrame = stack_frame.raw
        except IndexError:
            raise Error('attempted to access stack frame {} out of {}'
                .format(idx, len(stack_frames)))

    def get_step_info(self):
        thread = self._debugger.CurrentThread
        stackframes = thread.StackFrames

        frames = []
        state_frames = []


        for idx, sf in enumerate(stackframes):
            frame = FrameIR(
                function=self._sanitize_function_name(sf.FunctionName),
                is_inlined=sf.FunctionName.startswith('[Inline Frame]'),
                loc=LocIR(path=None, lineno=None, column=None))

            fname = frame.function or ''  # pylint: disable=no-member
            if any(name in fname for name in self.frames_below_main):
                break


            state_frame = StackFrame(function=frame.function,
                                     is_inlined=frame.is_inlined,
                                     watches={})

            for watch in self.watches:
                state_frame.watches[watch] = self.evaluate_expression(
                    watch, idx)


            state_frames.append(state_frame)
            frames.append(frame)

        loc = LocIR(**self._location)
        if frames:
            frames[0].loc = loc
            state_frames[0].location = SourceLocation(**self._location)

        reason = StopReason.BREAKPOINT
        if loc.path is None:  # pylint: disable=no-member
            reason = StopReason.STEP

        program_state = ProgramState(frames=state_frames)

        return StepIR(
            step_index=self.step_index, frames=frames, stop_reason=reason,
            program_state=program_state)

    @property
    def is_running(self):
        return self._mode == VisualStudio.dbgRunMode

    @property
    def is_finished(self):
        return self._mode == VisualStudio.dbgDesignMode

    @property
    def frames_below_main(self):
        return [
            '[Inline Frame] invoke_main', '__scrt_common_main_seh',
            '__tmainCRTStartup', 'mainCRTStartup'
        ]

    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        self.set_current_stack_frame(frame_idx)
        result = self._debugger.GetExpression(expression)
        self.set_current_stack_frame(0)
        value = result.Value

        is_optimized_away = any(s in value for s in [
            'Variable is optimized away and not available',
            'Value is not available, possibly due to optimization',
        ])

        is_irretrievable = any(s in value for s in [
            '???',
            '<Unable to read memory>',
        ])

        # an optimized away value is still counted as being able to be
        # evaluated.
        could_evaluate = (result.IsValidValue or is_optimized_away
                          or is_irretrievable)

        return ValueIR(
            expression=expression,
            value=value,
            type_name=result.Type,
            error_string=None,
            is_optimized_away=is_optimized_away,
            could_evaluate=could_evaluate,
            is_irretrievable=is_irretrievable,
        )
