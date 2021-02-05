# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import os
import platform

from dex.debugger.DebuggerBase import DebuggerBase
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import ProgramState, StackFrame, SourceLocation
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode

if platform.system() == "Windows":
  # Don't load on linux; _load_interface will croak before any names are used.
  from . import setup
  from . import probe_process
  from . import breakpoint

class DbgEng(DebuggerBase):
    def __init__(self, context, *args):
        self.breakpoints = []
        self.running = False
        self.finished = False
        self.step_info = None
        super(DbgEng, self).__init__(context, *args)

    def _custom_init(self):
        try:
          res = setup.setup_everything(self.context.options.executable)
          self.client = res
          self.running = True
        except Exception as e:
          raise Exception('Failed to start debuggee: {}'.format(e))

    def _custom_exit(self):
        setup.cleanup(self.client)

    def _load_interface(self):
        arch = platform.architecture()[0]
        machine = platform.machine()
        if arch == '32bit' and machine == 'AMD64':
          # This python process is 32 bits, but is sitting on a 64 bit machine.
          # Bad things may happen, don't support it.
          raise LoadDebuggerException('Can\'t run Dexter dbgeng on 32 bit python in a 64 bit environment')

        if platform.system() != 'Windows':
          raise LoadDebuggerException('DbgEng supports Windows only')

        # Otherwise, everything was imported earlier

    @classmethod
    def get_name(cls):
        return 'dbgeng'

    @classmethod
    def get_option_name(cls):
        return 'dbgeng'

    @property
    def frames_below_main(self):
        return []

    @property
    def version(self):
        # I don't believe there's a well defined DbgEng version, outside of the
        # version of Windows being used.
        return "1"

    def clear_breakpoints(self):
        for x in self.breakpoints:
            x.RemoveFlags(breakpoint.BreakpointFlags.DEBUG_BREAKPOINT_ENABLED)
            self.client.Control.RemoveBreakpoint(x)

    def _add_breakpoint(self, file_, line):
        # Breakpoint setting/deleting is not supported by dbgeng at this moment
        # but is something that should be considered in the future.
        # TODO: this method is called in the DefaultController but has no effect.
        pass

    def _add_conditional_breakpoint(self, file_, line, condition):
        # breakpoint setting/deleting is not supported by dbgeng at this moment
        # but is something that should be considered in the future.
        raise NotImplementedError('add_conditional_breakpoint is not yet implemented by dbgeng')

    def _delete_conditional_breakpoint(self, file_, line, condition):
        # breakpoint setting/deleting is not supported by dbgeng at this moment
        # but is something that should be considered in the future.
        raise NotImplementedError('delete_conditional_breakpoint is not yet implemented by dbgeng')

    def launch(self):
        # We are, by this point, already launched.
        self.step_info = probe_process.probe_state(self.client)

    def step(self):
        res = setup.step_once(self.client)
        if not res:
          self.finished = True
        self.step_info = res

    def go(self):
        # FIXME: running freely doesn't seem to reliably stop when back in a
        # relevant source file -- this is likely to be a problem when setting
        # breakpoints. Until that's fixed, single step instead of running
        # freely. This isn't very efficient, but at least makes progress.
        self.step()

    def _get_step_info(self, watches, step_index):
        frames = self.step_info
        state_frames = []

        # For now assume the base function is the... function, ignoring
        # inlining.
        dex_frames = []
        for i, x in enumerate(frames):
          # XXX Might be able to get columns out through
          # GetSourceEntriesByOffset, not a priority now
          loc = LocIR(path=x.source_file, lineno=x.line_no, column=0)
          new_frame = FrameIR(function=x.function_name, is_inlined=False, loc=loc)
          dex_frames.append(new_frame)

          state_frame = StackFrame(function=new_frame.function,
                                   is_inlined=new_frame.is_inlined,
                                   location=SourceLocation(path=x.source_file,
                                                           lineno=x.line_no,
                                                           column=0),
                                   watches={})
          for expr in map(
              lambda watch, idx=i: self.evaluate_expression(watch, idx),
              watches):
              state_frame.watches[expr.expression] = expr
          state_frames.append(state_frame)

        return StepIR(
            step_index=step_index, frames=dex_frames,
            stop_reason=StopReason.STEP,
            program_state=ProgramState(state_frames))

    @property
    def is_running(self):
        return False # We're never free-running

    @property
    def is_finished(self):
        return self.finished

    def evaluate_expression(self, expression, frame_idx=0):
        # XXX: cdb insists on using '->' to examine fields of structures,
        # as it appears to reserve '.' for other purposes.
        fixed_expr = expression.replace('.', '->')

        orig_scope_idx = self.client.Symbols.GetCurrentScopeFrameIndex()
        self.client.Symbols.SetScopeFrameByIndex(frame_idx)

        res = self.client.Control.Evaluate(fixed_expr)
        if res is not None:
          result, typename = self.client.Control.Evaluate(fixed_expr)
          could_eval = True
        else:
          result, typename = (None, None)
          could_eval = False

        self.client.Symbols.SetScopeFrameByIndex(orig_scope_idx)

        return ValueIR(
            expression=expression,
            value=str(result),
            type_name=typename,
            error_string="",
            could_evaluate=could_eval,
            is_optimized_away=False,
            is_irretrievable=not could_eval)
