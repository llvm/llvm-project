# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from .utils import *

class Frame(object):
  def __init__(self, frame, idx, Symbols):
    # Store some base information about the frame
    self.ip = frame.InstructionOffset
    self.scope_idx = idx
    self.virtual = frame.Virtual
    self.inline_frame_context = frame.InlineFrameContext
    self.func_tbl_entry = frame.FuncTableEntry

    # Fetch the module/symbol we're in, with displacement. Useful for debugging.
    self.descr = Symbols.GetNearNameByOffset(self.ip)
    split = self.descr.split('!')[0]
    self.module = split[0]
    self.symbol = split[1]

    # Fetch symbol group for this scope.
    prevscope = Symbols.GetCurrentScopeFrameIndex()
    if Symbols.SetScopeFrameByIndex(idx):
      symgroup = Symbols.GetScopeSymbolGroup2()
      Symbols.SetScopeFrameByIndex(prevscope)
      self.symgroup = symgroup
    else:
      self.symgroup = None

    # Fetch the name according to the line-table, using inlining context.
    name = Symbols.GetNameByInlineContext(self.ip, self.inline_frame_context)
    self.function_name = name.split('!')[-1]

    try:
      tup = Symbols.GetLineByInlineContext(self.ip, self.inline_frame_context)
      self.source_file, self.line_no = tup
    except WinError as e:
      # Fall back to trying to use a non-inlining-aware line number
      # XXX - this is not inlining aware
      sym = Symbols.GetLineByOffset(self.ip)
      if sym is not None:
        self.source_file, self.line_no = sym
      else:
        self.source_file = None
        self.line_no = None
        self.basename = None

    if self.source_file is not None:
      self.basename = os.path.basename(self.source_file)
    else:
      self.basename = None



  def __str__(self):
    return '{}:{}({}) {}'.format(self.basename, self.line, self.descr, self.function_name)

def main_on_stack(Symbols, frames):
  module_name = Symbols.get_exefile_module_name()
  main_name = "{}!main".format(module_name)
  for x in frames:
    if main_name in x.descr: # Could be less hard coded...
      return True
  return False

def probe_state(Client):
  # Fetch the state of the program -- represented by the stack frames.
  frames, numframes = Client.Control.GetStackTraceEx()

  the_frames = [Frame(frames[x], x, Client.Symbols) for x in range(numframes)]
  if not main_on_stack(Client.Symbols, the_frames):
    return None

  return the_frames
