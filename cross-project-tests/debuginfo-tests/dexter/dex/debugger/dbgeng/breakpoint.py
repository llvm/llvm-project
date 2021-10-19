# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ctypes import *
from enum import *
from functools import partial

from .utils import *

class BreakpointTypes(IntEnum):
  DEBUG_BREAKPOINT_CODE =   0
  DEBUG_BREAKPOINT_DATA =   1
  DEBUG_BREAKPOINT_TIME =   2
  DEBUG_BREAKPOINT_INLINE = 3

class BreakpointFlags(IntFlag):
  DEBUG_BREAKPOINT_GO_ONLY =    0x00000001
  DEBUG_BREAKPOINT_DEFERRED =   0x00000002
  DEBUG_BREAKPOINT_ENABLED =    0x00000004
  DEBUG_BREAKPOINT_ADDER_ONLY = 0x00000008
  DEBUG_BREAKPOINT_ONE_SHOT =   0x00000010

DebugBreakpoint2IID = IID(0x1b278d20, 0x79f2, 0x426e, IID_Data4_Type(0xa3, 0xf9, 0xc1, 0xdd, 0xf3, 0x75, 0xd4, 0x8e))

class DebugBreakpoint2(Structure):
  pass

class DebugBreakpoint2Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(DebugBreakpoint2))
  idb_setoffset = wrp(c_ulonglong)
  idb_setflags = wrp(c_ulong)
  _fields_ = [
      ("QueryInterface", c_void_p),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("GetId", c_void_p),
      ("GetType", c_void_p),
      ("GetAdder", c_void_p),
      ("GetFlags", c_void_p),
      ("AddFlags", c_void_p),
      ("RemoveFlags", c_void_p),
      ("SetFlags", idb_setflags),
      ("GetOffset", c_void_p),
      ("SetOffset", idb_setoffset),
      ("GetDataParameters", c_void_p),
      ("SetDataParameters", c_void_p),
      ("GetPassCount", c_void_p),
      ("SetPassCount", c_void_p),
      ("GetCurrentPassCount", c_void_p),
      ("GetMatchThreadId", c_void_p),
      ("SetMatchThreadId", c_void_p),
      ("GetCommand", c_void_p),
      ("SetCommand", c_void_p),
      ("GetOffsetExpression", c_void_p),
      ("SetOffsetExpression", c_void_p),
      ("GetParameters", c_void_p),
      ("GetCommandWide", c_void_p),
      ("SetCommandWide", c_void_p),
      ("GetOffsetExpressionWide", c_void_p),
      ("SetOffsetExpressionWide", c_void_p)
    ]

DebugBreakpoint2._fields_ = [("lpVtbl", POINTER(DebugBreakpoint2Vtbl))]

class Breakpoint(object):
  def __init__(self, breakpoint):
    self.breakpoint = breakpoint.contents
    self.vt = self.breakpoint.lpVtbl.contents

  def SetFlags(self, flags):
    res = self.vt.SetFlags(self.breakpoint, flags)
    aborter(res, "Breakpoint SetFlags")

  def SetOffset(self, offs):
    res = self.vt.SetOffset(self.breakpoint, offs)
    aborter(res, "Breakpoint SetOffset")

  def RemoveFlags(self, flags):
    res = self.vt.RemoveFlags(self.breakpoint, flags)
    aborter(res, "Breakpoint RemoveFlags")

  def die(self):
    self.breakpoint = None
    self.vt = None
