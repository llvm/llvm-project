# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from ctypes import *
from functools import partial

from .utils import *

Symbol = namedtuple("Symbol", ["num", "name", "type", "value"])

class IDebugSymbolGroup2(Structure):
  pass

class IDebugSymbolGroup2Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(IDebugSymbolGroup2))
  ids_getnumbersymbols = wrp(c_ulong_p)
  ids_getsymbolname = wrp(c_ulong, c_char_p, c_ulong, c_ulong_p)
  ids_getsymboltypename = wrp(c_ulong, c_char_p, c_ulong, c_ulong_p)
  ids_getsymbolvaluetext = wrp(c_ulong, c_char_p, c_ulong, c_ulong_p)
  _fields_ = [
      ("QueryInterface", c_void_p),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("GetNumberSymbols", ids_getnumbersymbols),
      ("AddSymbol", c_void_p),
      ("RemoveSymbolByName", c_void_p),
      ("RemoveSymbolByIndex", c_void_p),
      ("GetSymbolName", ids_getsymbolname),
      ("GetSymbolParameters", c_void_p),
      ("ExpandSymbol", c_void_p),
      ("OutputSymbols", c_void_p),
      ("WriteSymbol", c_void_p),
      ("OutputAsType", c_void_p),
      ("AddSymbolWide", c_void_p),
      ("RemoveSymbolByNameWide", c_void_p),
      ("GetSymbolNameWide", c_void_p),
      ("WritesymbolWide", c_void_p),
      ("OutputAsTypeWide", c_void_p),
      ("GetSymbolTypeName", ids_getsymboltypename),
      ("GetSymbolTypeNameWide", c_void_p),
      ("GetSymbolSize", c_void_p),
      ("GetSymbolOffset", c_void_p),
      ("GetSymbolRegister", c_void_p),
      ("GetSymbolValueText", ids_getsymbolvaluetext),
      ("GetSymbolValueTextWide", c_void_p),
      ("GetSymbolEntryInformation", c_void_p)
    ]

IDebugSymbolGroup2._fields_ = [("lpVtbl", POINTER(IDebugSymbolGroup2Vtbl))]

class SymbolGroup(object):
  def __init__(self, symgroup):
    self.symgroup = symgroup.contents
    self.vt = self.symgroup.lpVtbl.contents
    self.ulong = c_ulong()

  def GetNumberSymbols(self):
    res = self.vt.GetNumberSymbols(self.symgroup, byref(self.ulong))
    aborter(res, "GetNumberSymbols")
    return self.ulong.value

  def GetSymbolName(self, idx):
    buf = create_string_buffer(256)
    res = self.vt.GetSymbolName(self.symgroup, idx, buf, 255, byref(self.ulong))
    aborter(res, "GetSymbolName")
    thelen = self.ulong.value
    return string_at(buf).decode("ascii")

  def GetSymbolTypeName(self, idx):
    buf = create_string_buffer(256)
    res = self.vt.GetSymbolTypeName(self.symgroup, idx, buf, 255, byref(self.ulong))
    aborter(res, "GetSymbolTypeName")
    thelen = self.ulong.value
    return string_at(buf).decode("ascii")

  def GetSymbolValueText(self, idx, handleserror=False):
    buf = create_string_buffer(256)
    res = self.vt.GetSymbolValueText(self.symgroup, idx, buf, 255, byref(self.ulong))
    if res != 0 and handleserror:
      return None
    aborter(res, "GetSymbolTypeName")
    thelen = self.ulong.value
    return string_at(buf).decode("ascii")

  def get_symbol(self, idx):
    name = self.GetSymbolName(idx)
    thetype = self.GetSymbolTypeName(idx)
    value = self.GetSymbolValueText(idx)
    return Symbol(idx, name, thetype, value)

  def get_all_symbols(self):
    num_syms = self.GetNumberSymbols()
    return list(map(self.get_symbol, list(range(num_syms))))
