# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from ctypes import *
from enum import *
from functools import reduce, partial

from .symgroup import SymbolGroup, IDebugSymbolGroup2
from .utils import *

class SymbolOptionFlags(IntFlag):
  SYMOPT_CASE_INSENSITIVE          = 0x00000001
  SYMOPT_UNDNAME                   = 0x00000002
  SYMOPT_DEFERRED_LOADS            = 0x00000004
  SYMOPT_NO_CPP                    = 0x00000008
  SYMOPT_LOAD_LINES                = 0x00000010
  SYMOPT_OMAP_FIND_NEAREST         = 0x00000020
  SYMOPT_LOAD_ANYTHING             = 0x00000040
  SYMOPT_IGNORE_CVREC              = 0x00000080
  SYMOPT_NO_UNQUALIFIED_LOADS      = 0x00000100
  SYMOPT_FAIL_CRITICAL_ERRORS      = 0x00000200
  SYMOPT_EXACT_SYMBOLS             = 0x00000400
  SYMOPT_ALLOW_ABSOLUTE_SYMBOLS    = 0x00000800
  SYMOPT_IGNORE_NT_SYMPATH         = 0x00001000
  SYMOPT_INCLUDE_32BIT_MODULES     = 0x00002000
  SYMOPT_PUBLICS_ONLY              = 0x00004000
  SYMOPT_NO_PUBLICS                = 0x00008000
  SYMOPT_AUTO_PUBLICS              = 0x00010000
  SYMOPT_NO_IMAGE_SEARCH           = 0x00020000
  SYMOPT_SECURE                    = 0x00040000
  SYMOPT_NO_PROMPTS                = 0x00080000
  SYMOPT_DEBUG                     = 0x80000000

class ScopeGroupFlags(IntFlag):
  DEBUG_SCOPE_GROUP_ARGUMENTS    = 0x00000001
  DEBUG_SCOPE_GROUP_LOCALS       = 0x00000002
  DEBUG_SCOPE_GROUP_ALL          = 0x00000003
  DEBUG_SCOPE_GROUP_BY_DATAMODEL = 0x00000004

class DebugModuleNames(IntEnum):
  DEBUG_MODNAME_IMAGE        = 0x00000000
  DEBUG_MODNAME_MODULE       = 0x00000001
  DEBUG_MODNAME_LOADED_IMAGE = 0x00000002
  DEBUG_MODNAME_SYMBOL_FILE  = 0x00000003
  DEBUG_MODNAME_MAPPED_IMAGE = 0x00000004

class DebugModuleFlags(IntFlag):
  DEBUG_MODULE_LOADED            = 0x00000000
  DEBUG_MODULE_UNLOADED          = 0x00000001
  DEBUG_MODULE_USER_MODE         = 0x00000002
  DEBUG_MODULE_EXE_MODULE        = 0x00000004
  DEBUG_MODULE_EXPLICIT          = 0x00000008
  DEBUG_MODULE_SECONDARY         = 0x00000010
  DEBUG_MODULE_SYNTHETIC         = 0x00000020
  DEBUG_MODULE_SYM_BAD_CHECKSUM  = 0x00010000

class DEBUG_MODULE_PARAMETERS(Structure):
  _fields_ = [
      ("Base", c_ulonglong),
      ("Size", c_ulong),
      ("TimeDateStamp", c_ulong),
      ("Checksum", c_ulong),
      ("Flags", c_ulong),
      ("SymbolType", c_ulong),
      ("ImageNameSize", c_ulong),
      ("ModuleNameSize", c_ulong),
      ("LoadedImageNameSize", c_ulong),
      ("SymbolFileNameSize", c_ulong),
      ("MappedImageNameSize", c_ulong),
      ("Reserved", c_ulonglong * 2)
    ]
PDEBUG_MODULE_PARAMETERS = POINTER(DEBUG_MODULE_PARAMETERS)

class DEBUG_MODULE_AND_ID(Structure):
  _fields_ = [
      ("ModuleBase", c_ulonglong),
      ("Id", c_ulonglong)
    ]
PDEBUG_MODULE_AND_ID = POINTER(DEBUG_MODULE_AND_ID)

class DEBUG_SYMBOL_ENTRY(Structure):
  _fields_ = [
      ("ModuleBase", c_ulonglong),
      ("Offset", c_ulonglong),
      ("Id", c_ulonglong),
      ("Arg64", c_ulonglong),
      ("Size", c_ulong),
      ("Flags", c_ulong),
      ("TypeId", c_ulong),
      ("NameSize", c_ulong),
      ("Token", c_ulong),
      ("Tag", c_ulong),
      ("Arg32", c_ulong),
      ("Reserved", c_ulong)
    ]
PDEBUG_SYMBOL_ENTRY = POINTER(DEBUG_SYMBOL_ENTRY)

# UUID for DebugSymbols5 interface.
DebugSymbols5IID = IID(0xc65fa83e, 0x1e69, 0x475e, IID_Data4_Type(0x8e, 0x0e, 0xb5, 0xd7, 0x9e, 0x9c, 0xc1, 0x7e))

class IDebugSymbols5(Structure):
  pass

class IDebugSymbols5Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(IDebugSymbols5))
  ids_getsymboloptions = wrp(c_ulong_p)
  ids_setsymboloptions = wrp(c_ulong)
  ids_getmoduleparameters = wrp(c_ulong, c_ulong64_p, c_ulong, PDEBUG_MODULE_PARAMETERS)
  ids_getmodulenamestring = wrp(c_ulong, c_ulong, c_ulonglong, c_char_p, c_ulong, c_ulong_p)
  ids_getoffsetbyname = wrp(c_char_p, c_ulong64_p)
  ids_getlinebyoffset = wrp(c_ulonglong, c_ulong_p, c_char_p, c_ulong, c_ulong_p, c_ulong64_p)
  ids_getsymbolentriesbyname = wrp(c_char_p, c_ulong, PDEBUG_MODULE_AND_ID, c_ulong, c_ulong_p)
  ids_getsymbolentrystring = wrp(PDEBUG_MODULE_AND_ID, c_ulong, c_char_p, c_ulong, c_ulong_p)
  ids_getsymbolentryinformation = wrp(PDEBUG_MODULE_AND_ID, PDEBUG_SYMBOL_ENTRY)
  ids_getcurrentscopeframeindex = wrp(c_ulong_p)
  ids_getnearnamebyoffset = wrp(c_ulonglong, c_long, c_char_p, c_ulong, c_ulong_p, c_ulong64_p)
  ids_setscopeframebyindex = wrp(c_ulong)
  ids_getscopesymbolgroup2 = wrp(c_ulong, POINTER(IDebugSymbolGroup2), POINTER(POINTER(IDebugSymbolGroup2)))
  ids_getnamebyinlinecontext = wrp(c_ulonglong, c_ulong, c_char_p, c_ulong, c_ulong_p, c_ulong64_p)
  ids_getlinebyinlinecontext = wrp(c_ulonglong, c_ulong, c_ulong_p, c_char_p, c_ulong, c_ulong_p, c_ulong64_p)
  _fields_ = [
      ("QueryInterface", c_void_p),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("GetSymbolOptions", ids_getsymboloptions),
      ("AddSymbolOptions", c_void_p),
      ("RemoveSymbolOptions", c_void_p),
      ("SetSymbolOptions", ids_setsymboloptions),
      ("GetNameByOffset", c_void_p),
      ("GetOffsetByName", ids_getoffsetbyname),
      ("GetNearNameByOffset", ids_getnearnamebyoffset),
      ("GetLineByOffset", ids_getlinebyoffset),
      ("GetOffsetByLine", c_void_p),
      ("GetNumberModules", c_void_p),
      ("GetModuleByIndex", c_void_p),
      ("GetModuleByModuleName", c_void_p),
      ("GetModuleByOffset", c_void_p),
      ("GetModuleNames", c_void_p),
      ("GetModuleParameters", ids_getmoduleparameters),
      ("GetSymbolModule", c_void_p),
      ("GetTypeName", c_void_p),
      ("GetTypeId", c_void_p),
      ("GetTypeSize", c_void_p),
      ("GetFieldOffset", c_void_p),
      ("GetSymbolTypeId", c_void_p),
      ("GetOffsetTypeId", c_void_p),
      ("ReadTypedDataVirtual", c_void_p),
      ("WriteTypedDataVirtual", c_void_p),
      ("OutputTypedDataVirtual", c_void_p),
      ("ReadTypedDataPhysical", c_void_p),
      ("WriteTypedDataPhysical", c_void_p),
      ("OutputTypedDataPhysical", c_void_p),
      ("GetScope", c_void_p),
      ("SetScope", c_void_p),
      ("ResetScope", c_void_p),
      ("GetScopeSymbolGroup", c_void_p),
      ("CreateSymbolGroup", c_void_p),
      ("StartSymbolMatch", c_void_p),
      ("GetNextSymbolMatch", c_void_p),
      ("EndSymbolMatch", c_void_p),
      ("Reload", c_void_p),
      ("GetSymbolPath", c_void_p),
      ("SetSymbolPath", c_void_p),
      ("AppendSymbolPath", c_void_p),
      ("GetImagePath", c_void_p),
      ("SetImagePath", c_void_p),
      ("AppendImagePath", c_void_p),
      ("GetSourcePath", c_void_p),
      ("GetSourcePathElement", c_void_p),
      ("SetSourcePath", c_void_p),
      ("AppendSourcePath", c_void_p),
      ("FindSourceFile", c_void_p),
      ("GetSourceFileLineOffsets", c_void_p),
      ("GetModuleVersionInformation", c_void_p),
      ("GetModuleNameString", ids_getmodulenamestring),
      ("GetConstantName", c_void_p),
      ("GetFieldName", c_void_p),
      ("GetTypeOptions", c_void_p),
      ("AddTypeOptions", c_void_p),
      ("RemoveTypeOptions", c_void_p),
      ("SetTypeOptions", c_void_p),
      ("GetNameByOffsetWide", c_void_p),
      ("GetOffsetByNameWide", c_void_p),
      ("GetNearNameByOffsetWide", c_void_p),
      ("GetLineByOffsetWide", c_void_p),
      ("GetOffsetByLineWide", c_void_p),
      ("GetModuleByModuleNameWide", c_void_p),
      ("GetSymbolModuleWide", c_void_p),
      ("GetTypeNameWide", c_void_p),
      ("GetTypeIdWide", c_void_p),
      ("GetFieldOffsetWide", c_void_p),
      ("GetSymbolTypeIdWide", c_void_p),
      ("GetScopeSymbolGroup2", ids_getscopesymbolgroup2),
      ("CreateSymbolGroup2", c_void_p),
      ("StartSymbolMatchWide", c_void_p),
      ("GetNextSymbolMatchWide", c_void_p),
      ("ReloadWide", c_void_p),
      ("GetSymbolPathWide", c_void_p),
      ("SetSymbolPathWide", c_void_p),
      ("AppendSymbolPathWide", c_void_p),
      ("GetImagePathWide", c_void_p),
      ("SetImagePathWide", c_void_p),
      ("AppendImagePathWide", c_void_p),
      ("GetSourcePathWide", c_void_p),
      ("GetSourcePathElementWide", c_void_p),
      ("SetSourcePathWide", c_void_p),
      ("AppendSourcePathWide", c_void_p),
      ("FindSourceFileWide", c_void_p),
      ("GetSourceFileLineOffsetsWide", c_void_p),
      ("GetModuleVersionInformationWide", c_void_p),
      ("GetModuleNameStringWide", c_void_p),
      ("GetConstantNameWide", c_void_p),
      ("GetFieldNameWide", c_void_p),
      ("IsManagedModule", c_void_p),
      ("GetModuleByModuleName2", c_void_p),
      ("GetModuleByModuleName2Wide", c_void_p),
      ("GetModuleByOffset2", c_void_p),
      ("AddSyntheticModule", c_void_p),
      ("AddSyntheticModuleWide", c_void_p),
      ("RemoveSyntheticModule", c_void_p),
      ("GetCurrentScopeFrameIndex", ids_getcurrentscopeframeindex),
      ("SetScopeFrameByIndex", ids_setscopeframebyindex),
      ("SetScopeFromJitDebugInfo", c_void_p),
      ("SetScopeFromStoredEvent", c_void_p),
      ("OutputSymbolByOffset", c_void_p),
      ("GetFunctionEntryByOffset", c_void_p),
      ("GetFieldTypeAndOffset", c_void_p),
      ("GetFieldTypeAndOffsetWide", c_void_p),
      ("AddSyntheticSymbol", c_void_p),
      ("AddSyntheticSymbolWide", c_void_p),
      ("RemoveSyntheticSymbol", c_void_p),
      ("GetSymbolEntriesByOffset", c_void_p),
      ("GetSymbolEntriesByName", ids_getsymbolentriesbyname),
      ("GetSymbolEntriesByNameWide", c_void_p),
      ("GetSymbolEntryByToken", c_void_p),
      ("GetSymbolEntryInformation", ids_getsymbolentryinformation),
      ("GetSymbolEntryString", ids_getsymbolentrystring),
      ("GetSymbolEntryStringWide", c_void_p),
      ("GetSymbolEntryOffsetRegions", c_void_p),
      ("GetSymbolEntryBySymbolEntry", c_void_p),
      ("GetSourceEntriesByOffset", c_void_p),
      ("GetSourceEntriesByLine", c_void_p),
      ("GetSourceEntriesByLineWide", c_void_p),
      ("GetSourceEntryString", c_void_p),
      ("GetSourceEntryStringWide", c_void_p),
      ("GetSourceEntryOffsetRegions", c_void_p),
      ("GetsourceEntryBySourceEntry", c_void_p),
      ("GetScopeEx", c_void_p),
      ("SetScopeEx", c_void_p),
      ("GetNameByInlineContext", ids_getnamebyinlinecontext),
      ("GetNameByInlineContextWide", c_void_p),
      ("GetLineByInlineContext", ids_getlinebyinlinecontext),
      ("GetLineByInlineContextWide", c_void_p),
      ("OutputSymbolByInlineContext", c_void_p),
      ("GetCurrentScopeFrameIndexEx", c_void_p),
      ("SetScopeFrameByIndexEx", c_void_p)
    ]

IDebugSymbols5._fields_ = [("lpVtbl", POINTER(IDebugSymbols5Vtbl))]

SymbolId = namedtuple("SymbolId", ["ModuleBase", "Id"])
SymbolEntry = namedtuple("SymbolEntry", ["ModuleBase", "Offset", "Id", "Arg64", "Size", "Flags", "TypeId", "NameSize", "Token", "Tag", "Arg32"])
DebugModuleParams = namedtuple("DebugModuleParams", ["Base", "Size", "TimeDateStamp", "Checksum", "Flags", "SymbolType", "ImageNameSize", "ModuleNameSize", "LoadedImageNameSize", "SymbolFileNameSize", "MappedImageNameSize"])

class SymTags(IntEnum):
  Null = 0
  Exe = 1
  SymTagFunction = 5

def make_debug_module_params(cdata):
  fieldvalues = map(lambda y: getattr(cdata, y), DebugModuleParams._fields)
  return DebugModuleParams(*fieldvalues)

class Symbols(object):
  def __init__(self, symbols):
    self.ptr = symbols
    self.symbols = symbols.contents
    self.vt = self.symbols.lpVtbl.contents
    # Keep some handy ulongs for passing into C methods.
    self.ulong = c_ulong()
    self.ulong64 = c_ulonglong()

  def GetCurrentScopeFrameIndex(self):
    res = self.vt.GetCurrentScopeFrameIndex(self.symbols, byref(self.ulong))
    aborter(res, "GetCurrentScopeFrameIndex")
    return self.ulong.value

  def SetScopeFrameByIndex(self, idx):
    res = self.vt.SetScopeFrameByIndex(self.symbols, idx)
    aborter(res, "SetScopeFrameByIndex", ignore=[E_EINVAL])
    return res != E_EINVAL

  def GetOffsetByName(self, name):
    res = self.vt.GetOffsetByName(self.symbols, name.encode("ascii"), byref(self.ulong64))
    aborter(res, "GetOffsetByName {}".format(name))
    return self.ulong64.value

  def GetNearNameByOffset(self, addr):
    ptr = create_string_buffer(256)
    pulong = c_ulong()
    disp = c_ulonglong()
    # Zero arg -> "delta" indicating how many symbols to skip
    res = self.vt.GetNearNameByOffset(self.symbols, addr, 0, ptr, 255, byref(pulong), byref(disp))
    if res == E_NOINTERFACE:
      return "{noname}"
    aborter(res, "GetNearNameByOffset")
    ptr[255] = '\0'.encode("ascii")
    return '{}+{}'.format(string_at(ptr).decode("ascii"), disp.value)

  def GetModuleByModuleName2(self, name):
    # First zero arg -> module index to search from, second zero arg ->
    # DEBUG_GETMOD_* flags, none of which we use.
    res = self.vt.GetModuleByModuleName2(self.symbols, name, 0, 0, None, byref(self.ulong64))
    aborter(res, "GetModuleByModuleName2")
    return self.ulong64.value

  def GetScopeSymbolGroup2(self):
    retptr = POINTER(IDebugSymbolGroup2)()
    res = self.vt.GetScopeSymbolGroup2(self.symbols, ScopeGroupFlags.DEBUG_SCOPE_GROUP_ALL, None, retptr)
    aborter(res, "GetScopeSymbolGroup2")
    return SymbolGroup(retptr)

  def GetSymbolEntryString(self, idx, module):
    symid = DEBUG_MODULE_AND_ID()
    symid.ModuleBase = module
    symid.Id = idx
    ptr = create_string_buffer(1024)
    # Zero arg is the string index -- symbols can have multiple names, for now
    # only support the first one.
    res = self.vt.GetSymbolEntryString(self.symbols, symid, 0, ptr, 1023, byref(self.ulong))
    aborter(res, "GetSymbolEntryString")
    return string_at(ptr).decode("ascii")

  def GetSymbolEntryInformation(self, module, theid):
    symid = DEBUG_MODULE_AND_ID()
    symentry = DEBUG_SYMBOL_ENTRY()
    symid.ModuleBase = module
    symid.Id = theid
    res = self.vt.GetSymbolEntryInformation(self.symbols, symid, symentry)
    aborter(res, "GetSymbolEntryInformation")
    # Fetch fields into SymbolEntry object
    fields = map(lambda x: getattr(symentry, x), SymbolEntry._fields)
    return SymbolEntry(*fields)

  def GetSymbolEntriesByName(self, symstr):
    # Initial query to find number of symbol entries
    res = self.vt.GetSymbolEntriesByName(self.symbols, symstr.encode("ascii"), 0, None, 0, byref(self.ulong))
    aborter(res, "GetSymbolEntriesByName")

    # Build a buffer and query for 'length' entries
    length = self.ulong.value
    symrecs = (DEBUG_MODULE_AND_ID * length)()
    # Zero arg -> flags, of which there are none defined.
    res = self.vt.GetSymbolEntriesByName(self.symbols, symstr.encode("ascii"), 0, symrecs, length, byref(self.ulong))
    aborter(res, "GetSymbolEntriesByName")

    # Extract 'length' number of SymbolIds
    length = self.ulong.value
    def extract(x):
      sym = symrecs[x]
      return SymbolId(sym.ModuleBase, sym.Id)
    return [extract(x) for x in range(length)]

  def GetSymbolPath(self):
    # Query for length of buffer to allocate
    res = self.vt.GetSymbolPath(self.symbols, None, 0, byref(self.ulong))
    aborter(res, "GetSymbolPath", ignore=[S_FALSE])

    # Fetch 'length' length symbol path string
    length = self.ulong.value
    arr = create_string_buffer(length)
    res = self.vt.GetSymbolPath(self.symbols, arr, length, byref(self.ulong))
    aborter(res, "GetSymbolPath")

    return string_at(arr).decode("ascii")

  def GetSourcePath(self):
    # Query for length of buffer to allocate
    res = self.vt.GetSourcePath(self.symbols, None, 0, byref(self.ulong))
    aborter(res, "GetSourcePath", ignore=[S_FALSE])

    # Fetch a string of len 'length'
    length = self.ulong.value
    arr = create_string_buffer(length)
    res = self.vt.GetSourcePath(self.symbols, arr, length, byref(self.ulong))
    aborter(res, "GetSourcePath")

    return string_at(arr).decode("ascii")

  def SetSourcePath(self, string):
    res = self.vt.SetSourcePath(self.symbols, string.encode("ascii"))
    aborter(res, "SetSourcePath")
    return

  def GetModuleParameters(self, base):
    self.ulong64.value = base
    params = DEBUG_MODULE_PARAMETERS()
    # Fetch one module params struct, starting at idx zero
    res = self.vt.GetModuleParameters(self.symbols, 1, byref(self.ulong64), 0, byref(params))
    aborter(res, "GetModuleParameters")
    return make_debug_module_params(params)

  def GetSymbolOptions(self):
    res = self.vt.GetSymbolOptions(self.symbols, byref(self.ulong))
    aborter(res, "GetSymbolOptions")
    return SymbolOptionFlags(self.ulong.value)

  def SetSymbolOptions(self, opts):
    assert isinstance(opts, SymbolOptionFlags)
    res = self.vt.SetSymbolOptions(self.symbols, opts.value)
    aborter(res, "SetSymbolOptions")
    return

  def GetLineByOffset(self, offs):
    # Initial query for filename buffer size
    res = self.vt.GetLineByOffset(self.symbols, offs, None, None, 0, byref(self.ulong), None)
    if res == E_FAIL:
      return None # Sometimes we just can't get line numbers, of course
    aborter(res, "GetLineByOffset", ignore=[S_FALSE])

    # Allocate filename buffer and query for line number too
    filenamelen = self.ulong.value
    text = create_string_buffer(filenamelen)
    line = c_ulong()
    res = self.vt.GetLineByOffset(self.symbols, offs, byref(line), text, filenamelen, byref(self.ulong), None)
    aborter(res, "GetLineByOffset")

    return string_at(text).decode("ascii"), line.value

  def GetModuleNameString(self, whichname, base):
    # Initial query for name string length
    res = self.vt.GetModuleNameString(self.symbols, whichname, DEBUG_ANY_ID, base, None, 0, byref(self.ulong))
    aborter(res, "GetModuleNameString", ignore=[S_FALSE])

    module_name_len = self.ulong.value
    module_name = (c_char * module_name_len)()
    res = self.vt.GetModuleNameString(self.symbols, whichname, DEBUG_ANY_ID, base, module_name, module_name_len, None)
    aborter(res, "GetModuleNameString")

    return string_at(module_name).decode("ascii")

  def GetNameByInlineContext(self, pc, ctx):
    # None args -> ignore output name size and displacement
    buf = create_string_buffer(256)
    res = self.vt.GetNameByInlineContext(self.symbols, pc, ctx, buf, 255, None, None)
    aborter(res, "GetNameByInlineContext")
    return string_at(buf).decode("ascii")

  def GetLineByInlineContext(self, pc, ctx):
    # None args -> ignore output filename size and displacement
    buf = create_string_buffer(256)
    res = self.vt.GetLineByInlineContext(self.symbols, pc, ctx, byref(self.ulong), buf, 255, None, None)
    aborter(res, "GetLineByInlineContext")
    return string_at(buf).decode("ascii"), self.ulong.value

  def get_all_symbols(self):
    main_module_name = self.get_exefile_module_name()
    idnumbers = self.GetSymbolEntriesByName("{}!*".format(main_module_name))
    lst = []
    for symid in idnumbers:
      s = self.GetSymbolEntryString(symid.Id, symid.ModuleBase)
      symentry = self.GetSymbolEntryInformation(symid.ModuleBase, symid.Id)
      lst.append((s, symentry))
    return lst

  def get_all_functions(self):
    syms = self.get_all_symbols()
    return [x for x in syms if x[1].Tag == SymTags.SymTagFunction]

  def get_all_modules(self):
    params = DEBUG_MODULE_PARAMETERS()
    idx = 0
    res = 0
    all_modules = []
    while res != E_EINVAL:
      res = self.vt.GetModuleParameters(self.symbols, 1, None, idx, byref(params))
      aborter(res, "GetModuleParameters", ignore=[E_EINVAL])
      all_modules.append(make_debug_module_params(params))
      idx += 1
    return all_modules

  def get_exefile_module(self):
    all_modules = self.get_all_modules()
    reduce_func = lambda x, y: y if y.Flags & DebugModuleFlags.DEBUG_MODULE_EXE_MODULE else x
    main_module = reduce(reduce_func, all_modules, None)
    if main_module is None:
      raise Exception("Couldn't find the exefile module")
    return main_module

  def get_module_name(self, base):
    return self.GetModuleNameString(DebugModuleNames.DEBUG_MODNAME_MODULE, base)

  def get_exefile_module_name(self):
    return self.get_module_name(self.get_exefile_module().Base)
