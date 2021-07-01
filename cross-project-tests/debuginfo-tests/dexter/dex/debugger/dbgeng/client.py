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
from . import control
from . import symbols
from . import sysobjs

class DebugAttach(IntFlag):
  DEBUG_ATTACH_DEFAULT =                      0
  DEBUG_ATTACH_NONINVASIVE =                  1
  DEBUG_ATTACH_EXISTING =                     2
  DEBUG_ATTACH_NONINVASIVE_NO_SUSPEND =       4
  DEBUG_ATTACH_INVASIVE_NO_INITIAL_BREAK =    8
  DEBUG_ATTACH_INVASIVE_RESUME_PROCESS =   0x10
  DEBUG_ATTACH_NONINVASIVE_ALLOW_PARTIAL = 0x20

# UUID for DebugClient7 interface.
DebugClient7IID = IID(0x13586be3, 0x542e, 0x481e, IID_Data4_Type(0xb1, 0xf2, 0x84, 0x97, 0xba, 0x74, 0xf9, 0xa9 ))

class DEBUG_CREATE_PROCESS_OPTIONS(Structure):
  _fields_ = [
    ("CreateFlags", c_ulong),
    ("EngCreateFlags", c_ulong),
    ("VerifierFlags", c_ulong),
    ("Reserved", c_ulong)
  ]

class IDebugClient7(Structure):
  pass

class IDebugClient7Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(IDebugClient7))
  idc_queryinterface = wrp(POINTER(IID), POINTER(c_void_p))
  idc_attachprocess = wrp(c_longlong, c_long, c_long)
  idc_detachprocesses = wrp()
  idc_terminateprocesses = wrp()
  idc_createprocessandattach2 = wrp(c_ulonglong, c_char_p, c_void_p, c_ulong, c_char_p, c_char_p, c_ulong, c_ulong)
  _fields_ = [
      ("QueryInterface", idc_queryinterface),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("AttachKernel", c_void_p),
      ("GetKernelConnectionOptions", c_void_p),
      ("SetKernelConnectionOptions", c_void_p),
      ("StartProcessServer", c_void_p),
      ("ConnectProcessServer", c_void_p),
      ("DisconnectProcessServer", c_void_p),
      ("GetRunningProcessSystemIds", c_void_p),
      ("GetRunningProcessSystemIdsByExecutableName", c_void_p),
      ("GetRunningProcessDescription", c_void_p),
      ("AttachProcess", idc_attachprocess),
      ("CreateProcess", c_void_p),
      ("CreateProcessAndAttach", c_void_p),
      ("GetProcessOptions", c_void_p),
      ("AddProcessOptions", c_void_p),
      ("RemoveProcessOptions", c_void_p),
      ("SetProcessOptions", c_void_p),
      ("OpenDumpFile", c_void_p),
      ("WriteDumpFile", c_void_p),
      ("ConnectSession", c_void_p),
      ("StartServer", c_void_p),
      ("OutputServers", c_void_p),
      ("TerminateProcesses", idc_terminateprocesses),
      ("DetachProcesses", idc_detachprocesses),
      ("EndSession", c_void_p),
      ("GetExitCode", c_void_p),
      ("DispatchCallbacks", c_void_p),
      ("ExitDispatch", c_void_p),
      ("CreateClient", c_void_p),
      ("GetInputCallbacks", c_void_p),
      ("SetInputCallbacks", c_void_p),
      ("GetOutputCallbacks", c_void_p),
      ("SetOutputCallbacks", c_void_p),
      ("GetOutputMask", c_void_p),
      ("SetOutputMask", c_void_p),
      ("GetOtherOutputMask", c_void_p),
      ("SetOtherOutputMask", c_void_p),
      ("GetOutputWidth", c_void_p),
      ("SetOutputWidth", c_void_p),
      ("GetOutputLinePrefix", c_void_p),
      ("SetOutputLinePrefix", c_void_p),
      ("GetIdentity", c_void_p),
      ("OutputIdentity", c_void_p),
      ("GetEventCallbacks", c_void_p),
      ("SetEventCallbacks", c_void_p),
      ("FlushCallbacks", c_void_p),
      ("WriteDumpFile2", c_void_p),
      ("AddDumpInformationFile", c_void_p),
      ("EndProcessServer", c_void_p),
      ("WaitForProcessServerEnd", c_void_p),
      ("IsKernelDebuggerEnabled", c_void_p),
      ("TerminateCurrentProcess", c_void_p),
      ("DetachCurrentProcess", c_void_p),
      ("AbandonCurrentProcess", c_void_p),
      ("GetRunningProcessSystemIdByExecutableNameWide", c_void_p),
      ("GetRunningProcessDescriptionWide", c_void_p),
      ("CreateProcessWide", c_void_p),
      ("CreateProcessAndAttachWide", c_void_p),
      ("OpenDumpFileWide", c_void_p),
      ("WriteDumpFileWide", c_void_p),
      ("AddDumpInformationFileWide", c_void_p),
      ("GetNumberDumpFiles", c_void_p),
      ("GetDumpFile", c_void_p),
      ("GetDumpFileWide", c_void_p),
      ("AttachKernelWide", c_void_p),
      ("GetKernelConnectionOptionsWide", c_void_p),
      ("SetKernelConnectionOptionsWide", c_void_p),
      ("StartProcessServerWide", c_void_p),
      ("ConnectProcessServerWide", c_void_p),
      ("StartServerWide", c_void_p),
      ("OutputServerWide", c_void_p),
      ("GetOutputCallbacksWide", c_void_p),
      ("SetOutputCallbacksWide", c_void_p),
      ("GetOutputLinePrefixWide", c_void_p),
      ("SetOutputLinePrefixWide", c_void_p),
      ("GetIdentityWide", c_void_p),
      ("OutputIdentityWide", c_void_p),
      ("GetEventCallbacksWide", c_void_p),
      ("SetEventCallbacksWide", c_void_p),
      ("CreateProcess2", c_void_p),
      ("CreateProcess2Wide", c_void_p),
      ("CreateProcessAndAttach2", idc_createprocessandattach2),
      ("CreateProcessAndAttach2Wide", c_void_p),
      ("PushOutputLinePrefix", c_void_p),
      ("PushOutputLinePrefixWide", c_void_p),
      ("PopOutputLinePrefix", c_void_p),
      ("GetNumberInputCallbacks", c_void_p),
      ("GetNumberOutputCallbacks", c_void_p),
      ("GetNumberEventCallbacks", c_void_p),
      ("GetQuitLockString", c_void_p),
      ("SetQuitLockString", c_void_p),
      ("GetQuitLockStringWide", c_void_p),
      ("SetQuitLockStringWide", c_void_p),
      ("SetEventContextCallbacks", c_void_p),
      ("SetClientContext", c_void_p),
    ]

IDebugClient7._fields_ = [("lpVtbl", POINTER(IDebugClient7Vtbl))]

class Client(object):
  def __init__(self):
    DbgEng = WinDLL("DbgEng")
    DbgEng.DebugCreate.argtypes = [POINTER(IID), POINTER(POINTER(IDebugClient7))]
    DbgEng.DebugCreate.restype = c_ulong

    # Call DebugCreate to create a new debug client
    ptr = POINTER(IDebugClient7)()
    res = DbgEng.DebugCreate(byref(DebugClient7IID), ptr)
    aborter(res, "DebugCreate")
    self.client = ptr.contents
    self.vt = vt = self.client.lpVtbl.contents

    def QI(iface, ptr):
      return vt.QueryInterface(self.client, byref(iface), byref(ptr))

    # Query for a control object
    ptr = c_void_p()
    res = QI(control.DebugControl7IID, ptr)
    aborter(res, "QueryInterface control")
    self.control_ptr = cast(ptr, POINTER(control.IDebugControl7))
    self.Control = control.Control(self.control_ptr)

    # Query for a SystemObjects object
    ptr = c_void_p()
    res = QI(sysobjs.DebugSystemObjects4IID, ptr)
    aborter(res, "QueryInterface sysobjects")
    self.sysobjects_ptr = cast(ptr, POINTER(sysobjs.IDebugSystemObjects4))
    self.SysObjects = sysobjs.SysObjects(self.sysobjects_ptr)

    # Query for a Symbols object
    ptr = c_void_p()
    res = QI(symbols.DebugSymbols5IID, ptr)
    aborter(res, "QueryInterface debugsymbosl5")
    self.symbols_ptr = cast(ptr, POINTER(symbols.IDebugSymbols5))
    self.Symbols = symbols.Symbols(self.symbols_ptr)

  def AttachProcess(self, pid):
    # Zero process-server id means no process-server.
    res = self.vt.AttachProcess(self.client, 0, pid, DebugAttach.DEBUG_ATTACH_DEFAULT)
    aborter(res, "AttachProcess")
    return

  def DetachProcesses(self):
    res = self.vt.DetachProcesses(self.client)
    aborter(res, "DetachProcesses")
    return

  def TerminateProcesses(self):
    res = self.vt.TerminateProcesses(self.client)
    aborter(res, "TerminateProcesses")
    return

  def CreateProcessAndAttach2(self, cmdline):
    options = DEBUG_CREATE_PROCESS_OPTIONS()
    options.CreateFlags = 0x2 # DEBUG_ONLY_THIS_PROCESS
    options.EngCreateFlags  = 0
    options.VerifierFlags = 0
    options.Reserved = 0
    attach_flags = 0
    res = self.vt.CreateProcessAndAttach2(self.client, 0, cmdline.encode("ascii"), byref(options), sizeof(options), None, None, 0, attach_flags)
    aborter(res, "CreateProcessAndAttach2")
    return
