# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ctypes import *
from functools import partial

from .utils import *

# UUID For SystemObjects4 interface.
DebugSystemObjects4IID = IID(0x489468e6, 0x7d0f, 0x4af5, IID_Data4_Type(0x87, 0xab, 0x25, 0x20, 0x74, 0x54, 0xd5, 0x53))

class IDebugSystemObjects4(Structure):
  pass

class IDebugSystemObjects4Vtbl(Structure):
  wrp = partial(WINFUNCTYPE, c_long, POINTER(IDebugSystemObjects4))
  ids_getnumberprocesses = wrp(POINTER(c_ulong))
  ids_getprocessidsbyindex = wrp(c_ulong, c_ulong, c_ulong_p, c_ulong_p)
  ids_setcurrentprocessid = wrp(c_ulong)
  ids_getnumberthreads = wrp(c_ulong_p)
  ids_getthreadidsbyindex = wrp(c_ulong, c_ulong, c_ulong_p, c_ulong_p)
  ids_setcurrentthreadid = wrp(c_ulong)
  _fields_ = [
      ("QueryInterface", c_void_p),
      ("AddRef", c_void_p),
      ("Release", c_void_p),
      ("GetEventThread", c_void_p),
      ("GetEventProcess", c_void_p),
      ("GetCurrentThreadId", c_void_p),
      ("SetCurrentThreadId", ids_setcurrentthreadid),
      ("GetCurrentProcessId", c_void_p),
      ("SetCurrentProcessId", ids_setcurrentprocessid),
      ("GetNumberThreads", ids_getnumberthreads),
      ("GetTotalNumberThreads", c_void_p),
      ("GetThreadIdsByIndex", ids_getthreadidsbyindex),
      ("GetThreadIdByProcessor", c_void_p),
      ("GetCurrentThreadDataOffset", c_void_p),
      ("GetThreadIdByDataOffset", c_void_p),
      ("GetCurrentThreadTeb", c_void_p),
      ("GetThreadIdByTeb", c_void_p),
      ("GetCurrentThreadSystemId", c_void_p),
      ("GetThreadIdBySystemId", c_void_p),
      ("GetCurrentThreadHandle", c_void_p),
      ("GetThreadIdByHandle", c_void_p),
      ("GetNumberProcesses", ids_getnumberprocesses),
      ("GetProcessIdsByIndex", ids_getprocessidsbyindex),
      ("GetCurrentProcessDataOffset", c_void_p),
      ("GetProcessIdByDataOffset", c_void_p),
      ("GetCurrentProcessPeb", c_void_p),
      ("GetProcessIdByPeb", c_void_p),
      ("GetCurrentProcessSystemId", c_void_p),
      ("GetProcessIdBySystemId", c_void_p),
      ("GetCurrentProcessHandle", c_void_p),
      ("GetProcessIdByHandle", c_void_p),
      ("GetCurrentProcessExecutableName", c_void_p),
      ("GetCurrentProcessUpTime", c_void_p),
      ("GetImplicitThreadDataOffset", c_void_p),
      ("SetImplicitThreadDataOffset", c_void_p),
      ("GetImplicitProcessDataOffset", c_void_p),
      ("SetImplicitProcessDataOffset", c_void_p),
      ("GetEventSystem", c_void_p),
      ("GetCurrentSystemId", c_void_p),
      ("SetCurrentSystemId", c_void_p),
      ("GetNumberSystems", c_void_p),
      ("GetSystemIdsByIndex", c_void_p),
      ("GetTotalNumberThreadsAndProcesses", c_void_p),
      ("GetCurrentSystemServer", c_void_p),
      ("GetSystemByServer", c_void_p),
      ("GetCurrentSystemServerName", c_void_p),
      ("GetCurrentProcessExecutableNameWide", c_void_p),
      ("GetCurrentSystemServerNameWide", c_void_p)
    ]

IDebugSystemObjects4._fields_ = [("lpVtbl", POINTER(IDebugSystemObjects4Vtbl))]

class SysObjects(object):
  def __init__(self, sysobjects):
    self.ptr = sysobjects
    self.sysobjects = sysobjects.contents
    self.vt = self.sysobjects.lpVtbl.contents
    # Keep a handy ulong for passing into C methods.
    self.ulong = c_ulong()

  def GetNumberSystems(self):
    res = self.vt.GetNumberSystems(self.sysobjects, byref(self.ulong))
    aborter(res, "GetNumberSystems")
    return self.ulong.value

  def GetNumberProcesses(self):
    res = self.vt.GetNumberProcesses(self.sysobjects, byref(self.ulong))
    aborter(res, "GetNumberProcesses")
    return self.ulong.value

  def GetNumberThreads(self):
    res = self.vt.GetNumberThreads(self.sysobjects, byref(self.ulong))
    aborter(res, "GetNumberThreads")
    return self.ulong.value

  def GetTotalNumberThreadsAndProcesses(self):
    tthreads = c_ulong()
    tprocs = c_ulong()
    pulong3 = c_ulong()
    res = self.vt.GetTotalNumberThreadsAndProcesses(self.sysobjects, byref(tthreads), byref(tprocs), byref(pulong3), byref(pulong3), byref(pulong3))
    aborter(res, "GettotalNumberThreadsAndProcesses")
    return tthreads.value, tprocs.value

  def GetCurrentProcessId(self):
    res = self.vt.GetCurrentProcessId(self.sysobjects, byref(self.ulong))
    aborter(res, "GetCurrentProcessId")
    return self.ulong.value

  def SetCurrentProcessId(self, sysid):
    res = self.vt.SetCurrentProcessId(self.sysobjects, sysid)
    aborter(res, "SetCurrentProcessId")
    return

  def GetCurrentThreadId(self):
    res = self.vt.GetCurrentThreadId(self.sysobjects, byref(self.ulong))
    aborter(res, "GetCurrentThreadId")
    return self.ulong.value

  def SetCurrentThreadId(self, sysid):
    res = self.vt.SetCurrentThreadId(self.sysobjects, sysid)
    aborter(res, "SetCurrentThreadId")
    return

  def GetProcessIdsByIndex(self):
    num_processes = self.GetNumberProcesses()
    if num_processes == 0:
      return []
    engineids = (c_ulong * num_processes)()
    pids = (c_ulong * num_processes)()
    for x in range(num_processes):
      engineids[x] = DEBUG_ANY_ID
      pids[x] = DEBUG_ANY_ID
    res = self.vt.GetProcessIdsByIndex(self.sysobjects, 0, num_processes, engineids, pids)
    aborter(res, "GetProcessIdsByIndex")
    return list(zip(engineids, pids))

  def GetThreadIdsByIndex(self):
    num_threads = self.GetNumberThreads()
    if num_threads == 0:
      return []
    engineids = (c_ulong * num_threads)()
    tids = (c_ulong * num_threads)()
    for x in range(num_threads):
      engineids[x] = DEBUG_ANY_ID
      tids[x] = DEBUG_ANY_ID
    # Zero -> start index
    res = self.vt.GetThreadIdsByIndex(self.sysobjects, 0, num_threads, engineids, tids)
    aborter(res, "GetThreadIdsByIndex")
    return list(zip(engineids, tids))

  def GetCurThreadHandle(self):
    pulong64 = c_ulonglong()
    res = self.vt.GetCurrentThreadHandle(self.sysobjects, byref(pulong64))
    aborter(res, "GetCurrentThreadHandle")
    return pulong64.value

  def set_current_thread(self, pid, tid):
    proc_sys_id = -1
    for x in self.GetProcessIdsByIndex():
      sysid, procid = x
      if procid == pid:
        proc_sys_id = sysid

    if proc_sys_id == -1:
      raise Exception("Couldn't find designated PID {}".format(pid))

    self.SetCurrentProcessId(proc_sys_id)

    thread_sys_id = -1
    for x in self.GetThreadIdsByIndex():
      sysid, threadid = x
      if threadid == tid:
        thread_sys_id = sysid

    if thread_sys_id == -1:
      raise Exception("Couldn't find designated TID {}".format(tid))

    self.SetCurrentThreadId(thread_sys_id)
    return

  def print_current_procs_threads(self):
    procs = []
    for x in self.GetProcessIdsByIndex():
      sysid, procid = x
      procs.append(procid)

    threads = []
    for x in self.GetThreadIdsByIndex():
      sysid, threadid = x
      threads.append(threadid)

    print("Current processes: {}".format(procs))
    print("Current threads: {}".format(threads))
