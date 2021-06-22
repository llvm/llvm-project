//===-- SBProcess.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProcess.h"
#include "SBReproducerPrivate.h"

#include <cinttypes>

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SystemRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Stream.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFile.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/API/SBMemoryRegionInfoList.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBTrace.h"
#include "lldb/API/SBUnixSignals.h"

using namespace lldb;
using namespace lldb_private;

SBProcess::SBProcess() : m_opaque_wp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBProcess);
}

// SBProcess constructor

SBProcess::SBProcess(const SBProcess &rhs) : m_opaque_wp(rhs.m_opaque_wp) {
  LLDB_RECORD_CONSTRUCTOR(SBProcess, (const lldb::SBProcess &), rhs);
}

SBProcess::SBProcess(const lldb::ProcessSP &process_sp)
    : m_opaque_wp(process_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBProcess, (const lldb::ProcessSP &), process_sp);
}

const SBProcess &SBProcess::operator=(const SBProcess &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBProcess &,
                     SBProcess, operator=,(const lldb::SBProcess &), rhs);

  if (this != &rhs)
    m_opaque_wp = rhs.m_opaque_wp;
  return LLDB_RECORD_RESULT(*this);
}

// Destructor
SBProcess::~SBProcess() = default;

const char *SBProcess::GetBroadcasterClassName() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(const char *, SBProcess,
                                    GetBroadcasterClassName);

  return Process::GetStaticBroadcasterClass().AsCString();
}

const char *SBProcess::GetPluginName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBProcess, GetPluginName);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    return process_sp->GetPluginName().GetCString();
  }
  return "<Unknown>";
}

const char *SBProcess::GetShortPluginName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBProcess, GetShortPluginName);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    return process_sp->GetPluginName().GetCString();
  }
  return "<Unknown>";
}

lldb::ProcessSP SBProcess::GetSP() const { return m_opaque_wp.lock(); }

void SBProcess::SetSP(const ProcessSP &process_sp) { m_opaque_wp = process_sp; }

void SBProcess::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBProcess, Clear);

  m_opaque_wp.reset();
}

bool SBProcess::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBProcess, IsValid);
  return this->operator bool();
}
SBProcess::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBProcess, operator bool);

  ProcessSP process_sp(m_opaque_wp.lock());
  return ((bool)process_sp && process_sp->IsValid());
}

bool SBProcess::RemoteLaunch(char const **argv, char const **envp,
                             const char *stdin_path, const char *stdout_path,
                             const char *stderr_path,
                             const char *working_directory,
                             uint32_t launch_flags, bool stop_at_entry,
                             lldb::SBError &error) {
  LLDB_RECORD_METHOD(bool, SBProcess, RemoteLaunch,
                     (const char **, const char **, const char *, const char *,
                      const char *, const char *, uint32_t, bool,
                      lldb::SBError &),
                     argv, envp, stdin_path, stdout_path, stderr_path,
                     working_directory, launch_flags, stop_at_entry, error);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    if (process_sp->GetState() == eStateConnected) {
      if (stop_at_entry)
        launch_flags |= eLaunchFlagStopAtEntry;
      ProcessLaunchInfo launch_info(FileSpec(stdin_path), FileSpec(stdout_path),
                                    FileSpec(stderr_path),
                                    FileSpec(working_directory), launch_flags);
      Module *exe_module = process_sp->GetTarget().GetExecutableModulePointer();
      if (exe_module)
        launch_info.SetExecutableFile(exe_module->GetPlatformFileSpec(), true);
      if (argv)
        launch_info.GetArguments().AppendArguments(argv);
      if (envp)
        launch_info.GetEnvironment() = Environment(envp);
      error.SetError(process_sp->Launch(launch_info));
    } else {
      error.SetErrorString("must be in eStateConnected to call RemoteLaunch");
    }
  } else {
    error.SetErrorString("unable to attach pid");
  }

  return error.Success();
}

bool SBProcess::RemoteAttachToProcessWithID(lldb::pid_t pid,
                                            lldb::SBError &error) {
  LLDB_RECORD_METHOD(bool, SBProcess, RemoteAttachToProcessWithID,
                     (lldb::pid_t, lldb::SBError &), pid, error);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    if (process_sp->GetState() == eStateConnected) {
      ProcessAttachInfo attach_info;
      attach_info.SetProcessID(pid);
      error.SetError(process_sp->Attach(attach_info));
    } else {
      error.SetErrorString(
          "must be in eStateConnected to call RemoteAttachToProcessWithID");
    }
  } else {
    error.SetErrorString("unable to attach pid");
  }

  return error.Success();
}

uint32_t SBProcess::GetNumThreads() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBProcess, GetNumThreads);

  uint32_t num_threads = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;

    const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    num_threads = process_sp->GetThreadList().GetSize(can_update);
  }

  return num_threads;
}

SBThread SBProcess::GetSelectedThread() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBThread, SBProcess,
                                   GetSelectedThread);

  SBThread sb_thread;
  ThreadSP thread_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    thread_sp = process_sp->GetThreadList().GetSelectedThread();
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

SBThread SBProcess::CreateOSPluginThread(lldb::tid_t tid,
                                         lldb::addr_t context) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBProcess, CreateOSPluginThread,
                     (lldb::tid_t, lldb::addr_t), tid, context);

  SBThread sb_thread;
  ThreadSP thread_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    thread_sp = process_sp->CreateOSPluginThread(tid, context);
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

SBTarget SBProcess::GetTarget() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBTarget, SBProcess, GetTarget);

  SBTarget sb_target;
  TargetSP target_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    target_sp = process_sp->GetTarget().shared_from_this();
    sb_target.SetSP(target_sp);
  }

  return LLDB_RECORD_RESULT(sb_target);
}

size_t SBProcess::PutSTDIN(const char *src, size_t src_len) {
  LLDB_RECORD_METHOD(size_t, SBProcess, PutSTDIN, (const char *, size_t), src,
                     src_len);

  size_t ret_val = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Status error;
    ret_val = process_sp->PutSTDIN(src, src_len, error);
  }

  return ret_val;
}

size_t SBProcess::GetSTDOUT(char *dst, size_t dst_len) const {
  LLDB_RECORD_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetSTDOUT,
                                    (char *, size_t), dst, "", dst_len);

  size_t bytes_read = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Status error;
    bytes_read = process_sp->GetSTDOUT(dst, dst_len, error);
  }

  return bytes_read;
}

size_t SBProcess::GetSTDERR(char *dst, size_t dst_len) const {
  LLDB_RECORD_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetSTDERR,
                                    (char *, size_t), dst, "", dst_len);

  size_t bytes_read = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Status error;
    bytes_read = process_sp->GetSTDERR(dst, dst_len, error);
  }

  return bytes_read;
}

size_t SBProcess::GetAsyncProfileData(char *dst, size_t dst_len) const {
  LLDB_RECORD_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetAsyncProfileData,
                                    (char *, size_t), dst, "", dst_len);

  size_t bytes_read = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Status error;
    bytes_read = process_sp->GetAsyncProfileData(dst, dst_len, error);
  }

  return bytes_read;
}

void SBProcess::ReportEventState(const SBEvent &event, SBFile out) const {
  LLDB_RECORD_METHOD_CONST(void, SBProcess, ReportEventState,
                           (const SBEvent &, SBFile), event, out);

  return ReportEventState(event, out.m_opaque_sp);
}

void SBProcess::ReportEventState(const SBEvent &event, FILE *out) const {
  LLDB_RECORD_METHOD_CONST(void, SBProcess, ReportEventState,
                           (const lldb::SBEvent &, FILE *), event, out);
  FileSP outfile = std::make_shared<NativeFile>(out, false);
  return ReportEventState(event, outfile);
}

void SBProcess::ReportEventState(const SBEvent &event, FileSP out) const {

  LLDB_RECORD_METHOD_CONST(void, SBProcess, ReportEventState,
                           (const SBEvent &, FileSP), event, out);

  if (!out || !out->IsValid())
    return;

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    StreamFile stream(out);
    const StateType event_state = SBProcess::GetStateFromEvent(event);
    stream.Printf("Process %" PRIu64 " %s\n",
        process_sp->GetID(), SBDebugger::StateAsCString(event_state));
  }
}

void SBProcess::AppendEventStateReport(const SBEvent &event,
                                       SBCommandReturnObject &result) {
  LLDB_RECORD_METHOD(void, SBProcess, AppendEventStateReport,
                     (const lldb::SBEvent &, lldb::SBCommandReturnObject &),
                     event, result);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    const StateType event_state = SBProcess::GetStateFromEvent(event);
    char message[1024];
    ::snprintf(message, sizeof(message), "Process %" PRIu64 " %s\n",
               process_sp->GetID(), SBDebugger::StateAsCString(event_state));

    result.AppendMessage(message);
  }
}

bool SBProcess::SetSelectedThread(const SBThread &thread) {
  LLDB_RECORD_METHOD(bool, SBProcess, SetSelectedThread,
                     (const lldb::SBThread &), thread);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    return process_sp->GetThreadList().SetSelectedThreadByID(
        thread.GetThreadID());
  }
  return false;
}

bool SBProcess::SetSelectedThreadByID(lldb::tid_t tid) {
  LLDB_RECORD_METHOD(bool, SBProcess, SetSelectedThreadByID, (lldb::tid_t),
                     tid);


  bool ret_val = false;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    ret_val = process_sp->GetThreadList().SetSelectedThreadByID(tid);
  }

  return ret_val;
}

bool SBProcess::SetSelectedThreadByIndexID(uint32_t index_id) {
  LLDB_RECORD_METHOD(bool, SBProcess, SetSelectedThreadByIndexID, (uint32_t),
                     index_id);

  bool ret_val = false;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    ret_val = process_sp->GetThreadList().SetSelectedThreadByIndexID(index_id);
  }


  return ret_val;
}

SBThread SBProcess::GetThreadAtIndex(size_t index) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBProcess, GetThreadAtIndex, (size_t),
                     index);

  SBThread sb_thread;
  ThreadSP thread_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    thread_sp = process_sp->GetThreadList().GetThreadAtIndex(index, can_update);
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

uint32_t SBProcess::GetNumQueues() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBProcess, GetNumQueues);

  uint32_t num_queues = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      num_queues = process_sp->GetQueueList().GetSize();
    }
  }

  return num_queues;
}

SBQueue SBProcess::GetQueueAtIndex(size_t index) {
  LLDB_RECORD_METHOD(lldb::SBQueue, SBProcess, GetQueueAtIndex, (size_t),
                     index);

  SBQueue sb_queue;
  QueueSP queue_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      queue_sp = process_sp->GetQueueList().GetQueueAtIndex(index);
      sb_queue.SetQueue(queue_sp);
    }
  }

  return LLDB_RECORD_RESULT(sb_queue);
}

uint32_t SBProcess::GetStopID(bool include_expression_stops) {
  LLDB_RECORD_METHOD(uint32_t, SBProcess, GetStopID, (bool),
                     include_expression_stops);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    if (include_expression_stops)
      return process_sp->GetStopID();
    else
      return process_sp->GetLastNaturalStopID();
  }
  return 0;
}

SBEvent SBProcess::GetStopEventForStopID(uint32_t stop_id) {
  LLDB_RECORD_METHOD(lldb::SBEvent, SBProcess, GetStopEventForStopID,
                     (uint32_t), stop_id);

  SBEvent sb_event;
  EventSP event_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    event_sp = process_sp->GetStopEventForStopID(stop_id);
    sb_event.reset(event_sp);
  }

  return LLDB_RECORD_RESULT(sb_event);
}

StateType SBProcess::GetState() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::StateType, SBProcess, GetState);

  StateType ret_val = eStateInvalid;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    ret_val = process_sp->GetState();
  }

  return ret_val;
}

int SBProcess::GetExitStatus() {
  LLDB_RECORD_METHOD_NO_ARGS(int, SBProcess, GetExitStatus);

  int exit_status = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    exit_status = process_sp->GetExitStatus();
  }

  return exit_status;
}

const char *SBProcess::GetExitDescription() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBProcess, GetExitDescription);

  const char *exit_desc = nullptr;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    exit_desc = process_sp->GetExitDescription();
  }
  return exit_desc;
}

lldb::pid_t SBProcess::GetProcessID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::pid_t, SBProcess, GetProcessID);

  lldb::pid_t ret_val = LLDB_INVALID_PROCESS_ID;
  ProcessSP process_sp(GetSP());
  if (process_sp)
    ret_val = process_sp->GetID();

  return ret_val;
}

uint32_t SBProcess::GetUniqueID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBProcess, GetUniqueID);

  uint32_t ret_val = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp)
    ret_val = process_sp->GetUniqueID();
  return ret_val;
}

ByteOrder SBProcess::GetByteOrder() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::ByteOrder, SBProcess, GetByteOrder);

  ByteOrder byteOrder = eByteOrderInvalid;
  ProcessSP process_sp(GetSP());
  if (process_sp)
    byteOrder = process_sp->GetTarget().GetArchitecture().GetByteOrder();


  return byteOrder;
}

uint32_t SBProcess::GetAddressByteSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBProcess, GetAddressByteSize);

  uint32_t size = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp)
    size = process_sp->GetTarget().GetArchitecture().GetAddressByteSize();


  return size;
}

SBError SBProcess::Continue() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBProcess, Continue);

  SBError sb_error;
  ProcessSP process_sp(GetSP());

  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());

    if (process_sp->GetTarget().GetDebugger().GetAsyncExecution())
      sb_error.ref() = process_sp->Resume();
    else
      sb_error.ref() = process_sp->ResumeSynchronous(nullptr);
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBError SBProcess::Destroy() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBProcess, Destroy);

  SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->Destroy(false));
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBError SBProcess::Stop() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBProcess, Stop);

  SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->Halt());
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBError SBProcess::Kill() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBProcess, Kill);

  SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->Destroy(true));
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBError SBProcess::Detach() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBProcess, Detach);

  // FIXME: This should come from a process default.
  bool keep_stopped = false;
  return LLDB_RECORD_RESULT(Detach(keep_stopped));
}

SBError SBProcess::Detach(bool keep_stopped) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, Detach, (bool), keep_stopped);

  SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->Detach(keep_stopped));
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBError SBProcess::Signal(int signo) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, Signal, (int), signo);

  SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->Signal(signo));
  } else
    sb_error.SetErrorString("SBProcess is invalid");

  return LLDB_RECORD_RESULT(sb_error);
}

SBUnixSignals SBProcess::GetUnixSignals() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBUnixSignals, SBProcess, GetUnixSignals);

  if (auto process_sp = GetSP())
    return LLDB_RECORD_RESULT(SBUnixSignals{process_sp});

  return LLDB_RECORD_RESULT(SBUnixSignals{});
}

void SBProcess::SendAsyncInterrupt() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBProcess, SendAsyncInterrupt);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    process_sp->SendAsyncInterrupt();
  }
}

SBThread SBProcess::GetThreadByID(tid_t tid) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBProcess, GetThreadByID, (lldb::tid_t),
                     tid);

  SBThread sb_thread;
  ThreadSP thread_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    thread_sp = process_sp->GetThreadList().FindThreadByID(tid, can_update);
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

SBThread SBProcess::GetThreadByIndexID(uint32_t index_id) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBProcess, GetThreadByIndexID, (uint32_t),
                     index_id);

  SBThread sb_thread;
  ThreadSP thread_sp;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    thread_sp =
        process_sp->GetThreadList().FindThreadByIndexID(index_id, can_update);
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

StateType SBProcess::GetStateFromEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(lldb::StateType, SBProcess, GetStateFromEvent,
                            (const lldb::SBEvent &), event);

  StateType ret_val = Process::ProcessEventData::GetStateFromEvent(event.get());

  return ret_val;
}

bool SBProcess::GetRestartedFromEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBProcess, GetRestartedFromEvent,
                            (const lldb::SBEvent &), event);

  bool ret_val = Process::ProcessEventData::GetRestartedFromEvent(event.get());

  return ret_val;
}

size_t SBProcess::GetNumRestartedReasonsFromEvent(const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(size_t, SBProcess, GetNumRestartedReasonsFromEvent,
                            (const lldb::SBEvent &), event);

  return Process::ProcessEventData::GetNumRestartedReasons(event.get());
}

const char *
SBProcess::GetRestartedReasonAtIndexFromEvent(const lldb::SBEvent &event,
                                              size_t idx) {
  LLDB_RECORD_STATIC_METHOD(const char *, SBProcess,
                            GetRestartedReasonAtIndexFromEvent,
                            (const lldb::SBEvent &, size_t), event, idx);

  return Process::ProcessEventData::GetRestartedReasonAtIndex(event.get(), idx);
}

SBProcess SBProcess::GetProcessFromEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBProcess, SBProcess, GetProcessFromEvent,
                            (const lldb::SBEvent &), event);

  ProcessSP process_sp =
      Process::ProcessEventData::GetProcessFromEvent(event.get());
  if (!process_sp) {
    // StructuredData events also know the process they come from. Try that.
    process_sp = EventDataStructuredData::GetProcessFromEvent(event.get());
  }

  return LLDB_RECORD_RESULT(SBProcess(process_sp));
}

bool SBProcess::GetInterruptedFromEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBProcess, GetInterruptedFromEvent,
                            (const lldb::SBEvent &), event);

  return Process::ProcessEventData::GetInterruptedFromEvent(event.get());
}

lldb::SBStructuredData
SBProcess::GetStructuredDataFromEvent(const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBStructuredData, SBProcess,
                            GetStructuredDataFromEvent, (const lldb::SBEvent &),
                            event);

  return LLDB_RECORD_RESULT(SBStructuredData(event.GetSP()));
}

bool SBProcess::EventIsProcessEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBProcess, EventIsProcessEvent,
                            (const lldb::SBEvent &), event);

  return (event.GetBroadcasterClass() == SBProcess::GetBroadcasterClass()) &&
         !EventIsStructuredDataEvent(event);
}

bool SBProcess::EventIsStructuredDataEvent(const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBProcess, EventIsStructuredDataEvent,
                            (const lldb::SBEvent &), event);

  EventSP event_sp = event.GetSP();
  EventData *event_data = event_sp ? event_sp->GetData() : nullptr;
  return event_data && (event_data->GetFlavor() ==
                        EventDataStructuredData::GetFlavorString());
}

SBBroadcaster SBProcess::GetBroadcaster() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBBroadcaster, SBProcess,
                                   GetBroadcaster);


  ProcessSP process_sp(GetSP());

  SBBroadcaster broadcaster(process_sp.get(), false);


  return LLDB_RECORD_RESULT(broadcaster);
}

const char *SBProcess::GetBroadcasterClass() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(const char *, SBProcess,
                                    GetBroadcasterClass);

  return Process::GetStaticBroadcasterClass().AsCString();
}

size_t SBProcess::ReadMemory(addr_t addr, void *dst, size_t dst_len,
                             SBError &sb_error) {
  LLDB_RECORD_DUMMY(size_t, SBProcess, ReadMemory,
                    (lldb::addr_t, void *, size_t, lldb::SBError &), addr, dst,
                    dst_len, sb_error);

  size_t bytes_read = 0;

  ProcessSP process_sp(GetSP());


  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      bytes_read = process_sp->ReadMemory(addr, dst, dst_len, sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }

  return bytes_read;
}

size_t SBProcess::ReadCStringFromMemory(addr_t addr, void *buf, size_t size,
                                        lldb::SBError &sb_error) {
  LLDB_RECORD_DUMMY(size_t, SBProcess, ReadCStringFromMemory,
                    (lldb::addr_t, void *, size_t, lldb::SBError &), addr, buf,
                    size, sb_error);

  size_t bytes_read = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      bytes_read = process_sp->ReadCStringFromMemory(addr, (char *)buf, size,
                                                     sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }
  return bytes_read;
}

uint64_t SBProcess::ReadUnsignedFromMemory(addr_t addr, uint32_t byte_size,
                                           lldb::SBError &sb_error) {
  LLDB_RECORD_METHOD(uint64_t, SBProcess, ReadUnsignedFromMemory,
                     (lldb::addr_t, uint32_t, lldb::SBError &), addr, byte_size,
                     sb_error);

  uint64_t value = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      value = process_sp->ReadUnsignedIntegerFromMemory(addr, byte_size, 0,
                                                        sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }
  return value;
}

lldb::addr_t SBProcess::ReadPointerFromMemory(addr_t addr,
                                              lldb::SBError &sb_error) {
  LLDB_RECORD_METHOD(lldb::addr_t, SBProcess, ReadPointerFromMemory,
                     (lldb::addr_t, lldb::SBError &), addr, sb_error);

  lldb::addr_t ptr = LLDB_INVALID_ADDRESS;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      ptr = process_sp->ReadPointerFromMemory(addr, sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }
  return ptr;
}

size_t SBProcess::WriteMemory(addr_t addr, const void *src, size_t src_len,
                              SBError &sb_error) {
  LLDB_RECORD_DUMMY(size_t, SBProcess, WriteMemory,
                    (lldb::addr_t, const void *, size_t, lldb::SBError &), addr,
                    src, src_len, sb_error);

  size_t bytes_written = 0;

  ProcessSP process_sp(GetSP());

  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      bytes_written =
          process_sp->WriteMemory(addr, src, src_len, sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  }

  return bytes_written;
}

bool SBProcess::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBProcess, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    char path[PATH_MAX];
    GetTarget().GetExecutable().GetPath(path, sizeof(path));
    Module *exe_module = process_sp->GetTarget().GetExecutableModulePointer();
    const char *exe_name = nullptr;
    if (exe_module)
      exe_name = exe_module->GetFileSpec().GetFilename().AsCString();

    strm.Printf("SBProcess: pid = %" PRIu64 ", state = %s, threads = %d%s%s",
                process_sp->GetID(), lldb_private::StateAsCString(GetState()),
                GetNumThreads(), exe_name ? ", executable = " : "",
                exe_name ? exe_name : "");
  } else
    strm.PutCString("No value");

  return true;
}

SBStructuredData SBProcess::GetExtendedCrashInformation() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBStructuredData, SBProcess,
                             GetExtendedCrashInformation);
  SBStructuredData data;
  ProcessSP process_sp(GetSP());
  if (!process_sp)
    return LLDB_RECORD_RESULT(data);

  PlatformSP platform_sp = process_sp->GetTarget().GetPlatform();

  if (!platform_sp)
    return LLDB_RECORD_RESULT(data);

  auto expected_data =
      platform_sp->FetchExtendedCrashInformation(*process_sp.get());

  if (!expected_data)
    return LLDB_RECORD_RESULT(data);

  StructuredData::ObjectSP fetched_data = *expected_data;
  data.m_impl_up->SetObjectSP(fetched_data);
  return LLDB_RECORD_RESULT(data);
}

uint32_t
SBProcess::GetNumSupportedHardwareWatchpoints(lldb::SBError &sb_error) const {
  LLDB_RECORD_METHOD_CONST(uint32_t, SBProcess,
                           GetNumSupportedHardwareWatchpoints,
                           (lldb::SBError &), sb_error);

  uint32_t num = 0;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
    sb_error.SetError(process_sp->GetWatchpointSupportInfo(num));
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }
  return num;
}

uint32_t SBProcess::LoadImage(lldb::SBFileSpec &sb_remote_image_spec,
                              lldb::SBError &sb_error) {
  LLDB_RECORD_METHOD(uint32_t, SBProcess, LoadImage,
                     (lldb::SBFileSpec &, lldb::SBError &),
                     sb_remote_image_spec, sb_error);

  return LoadImage(SBFileSpec(), sb_remote_image_spec, sb_error);
}

uint32_t SBProcess::LoadImage(const lldb::SBFileSpec &sb_local_image_spec,
                              const lldb::SBFileSpec &sb_remote_image_spec,
                              lldb::SBError &sb_error) {
  LLDB_RECORD_METHOD(
      uint32_t, SBProcess, LoadImage,
      (const lldb::SBFileSpec &, const lldb::SBFileSpec &, lldb::SBError &),
      sb_local_image_spec, sb_remote_image_spec, sb_error);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
      PlatformSP platform_sp = process_sp->GetTarget().GetPlatform();
      return platform_sp->LoadImage(process_sp.get(), *sb_local_image_spec,
                                    *sb_remote_image_spec, sb_error.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("process is invalid");
  }
  return LLDB_INVALID_IMAGE_TOKEN;
}

uint32_t SBProcess::LoadImageUsingPaths(const lldb::SBFileSpec &image_spec,
                                        SBStringList &paths,
                                        lldb::SBFileSpec &loaded_path,
                                        lldb::SBError &error) {
  LLDB_RECORD_METHOD(uint32_t, SBProcess, LoadImageUsingPaths,
                     (const lldb::SBFileSpec &, lldb::SBStringList &,
                      lldb::SBFileSpec &, lldb::SBError &),
                     image_spec, paths, loaded_path, error);

  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());
      PlatformSP platform_sp = process_sp->GetTarget().GetPlatform();
      size_t num_paths = paths.GetSize();
      std::vector<std::string> paths_vec;
      paths_vec.reserve(num_paths);
      for (size_t i = 0; i < num_paths; i++)
        paths_vec.push_back(paths.GetStringAtIndex(i));
      FileSpec loaded_spec;

      uint32_t token = platform_sp->LoadImageUsingPaths(
          process_sp.get(), *image_spec, paths_vec, error.ref(), &loaded_spec);
      if (token != LLDB_INVALID_IMAGE_TOKEN)
        loaded_path = loaded_spec;
      return token;
    } else {
      error.SetErrorString("process is running");
    }
  } else {
    error.SetErrorString("process is invalid");
  }

  return LLDB_INVALID_IMAGE_TOKEN;
}

lldb::SBError SBProcess::UnloadImage(uint32_t image_token) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, UnloadImage, (uint32_t),
                     image_token);

  lldb::SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      PlatformSP platform_sp = process_sp->GetTarget().GetPlatform();
      sb_error.SetError(
          platform_sp->UnloadImage(process_sp.get(), image_token));
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else
    sb_error.SetErrorString("invalid process");
  return LLDB_RECORD_RESULT(sb_error);
}

lldb::SBError SBProcess::SendEventData(const char *event_data) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, SendEventData, (const char *),
                     event_data);

  lldb::SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());
      sb_error.SetError(process_sp->SendEventData(event_data));
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else
    sb_error.SetErrorString("invalid process");
  return LLDB_RECORD_RESULT(sb_error);
}

uint32_t SBProcess::GetNumExtendedBacktraceTypes() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBProcess, GetNumExtendedBacktraceTypes);

  ProcessSP process_sp(GetSP());
  if (process_sp && process_sp->GetSystemRuntime()) {
    SystemRuntime *runtime = process_sp->GetSystemRuntime();
    return runtime->GetExtendedBacktraceTypes().size();
  }
  return 0;
}

const char *SBProcess::GetExtendedBacktraceTypeAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(const char *, SBProcess, GetExtendedBacktraceTypeAtIndex,
                     (uint32_t), idx);

  ProcessSP process_sp(GetSP());
  if (process_sp && process_sp->GetSystemRuntime()) {
    SystemRuntime *runtime = process_sp->GetSystemRuntime();
    const std::vector<ConstString> &names =
        runtime->GetExtendedBacktraceTypes();
    if (idx < names.size()) {
      return names[idx].AsCString();
    }
  }
  return nullptr;
}

SBThreadCollection SBProcess::GetHistoryThreads(addr_t addr) {
  LLDB_RECORD_METHOD(lldb::SBThreadCollection, SBProcess, GetHistoryThreads,
                     (lldb::addr_t), addr);

  ProcessSP process_sp(GetSP());
  SBThreadCollection threads;
  if (process_sp) {
    threads = SBThreadCollection(process_sp->GetHistoryThreads(addr));
  }
  return LLDB_RECORD_RESULT(threads);
}

bool SBProcess::IsInstrumentationRuntimePresent(
    InstrumentationRuntimeType type) {
  LLDB_RECORD_METHOD(bool, SBProcess, IsInstrumentationRuntimePresent,
                     (lldb::InstrumentationRuntimeType), type);

  ProcessSP process_sp(GetSP());
  if (!process_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(
      process_sp->GetTarget().GetAPIMutex());

  InstrumentationRuntimeSP runtime_sp =
      process_sp->GetInstrumentationRuntime(type);

  if (!runtime_sp.get())
    return false;

  return runtime_sp->IsActive();
}

lldb::SBError SBProcess::SaveCore(const char *file_name) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, SaveCore, (const char *),
                     file_name);

  lldb::SBError error;
  ProcessSP process_sp(GetSP());
  if (!process_sp) {
    error.SetErrorString("SBProcess is invalid");
    return LLDB_RECORD_RESULT(error);
  }

  std::lock_guard<std::recursive_mutex> guard(
      process_sp->GetTarget().GetAPIMutex());

  if (process_sp->GetState() != eStateStopped) {
    error.SetErrorString("the process is not stopped");
    return LLDB_RECORD_RESULT(error);
  }

  FileSpec core_file(file_name);
  SaveCoreStyle core_style = SaveCoreStyle::eSaveCoreFull;
  error.ref() = PluginManager::SaveCore(process_sp, core_file, core_style);
  return LLDB_RECORD_RESULT(error);
}

lldb::SBError
SBProcess::GetMemoryRegionInfo(lldb::addr_t load_addr,
                               SBMemoryRegionInfo &sb_region_info) {
  LLDB_RECORD_METHOD(lldb::SBError, SBProcess, GetMemoryRegionInfo,
                     (lldb::addr_t, lldb::SBMemoryRegionInfo &), load_addr,
                     sb_region_info);

  lldb::SBError sb_error;
  ProcessSP process_sp(GetSP());
  if (process_sp) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      std::lock_guard<std::recursive_mutex> guard(
          process_sp->GetTarget().GetAPIMutex());

      sb_error.ref() =
          process_sp->GetMemoryRegionInfo(load_addr, sb_region_info.ref());
    } else {
      sb_error.SetErrorString("process is running");
    }
  } else {
    sb_error.SetErrorString("SBProcess is invalid");
  }
  return LLDB_RECORD_RESULT(sb_error);
}

lldb::SBMemoryRegionInfoList SBProcess::GetMemoryRegions() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBMemoryRegionInfoList, SBProcess,
                             GetMemoryRegions);

  lldb::SBMemoryRegionInfoList sb_region_list;

  ProcessSP process_sp(GetSP());
  Process::StopLocker stop_locker;
  if (process_sp && stop_locker.TryLock(&process_sp->GetRunLock())) {
    std::lock_guard<std::recursive_mutex> guard(
        process_sp->GetTarget().GetAPIMutex());

    process_sp->GetMemoryRegions(sb_region_list.ref());
  }

  return LLDB_RECORD_RESULT(sb_region_list);
}

lldb::SBProcessInfo SBProcess::GetProcessInfo() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBProcessInfo, SBProcess, GetProcessInfo);

  lldb::SBProcessInfo sb_proc_info;
  ProcessSP process_sp(GetSP());
  ProcessInstanceInfo proc_info;
  if (process_sp && process_sp->GetProcessInfo(proc_info)) {
    sb_proc_info.SetProcessInfo(proc_info);
  }
  return LLDB_RECORD_RESULT(sb_proc_info);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBProcess>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBProcess, ());
  LLDB_REGISTER_CONSTRUCTOR(SBProcess, (const lldb::SBProcess &));
  LLDB_REGISTER_CONSTRUCTOR(SBProcess, (const lldb::ProcessSP &));
  LLDB_REGISTER_METHOD(const lldb::SBProcess &,
                       SBProcess, operator=,(const lldb::SBProcess &));
  LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess,
                              GetBroadcasterClassName, ());
  LLDB_REGISTER_METHOD(const char *, SBProcess, GetPluginName, ());
  LLDB_REGISTER_METHOD(const char *, SBProcess, GetShortPluginName, ());
  LLDB_REGISTER_METHOD(void, SBProcess, Clear, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBProcess, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBProcess, operator bool, ());
  LLDB_REGISTER_METHOD(bool, SBProcess, RemoteLaunch,
                       (const char **, const char **, const char *,
                        const char *, const char *, const char *, uint32_t,
                        bool, lldb::SBError &));
  LLDB_REGISTER_METHOD(bool, SBProcess, RemoteAttachToProcessWithID,
                       (lldb::pid_t, lldb::SBError &));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumThreads, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBProcess, GetSelectedThread,
                             ());
  LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, CreateOSPluginThread,
                       (lldb::tid_t, lldb::addr_t));
  LLDB_REGISTER_METHOD_CONST(lldb::SBTarget, SBProcess, GetTarget, ());
  LLDB_REGISTER_METHOD(size_t, SBProcess, PutSTDIN, (const char *, size_t));
  LLDB_REGISTER_METHOD_CONST(void, SBProcess, ReportEventState,
                             (const lldb::SBEvent &, FILE *));
  LLDB_REGISTER_METHOD_CONST(void, SBProcess, ReportEventState,
                             (const lldb::SBEvent &, FileSP));
  LLDB_REGISTER_METHOD_CONST(void, SBProcess, ReportEventState,
                             (const lldb::SBEvent &, SBFile));
  LLDB_REGISTER_METHOD(
      void, SBProcess, AppendEventStateReport,
      (const lldb::SBEvent &, lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThread,
                       (const lldb::SBThread &));
  LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThreadByID, (lldb::tid_t));
  LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThreadByIndexID,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadAtIndex, (size_t));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumQueues, ());
  LLDB_REGISTER_METHOD(lldb::SBQueue, SBProcess, GetQueueAtIndex, (size_t));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetStopID, (bool));
  LLDB_REGISTER_METHOD(lldb::SBEvent, SBProcess, GetStopEventForStopID,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::StateType, SBProcess, GetState, ());
  LLDB_REGISTER_METHOD(int, SBProcess, GetExitStatus, ());
  LLDB_REGISTER_METHOD(const char *, SBProcess, GetExitDescription, ());
  LLDB_REGISTER_METHOD(lldb::pid_t, SBProcess, GetProcessID, ());
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetUniqueID, ());
  LLDB_REGISTER_METHOD_CONST(lldb::ByteOrder, SBProcess, GetByteOrder, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBProcess, GetAddressByteSize, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Continue, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Destroy, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Stop, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Kill, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Detach, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Detach, (bool));
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Signal, (int));
  LLDB_REGISTER_METHOD(lldb::SBUnixSignals, SBProcess, GetUnixSignals, ());
  LLDB_REGISTER_METHOD(void, SBProcess, SendAsyncInterrupt, ());
  LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadByID,
                       (lldb::tid_t));
  LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadByIndexID,
                       (uint32_t));
  LLDB_REGISTER_STATIC_METHOD(lldb::StateType, SBProcess, GetStateFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, GetRestartedFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(size_t, SBProcess,
                              GetNumRestartedReasonsFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess,
                              GetRestartedReasonAtIndexFromEvent,
                              (const lldb::SBEvent &, size_t));
  LLDB_REGISTER_STATIC_METHOD(lldb::SBProcess, SBProcess, GetProcessFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, GetInterruptedFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(lldb::SBStructuredData, SBProcess,
                              GetStructuredDataFromEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, EventIsProcessEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, EventIsStructuredDataEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_METHOD_CONST(lldb::SBBroadcaster, SBProcess, GetBroadcaster,
                             ());
  LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess, GetBroadcasterClass,
                              ());
  LLDB_REGISTER_METHOD(uint64_t, SBProcess, ReadUnsignedFromMemory,
                       (lldb::addr_t, uint32_t, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::addr_t, SBProcess, ReadPointerFromMemory,
                       (lldb::addr_t, lldb::SBError &));
  LLDB_REGISTER_METHOD(bool, SBProcess, GetDescription, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(lldb::SBStructuredData, SBProcess,
                       GetExtendedCrashInformation, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBProcess,
                             GetNumSupportedHardwareWatchpoints,
                             (lldb::SBError &));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, LoadImage,
                       (lldb::SBFileSpec &, lldb::SBError &));
  LLDB_REGISTER_METHOD(
      uint32_t, SBProcess, LoadImage,
      (const lldb::SBFileSpec &, const lldb::SBFileSpec &, lldb::SBError &));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, LoadImageUsingPaths,
                       (const lldb::SBFileSpec &, lldb::SBStringList &,
                        lldb::SBFileSpec &, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, UnloadImage, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, SendEventData,
                       (const char *));
  LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumExtendedBacktraceTypes, ());
  LLDB_REGISTER_METHOD(const char *, SBProcess,
                       GetExtendedBacktraceTypeAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBThreadCollection, SBProcess, GetHistoryThreads,
                       (lldb::addr_t));
  LLDB_REGISTER_METHOD(bool, SBProcess, IsInstrumentationRuntimePresent,
                       (lldb::InstrumentationRuntimeType));
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, SaveCore, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, GetMemoryRegionInfo,
                       (lldb::addr_t, lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(lldb::SBMemoryRegionInfoList, SBProcess,
                       GetMemoryRegions, ());
  LLDB_REGISTER_METHOD(lldb::SBProcessInfo, SBProcess, GetProcessInfo, ());

  LLDB_REGISTER_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetSTDOUT);
  LLDB_REGISTER_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetSTDERR);
  LLDB_REGISTER_CHAR_PTR_METHOD_CONST(size_t, SBProcess, GetAsyncProfileData);
}

}
}
