//===-- SBThread.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBThread.h"
#include "Utils.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFormat.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBThreadPlan.h"
#include "lldb/API/SBValue.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Queue.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/SystemRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

const char *SBThread::GetBroadcasterClassName() {
  LLDB_INSTRUMENT();

  return ConstString(Thread::GetStaticBroadcasterClass()).AsCString();
}

// Constructors
SBThread::SBThread() : m_opaque_sp(new ExecutionContextRef()) {
  LLDB_INSTRUMENT_VA(this);
}

SBThread::SBThread(const ThreadSP &lldb_object_sp)
    : m_opaque_sp(new ExecutionContextRef(lldb_object_sp)) {
  LLDB_INSTRUMENT_VA(this, lldb_object_sp);
}

SBThread::SBThread(const SBThread &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_sp = clone(rhs.m_opaque_sp);
}

// Assignment operator

const lldb::SBThread &SBThread::operator=(const SBThread &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_sp = clone(rhs.m_opaque_sp);
  return *this;
}

// Destructor
SBThread::~SBThread() = default;

lldb::SBQueue SBThread::GetQueue() const {
  LLDB_INSTRUMENT_VA(this);

  SBQueue sb_queue;
  QueueSP queue_sp;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBQueue();
  }

  if (exe_ctx->HasThreadScope()) {
    queue_sp = exe_ctx->GetThreadPtr()->GetQueue();
    if (queue_sp) {
      sb_queue.SetQueue(queue_sp);
    }
  }

  return sb_queue;
}

bool SBThread::IsValid() const {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}
SBThread::operator bool() const {
  LLDB_INSTRUMENT_VA(this);
  if (!m_opaque_sp)
    return false;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  return m_opaque_sp->GetThreadSP().get() != nullptr;
}

void SBThread::Clear() {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_sp->Clear();
}

StopReason SBThread::GetStopReason() {
  LLDB_INSTRUMENT_VA(this);

  StopReason reason = eStopReasonInvalid;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return reason;
  }

  if (exe_ctx->HasThreadScope())
    return exe_ctx->GetThreadPtr()->GetStopReason();

  return reason;
}

size_t SBThread::GetStopReasonDataCount() {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (exe_ctx) {
    if (exe_ctx->HasThreadScope()) {
      StopInfoSP stop_info_sp = exe_ctx->GetThreadPtr()->GetStopInfo();
      if (stop_info_sp)
        return stop_info_sp->GetStopReasonDataCount();
    }
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return 0;
  }
  return 0;
}

uint64_t SBThread::GetStopReasonDataAtIndex(uint32_t idx) {
  LLDB_INSTRUMENT_VA(this, idx);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (exe_ctx) {
    if (exe_ctx->HasThreadScope()) {
      Thread *thread = exe_ctx->GetThreadPtr();
      StopInfoSP stop_info_sp = thread->GetStopInfo();
      if (stop_info_sp)
        return stop_info_sp->GetStopReasonDataAtIndex(idx);
    }
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return 0;
  }
  return 0;
}

bool SBThread::GetStopReasonExtendedInfoAsJSON(lldb::SBStream &stream) {
  LLDB_INSTRUMENT_VA(this, stream);

  Stream &strm = stream.ref();

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  if (!exe_ctx->HasThreadScope())
    return false;

  StopInfoSP stop_info = exe_ctx->GetThreadPtr()->GetStopInfo();
  StructuredData::ObjectSP info = stop_info->GetExtendedInfo();
  if (!info)
    return false;

  info->Dump(strm);

  return true;
}

SBThreadCollection
SBThread::GetStopReasonExtendedBacktraces(InstrumentationRuntimeType type) {
  LLDB_INSTRUMENT_VA(this, type);

  SBThreadCollection threads;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBThreadCollection();
  }

  if (!exe_ctx->HasThreadScope())
    return SBThreadCollection();

  ProcessSP process_sp = exe_ctx->GetProcessSP();

  StopInfoSP stop_info = exe_ctx->GetThreadPtr()->GetStopInfo();
  StructuredData::ObjectSP info = stop_info->GetExtendedInfo();
  if (!info)
    return threads;

  threads = process_sp->GetInstrumentationRuntime(type)
                ->GetBacktracesFromExtendedStopInfo(info);
  return threads;
}

size_t SBThread::GetStopDescription(char *dst, size_t dst_len) {
  LLDB_INSTRUMENT_VA(this, dst, dst_len);

  if (dst)
    *dst = 0;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return 0;
  }

  if (!exe_ctx->HasThreadScope())
    return 0;

  std::string thread_stop_desc = exe_ctx->GetThreadPtr()->GetStopDescription();
  if (thread_stop_desc.empty())
    return 0;

  if (dst)
    return ::snprintf(dst, dst_len, "%s", thread_stop_desc.c_str()) + 1;

  // NULL dst passed in, return the length needed to contain the
  // description.
  return thread_stop_desc.size() + 1; // Include the NULL byte for size
}

SBValue SBThread::GetStopReturnValue() {
  LLDB_INSTRUMENT_VA(this);

  ValueObjectSP return_valobj_sp;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBValue();
  }

  if (exe_ctx->HasThreadScope()) {
    StopInfoSP stop_info_sp = exe_ctx->GetThreadPtr()->GetStopInfo();
    if (stop_info_sp) {
      return_valobj_sp = StopInfo::GetReturnValueObject(stop_info_sp);
    }
  }

  return SBValue(return_valobj_sp);
}

void SBThread::SetThread(const ThreadSP &lldb_object_sp) {
  m_opaque_sp->SetThreadSP(lldb_object_sp);
}

lldb::tid_t SBThread::GetThreadID() const {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (thread_sp)
    return thread_sp->GetID();
  return LLDB_INVALID_THREAD_ID;
}

uint32_t SBThread::GetIndexID() const {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (thread_sp)
    return thread_sp->GetIndexID();
  return LLDB_INVALID_INDEX32;
}

const char *SBThread::GetName() const {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return nullptr;
  }

  if (!exe_ctx->HasThreadScope())
    return nullptr;

  return ConstString(exe_ctx->GetThreadPtr()->GetName()).GetCString();
}

const char *SBThread::GetQueueName() const {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return nullptr;
  }

  if (!exe_ctx->HasThreadScope())
    return nullptr;

  return ConstString(exe_ctx->GetThreadPtr()->GetQueueName()).GetCString();
}

lldb::queue_id_t SBThread::GetQueueID() const {
  LLDB_INSTRUMENT_VA(this);

  queue_id_t id = LLDB_INVALID_QUEUE_ID;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return id;
  }

  if (exe_ctx->HasThreadScope()) {
    id = exe_ctx->GetThreadPtr()->GetQueueID();
  }

  return id;
}

bool SBThread::GetInfoItemByPathAsString(const char *path, SBStream &strm) {
  LLDB_INSTRUMENT_VA(this, path, strm);

  bool success = false;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (exe_ctx) {
    if (exe_ctx->HasThreadScope()) {
      Thread *thread = exe_ctx->GetThreadPtr();
      StructuredData::ObjectSP info_root_sp = thread->GetExtendedInfo();
      if (info_root_sp) {
        StructuredData::ObjectSP node =
            info_root_sp->GetObjectForDotSeparatedPath(path);
        if (node) {
          if (node->GetType() == eStructuredDataTypeString) {
            strm.ref() << node->GetAsString()->GetValue();
            success = true;
          }
          if (node->GetType() == eStructuredDataTypeInteger) {
            strm.Printf("0x%" PRIx64, node->GetUnsignedIntegerValue());
            success = true;
          }
          if (node->GetType() == eStructuredDataTypeFloat) {
            strm.Printf("0x%f", node->GetAsFloat()->GetValue());
            success = true;
          }
          if (node->GetType() == eStructuredDataTypeBoolean) {
            if (node->GetAsBoolean()->GetValue())
              strm.Printf("true");
            else
              strm.Printf("false");
            success = true;
          }
          if (node->GetType() == eStructuredDataTypeNull) {
            strm.Printf("null");
            success = true;
          }
        }
      }
    }
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return success;
  }

  return success;
}

static Status ResumeNewPlan(StoppedExecutionContext exe_ctx,
                            ThreadPlan *new_plan) {
  Thread *thread = exe_ctx.GetThreadPtr();
  if (!thread)
    return Status::FromErrorString("No thread in SBThread::ResumeNewPlan");

  // User level plans should be Controlling Plans so they can be interrupted,
  // other plans executed, and then a "continue" will resume the plan.
  if (new_plan != nullptr) {
    new_plan->SetIsControllingPlan(true);
    new_plan->SetOkayToDiscard(false);
  }

  // Why do we need to set the current thread by ID here???
  Process *process = exe_ctx.GetProcessPtr();
  process->GetThreadList().SetSelectedThreadByID(thread->GetID());

  // Release the run lock but keep the API lock.
  std::unique_lock<std::recursive_mutex> api_lock = exe_ctx.AllowResume();
  if (process->GetTarget().GetDebugger().GetAsyncExecution())
    return process->Resume();
  return process->ResumeSynchronous(nullptr);
}

void SBThread::StepOver(lldb::RunMode stop_other_threads) {
  LLDB_INSTRUMENT_VA(this, stop_other_threads);

  SBError error; // Ignored
  StepOver(stop_other_threads, error);
}

void SBThread::StepOver(lldb::RunMode stop_other_threads, SBError &error) {
  LLDB_INSTRUMENT_VA(this, stop_other_threads, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  Thread *thread = exe_ctx->GetThreadPtr();
  bool abort_other_plans = false;
  StackFrameSP frame_sp(thread->GetStackFrameAtIndex(0));

  Status new_plan_status;
  ThreadPlanSP new_plan_sp;
  if (frame_sp) {
    if (frame_sp->HasDebugInformation()) {
      const LazyBool avoid_no_debug = eLazyBoolCalculate;
      SymbolContext sc(frame_sp->GetSymbolContext(eSymbolContextEverything));
      new_plan_sp = thread->QueueThreadPlanForStepOverRange(
          abort_other_plans, sc.line_entry, sc, stop_other_threads,
          new_plan_status, avoid_no_debug);
    } else {
      new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction(
          true, abort_other_plans, stop_other_threads, new_plan_status);
    }
  }
  error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
}

void SBThread::StepInto(lldb::RunMode stop_other_threads) {
  LLDB_INSTRUMENT_VA(this, stop_other_threads);

  StepInto(nullptr, stop_other_threads);
}

void SBThread::StepInto(const char *target_name,
                        lldb::RunMode stop_other_threads) {
  LLDB_INSTRUMENT_VA(this, target_name, stop_other_threads);

  SBError error; // Ignored
  StepInto(target_name, LLDB_INVALID_LINE_NUMBER, error, stop_other_threads);
}

void SBThread::StepInto(const char *target_name, uint32_t end_line,
                        SBError &error, lldb::RunMode stop_other_threads) {
  LLDB_INSTRUMENT_VA(this, target_name, end_line, error, stop_other_threads);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  bool abort_other_plans = false;

  Thread *thread = exe_ctx->GetThreadPtr();
  StackFrameSP frame_sp(thread->GetStackFrameAtIndex(0));
  ThreadPlanSP new_plan_sp;
  Status new_plan_status;

  if (frame_sp && frame_sp->HasDebugInformation()) {
    SymbolContext sc(frame_sp->GetSymbolContext(eSymbolContextEverything));
    AddressRange range;
    if (end_line == LLDB_INVALID_LINE_NUMBER)
      range = sc.line_entry.range;
    else {
      llvm::Error err = sc.GetAddressRangeFromHereToEndLine(end_line, range);
      if (err) {
        error = Status::FromErrorString(llvm::toString(std::move(err)).c_str());
        return;
      }
    }

    const LazyBool step_out_avoids_code_without_debug_info =
        eLazyBoolCalculate;
    const LazyBool step_in_avoids_code_without_debug_info =
        eLazyBoolCalculate;
    new_plan_sp = thread->QueueThreadPlanForStepInRange(
        abort_other_plans, range, sc, target_name, stop_other_threads,
        new_plan_status, step_in_avoids_code_without_debug_info,
        step_out_avoids_code_without_debug_info);
  } else {
    new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction(
        false, abort_other_plans, stop_other_threads, new_plan_status);
  }

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());
}

void SBThread::StepOut() {
  LLDB_INSTRUMENT_VA(this);

  SBError error; // Ignored
  StepOut(error);
}

void SBThread::StepOut(SBError &error) {
  LLDB_INSTRUMENT_VA(this, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  bool abort_other_plans = false;
  bool stop_other_threads = false;

  Thread *thread = exe_ctx->GetThreadPtr();

  const LazyBool avoid_no_debug = eLazyBoolCalculate;
  Status new_plan_status;
  ThreadPlanSP new_plan_sp(thread->QueueThreadPlanForStepOut(
      abort_other_plans, nullptr, false, stop_other_threads, eVoteYes,
      eVoteNoOpinion, 0, new_plan_status, avoid_no_debug));

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());
}

void SBThread::StepOutOfFrame(SBFrame &sb_frame) {
  LLDB_INSTRUMENT_VA(this, sb_frame);

  SBError error; // Ignored
  StepOutOfFrame(sb_frame, error);
}

void SBThread::StepOutOfFrame(SBFrame &sb_frame, SBError &error) {
  LLDB_INSTRUMENT_VA(this, sb_frame, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!sb_frame.IsValid()) {
    error = Status::FromErrorString("passed invalid SBFrame object");
    return;
  }

  StackFrameSP frame_sp(sb_frame.GetFrameSP());

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  bool abort_other_plans = false;
  bool stop_other_threads = false;
  Thread *thread = exe_ctx->GetThreadPtr();
  if (sb_frame.GetThread().GetThreadID() != thread->GetID()) {
    error = Status::FromErrorString("passed a frame from another thread");
    return;
  }

  Status new_plan_status;
  ThreadPlanSP new_plan_sp(thread->QueueThreadPlanForStepOut(
      abort_other_plans, nullptr, false, stop_other_threads, eVoteYes,
      eVoteNoOpinion, frame_sp->GetFrameIndex(), new_plan_status));

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());
}

void SBThread::StepInstruction(bool step_over) {
  LLDB_INSTRUMENT_VA(this, step_over);

  SBError error; // Ignored
  StepInstruction(step_over, error);
}

void SBThread::StepInstruction(bool step_over, SBError &error) {
  LLDB_INSTRUMENT_VA(this, step_over, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  Thread *thread = exe_ctx->GetThreadPtr();
  Status new_plan_status;
  ThreadPlanSP new_plan_sp(thread->QueueThreadPlanForStepSingleInstruction(
      step_over, false, true, new_plan_status));

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());
}

void SBThread::RunToAddress(lldb::addr_t addr) {
  LLDB_INSTRUMENT_VA(this, addr);

  SBError error; // Ignored
  RunToAddress(addr, error);
}

void SBThread::RunToAddress(lldb::addr_t addr, SBError &error) {
  LLDB_INSTRUMENT_VA(this, addr, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return;
  }

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return;
  }

  bool abort_other_plans = false;
  bool stop_other_threads = true;

  Address target_addr(addr);

  Thread *thread = exe_ctx->GetThreadPtr();

  Status new_plan_status;
  ThreadPlanSP new_plan_sp(thread->QueueThreadPlanForRunToAddress(
      abort_other_plans, target_addr, stop_other_threads, new_plan_status));

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());
}

SBError SBThread::StepOverUntil(lldb::SBFrame &sb_frame,
                                lldb::SBFileSpec &sb_file_spec, uint32_t line) {
  LLDB_INSTRUMENT_VA(this, sb_frame, sb_file_spec, line);

  SBError sb_error;
  char path[PATH_MAX];

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx)
    return Status::FromError(exe_ctx.takeError());

  StackFrameSP frame_sp(sb_frame.GetFrameSP());

  if (exe_ctx->HasThreadScope()) {
    Target *target = exe_ctx->GetTargetPtr();
    Thread *thread = exe_ctx->GetThreadPtr();

    if (line == 0) {
      sb_error = Status::FromErrorString("invalid line argument");
      return sb_error;
    }

    if (!frame_sp) {
      // We don't want to run SelectMostRelevantFrame here, for instance if
      // you called a sequence of StepOverUntil's you wouldn't want the
      // frame changed out from under you because you stepped into a
      // recognized frame.
      frame_sp = thread->GetSelectedFrame(DoNoSelectMostRelevantFrame);
      if (!frame_sp)
        frame_sp = thread->GetStackFrameAtIndex(0);
    }

    SymbolContext frame_sc;
    if (!frame_sp) {
      sb_error = Status::FromErrorString("no valid frames in thread to step");
      return sb_error;
    }

    // If we have a frame, get its line
    frame_sc = frame_sp->GetSymbolContext(
        eSymbolContextCompUnit | eSymbolContextFunction |
        eSymbolContextLineEntry | eSymbolContextSymbol);

    if (frame_sc.comp_unit == nullptr) {
      sb_error = Status::FromErrorStringWithFormat(
          "frame %u doesn't have debug information", frame_sp->GetFrameIndex());
      return sb_error;
    }

    FileSpec step_file_spec;
    if (sb_file_spec.IsValid()) {
      // The file spec passed in was valid, so use it
      step_file_spec = sb_file_spec.ref();
    } else {
      if (frame_sc.line_entry.IsValid())
        step_file_spec = frame_sc.line_entry.GetFile();
      else {
        sb_error = Status::FromErrorString(
            "invalid file argument or no file for frame");
        return sb_error;
      }
    }

    // Grab the current function, then we will make sure the "until" address is
    // within the function.  We discard addresses that are out of the current
    // function, and then if there are no addresses remaining, give an
    // appropriate error message.

    bool all_in_function = true;

    std::vector<addr_t> step_over_until_addrs;
    const bool abort_other_plans = false;
    const bool stop_other_threads = false;
    // TODO: Handle SourceLocationSpec column information
    SourceLocationSpec location_spec(
        step_file_spec, line, /*column=*/std::nullopt, /*check_inlines=*/true,
        /*exact_match=*/false);

    SymbolContextList sc_list;
    frame_sc.comp_unit->ResolveSymbolContext(location_spec,
                                             eSymbolContextLineEntry, sc_list);
    for (const SymbolContext &sc : sc_list) {
      addr_t step_addr =
          sc.line_entry.range.GetBaseAddress().GetLoadAddress(target);
      if (step_addr != LLDB_INVALID_ADDRESS) {
        AddressRange unused_range;
        if (frame_sc.function->GetRangeContainingLoadAddress(step_addr, *target,
                                                             unused_range))
          step_over_until_addrs.push_back(step_addr);
        else
          all_in_function = false;
      }
    }

    if (step_over_until_addrs.empty()) {
      if (all_in_function) {
        step_file_spec.GetPath(path, sizeof(path));
        sb_error = Status::FromErrorStringWithFormat(
            "No line entries for %s:%u", path, line);
      } else
        sb_error = Status::FromErrorString(
            "step until target not in current function");
    } else {
      Status new_plan_status;
      ThreadPlanSP new_plan_sp(thread->QueueThreadPlanForStepUntil(
          abort_other_plans, &step_over_until_addrs[0],
          step_over_until_addrs.size(), stop_other_threads,
          frame_sp->GetFrameIndex(), new_plan_status));

      if (new_plan_status.Success())
        sb_error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
      else
        sb_error = Status::FromErrorString(new_plan_status.AsCString());
    }
  } else {
    sb_error = Status::FromErrorString("this SBThread object is invalid");
  }
  return sb_error;
}

SBError SBThread::StepUsingScriptedThreadPlan(const char *script_class_name) {
  LLDB_INSTRUMENT_VA(this, script_class_name);

  return StepUsingScriptedThreadPlan(script_class_name, true);
}

SBError SBThread::StepUsingScriptedThreadPlan(const char *script_class_name,
                                            bool resume_immediately) {
  LLDB_INSTRUMENT_VA(this, script_class_name, resume_immediately);

  lldb::SBStructuredData no_data;
  return StepUsingScriptedThreadPlan(script_class_name, no_data,
                                     resume_immediately);
}

SBError SBThread::StepUsingScriptedThreadPlan(const char *script_class_name,
                                              SBStructuredData &args_data,
                                              bool resume_immediately) {
  LLDB_INSTRUMENT_VA(this, script_class_name, args_data, resume_immediately);

  SBError error;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx)
    return Status::FromError(exe_ctx.takeError());

  if (!exe_ctx->HasThreadScope()) {
    error = Status::FromErrorString("this SBThread object is invalid");
    return error;
  }

  Thread *thread = exe_ctx->GetThreadPtr();
  Status new_plan_status;
  StructuredData::ObjectSP obj_sp = args_data.m_impl_up->GetObjectSP();

  ThreadPlanSP new_plan_sp = thread->QueueThreadPlanForStepScripted(
      false, script_class_name, obj_sp, false, new_plan_status);

  if (new_plan_status.Fail()) {
    error = Status::FromErrorString(new_plan_status.AsCString());
    return error;
  }

  if (!resume_immediately)
    return error;

  if (new_plan_status.Success())
    error = ResumeNewPlan(std::move(*exe_ctx), new_plan_sp.get());
  else
    error = Status::FromErrorString(new_plan_status.AsCString());

  return error;
}

SBError SBThread::JumpToLine(lldb::SBFileSpec &file_spec, uint32_t line) {
  LLDB_INSTRUMENT_VA(this, file_spec, line);

  SBError sb_error;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx)
    return Status::FromError(exe_ctx.takeError());

  if (!exe_ctx->HasThreadScope()) {
    sb_error = Status::FromErrorString("this SBThread object is invalid");
    return sb_error;
  }

  Thread *thread = exe_ctx->GetThreadPtr();

  Status err = thread->JumpToLine(file_spec.ref(), line, true);
  sb_error.SetError(std::move(err));
  return sb_error;
}

SBError SBThread::ReturnFromFrame(SBFrame &frame, SBValue &return_value) {
  LLDB_INSTRUMENT_VA(this, frame, return_value);

  SBError sb_error;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx)
    return Status::FromError(exe_ctx.takeError());

  if (exe_ctx->HasThreadScope()) {
    Thread *thread = exe_ctx->GetThreadPtr();
    sb_error.SetError(
        thread->ReturnFromFrame(frame.GetFrameSP(), return_value.GetSP()));
  }

  return sb_error;
}

SBError SBThread::UnwindInnermostExpression() {
  LLDB_INSTRUMENT_VA(this);

  SBError sb_error;

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx)
    return Status::FromError(exe_ctx.takeError());

  if (exe_ctx->HasThreadScope()) {
    Thread *thread = exe_ctx->GetThreadPtr();
    sb_error.SetError(thread->UnwindInnermostExpression());
    if (sb_error.Success())
      thread->SetSelectedFrameByIndex(0, false);
  }

  return sb_error;
}

bool SBThread::Suspend() {
  LLDB_INSTRUMENT_VA(this);

  SBError error; // Ignored
  return Suspend(error);
}

bool SBThread::Suspend(SBError &error) {
  LLDB_INSTRUMENT_VA(this, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    error = Status::FromError(exe_ctx.takeError());
    return false;
  }

  bool result = false;
  if (exe_ctx->HasThreadScope()) {
    exe_ctx->GetThreadPtr()->SetResumeState(eStateSuspended);
    result = true;
  } else
    error = Status::FromErrorString("this SBThread object is invalid");
  return result;
}

bool SBThread::Resume() {
  LLDB_INSTRUMENT_VA(this);

  SBError error; // Ignored
  return Resume(error);
}

bool SBThread::Resume(SBError &error) {
  LLDB_INSTRUMENT_VA(this, error);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    error = Status::FromErrorString("process is running");
    return false;
  }

  bool result = false;
  if (exe_ctx->HasThreadScope()) {
    const bool override_suspend = true;
    exe_ctx->GetThreadPtr()->SetResumeState(eStateRunning, override_suspend);
    result = true;
  } else
    error = Status::FromErrorString("this SBThread object is invalid");
  return result;
}

bool SBThread::IsSuspended() {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  if (exe_ctx->HasThreadScope())
    return exe_ctx->GetThreadPtr()->GetResumeState() == eStateSuspended;
  return false;
}

bool SBThread::IsStopped() {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  if (exe_ctx->HasThreadScope())
    return StateIsStoppedState(exe_ctx->GetThreadPtr()->GetState(), true);
  return false;
}

SBProcess SBThread::GetProcess() {
  LLDB_INSTRUMENT_VA(this);

  SBProcess sb_process;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBProcess();
  }

  if (exe_ctx->HasThreadScope()) {
    // Have to go up to the target so we can get a shared pointer to our
    // process...
    sb_process.SetSP(exe_ctx->GetProcessSP());
  }

  return sb_process;
}

uint32_t SBThread::GetNumFrames() {
  LLDB_INSTRUMENT_VA(this);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return 0;
  }

  if (exe_ctx->HasThreadScope())
    return exe_ctx->GetThreadPtr()->GetStackFrameCount();

  return 0;
}

SBFrame SBThread::GetFrameAtIndex(uint32_t idx) {
  LLDB_INSTRUMENT_VA(this, idx);

  SBFrame sb_frame;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBFrame();
  }

  if (exe_ctx->HasThreadScope()) {
    StackFrameSP frame_sp = exe_ctx->GetThreadPtr()->GetStackFrameAtIndex(idx);
    sb_frame.SetFrameSP(frame_sp);
  }

  return sb_frame;
}

lldb::SBFrame SBThread::GetSelectedFrame() {
  LLDB_INSTRUMENT_VA(this);

  SBFrame sb_frame;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBFrame();
  }

  if (exe_ctx->HasThreadScope()) {
    StackFrameSP frame_sp =
        exe_ctx->GetThreadPtr()->GetSelectedFrame(SelectMostRelevantFrame);
    sb_frame.SetFrameSP(frame_sp);
  }

  return sb_frame;
}

lldb::SBFrame SBThread::SetSelectedFrame(uint32_t idx) {
  LLDB_INSTRUMENT_VA(this, idx);

  SBFrame sb_frame;
  StackFrameSP frame_sp;
  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return SBFrame();
  }

  if (exe_ctx->HasThreadScope()) {
    Thread *thread = exe_ctx->GetThreadPtr();
    frame_sp = thread->GetStackFrameAtIndex(idx);
    if (frame_sp) {
      thread->SetSelectedFrame(frame_sp.get());
      sb_frame.SetFrameSP(frame_sp);
    }
  }

  return sb_frame;
}

bool SBThread::EventIsThreadEvent(const SBEvent &event) {
  LLDB_INSTRUMENT_VA(event);

  return Thread::ThreadEventData::GetEventDataFromEvent(event.get()) != nullptr;
}

SBFrame SBThread::GetStackFrameFromEvent(const SBEvent &event) {
  LLDB_INSTRUMENT_VA(event);

  return Thread::ThreadEventData::GetStackFrameFromEvent(event.get());
}

SBThread SBThread::GetThreadFromEvent(const SBEvent &event) {
  LLDB_INSTRUMENT_VA(event);

  return Thread::ThreadEventData::GetThreadFromEvent(event.get());
}

bool SBThread::operator==(const SBThread &rhs) const {
  LLDB_INSTRUMENT_VA(this, rhs);

  return m_opaque_sp->GetThreadSP().get() ==
         rhs.m_opaque_sp->GetThreadSP().get();
}

bool SBThread::operator!=(const SBThread &rhs) const {
  LLDB_INSTRUMENT_VA(this, rhs);

  return m_opaque_sp->GetThreadSP().get() !=
         rhs.m_opaque_sp->GetThreadSP().get();
}

bool SBThread::GetStatus(SBStream &status) const {
  LLDB_INSTRUMENT_VA(this, status);

  Stream &strm = status.ref();

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  if (exe_ctx->HasThreadScope()) {
    exe_ctx->GetThreadPtr()->GetStatus(strm, 0, 1, 1, true,
                                       /*show_hidden=*/true);
  } else
    strm.PutCString("No status");

  return true;
}

bool SBThread::GetDescription(SBStream &description) const {
  LLDB_INSTRUMENT_VA(this, description);

  return GetDescription(description, false);
}

bool SBThread::GetDescription(SBStream &description, bool stop_format) const {
  LLDB_INSTRUMENT_VA(this, description, stop_format);

  Stream &strm = description.ref();

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return false;
  }

  if (exe_ctx->HasThreadScope()) {
    exe_ctx->GetThreadPtr()->DumpUsingSettingsFormat(
        strm, LLDB_INVALID_THREAD_ID, stop_format);
  } else
    strm.PutCString("No value");

  return true;
}

SBError SBThread::GetDescriptionWithFormat(const SBFormat &format,
                                           SBStream &output) {
  Stream &strm = output.ref();

  SBError error;
  if (!format) {
    error = Status::FromErrorString("The provided SBFormat object is invalid");
    return error;
  }

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  if (!exe_ctx) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
    return error;
  }

  if (exe_ctx->HasThreadScope()) {
    if (exe_ctx->GetThreadPtr()->DumpUsingFormat(
            strm, LLDB_INVALID_THREAD_ID, format.GetFormatEntrySP().get())) {
      return error;
    }
  }

  error = Status::FromErrorStringWithFormat(
      "It was not possible to generate a thread description with the given "
      "format string '%s'",
      format.GetFormatEntrySP()->string.c_str());
  return error;
}

SBThread SBThread::GetExtendedBacktraceThread(const char *type) {
  LLDB_INSTRUMENT_VA(this, type);

  llvm::Expected<StoppedExecutionContext> exe_ctx =
      GetStoppedExecutionContext(m_opaque_sp);
  SBThread sb_origin_thread;
  if (exe_ctx) {
    if (exe_ctx->HasThreadScope()) {
      ThreadSP real_thread(exe_ctx->GetThreadSP());
      if (real_thread) {
        ConstString type_const(type);
        Process *process = exe_ctx->GetProcessPtr();
        if (process) {
          SystemRuntime *runtime = process->GetSystemRuntime();
          if (runtime) {
            ThreadSP new_thread_sp(
                runtime->GetExtendedBacktraceThread(real_thread, type_const));
            if (new_thread_sp) {
              // Save this in the Process' ExtendedThreadList so a strong
              // pointer retains the object.
              process->GetExtendedThreadList().AddThread(new_thread_sp);
              sb_origin_thread.SetThread(new_thread_sp);
            }
          }
        }
      }
    }
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::API), exe_ctx.takeError(), "{0}");
  }

  return sb_origin_thread;
}

uint32_t SBThread::GetExtendedBacktraceOriginatingIndexID() {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (thread_sp)
    return thread_sp->GetExtendedBacktraceOriginatingIndexID();
  return LLDB_INVALID_INDEX32;
}

SBValue SBThread::GetCurrentException() {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (!thread_sp)
    return SBValue();

  return SBValue(thread_sp->GetCurrentException());
}

SBThread SBThread::GetCurrentExceptionBacktrace() {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (!thread_sp)
    return SBThread();

  return SBThread(thread_sp->GetCurrentExceptionBacktrace());
}

bool SBThread::SafeToCallFunctions() {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp(m_opaque_sp->GetThreadSP());
  if (thread_sp)
    return thread_sp->SafeToCallFunctions();
  return true;
}

lldb::ThreadSP SBThread::GetSP() const { return m_opaque_sp->GetThreadSP(); }

lldb_private::Thread *SBThread::operator->() {
  return get();
}

lldb_private::Thread *SBThread::get() {
  return m_opaque_sp->GetThreadSP().get();
}

SBValue SBThread::GetSiginfo() {
  LLDB_INSTRUMENT_VA(this);

  ThreadSP thread_sp = m_opaque_sp->GetThreadSP();
  if (!thread_sp)
    return SBValue();
  return thread_sp->GetSiginfoValue();
}
