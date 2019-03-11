//===-- SBThreadPlan.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"
#include "lldb/API/SBThread.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamFile.h"
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
#include "lldb/Target/ThreadPlanPython.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StructuredData.h"

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThreadPlan.h"
#include "lldb/API/SBValue.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------
SBThreadPlan::SBThreadPlan() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBThreadPlan); }

SBThreadPlan::SBThreadPlan(const ThreadPlanSP &lldb_object_sp)
    : m_opaque_sp(lldb_object_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBThreadPlan, (const lldb::ThreadPlanSP &),
                          lldb_object_sp);
}

SBThreadPlan::SBThreadPlan(const SBThreadPlan &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBThreadPlan, (const lldb::SBThreadPlan &), rhs);
}

SBThreadPlan::SBThreadPlan(lldb::SBThread &sb_thread, const char *class_name) {
  LLDB_RECORD_CONSTRUCTOR(SBThreadPlan, (lldb::SBThread &, const char *),
                          sb_thread, class_name);

  Thread *thread = sb_thread.get();
  if (thread)
    m_opaque_sp = std::make_shared<ThreadPlanPython>(*thread, class_name);
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------

const lldb::SBThreadPlan &SBThreadPlan::operator=(const SBThreadPlan &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBThreadPlan &,
                     SBThreadPlan, operator=,(const lldb::SBThreadPlan &), rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}
//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBThreadPlan::~SBThreadPlan() {}

lldb_private::ThreadPlan *SBThreadPlan::get() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb_private::ThreadPlan *, SBThreadPlan, get);

  return m_opaque_sp.get();
}

bool SBThreadPlan::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBThreadPlan, IsValid);
  return this->operator bool();
}
SBThreadPlan::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBThreadPlan, operator bool);

  return m_opaque_sp.get() != NULL;
}

void SBThreadPlan::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBThreadPlan, Clear);

  m_opaque_sp.reset();
}

lldb::StopReason SBThreadPlan::GetStopReason() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::StopReason, SBThreadPlan, GetStopReason);

  return eStopReasonNone;
}

size_t SBThreadPlan::GetStopReasonDataCount() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBThreadPlan, GetStopReasonDataCount);

  return 0;
}

uint64_t SBThreadPlan::GetStopReasonDataAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(uint64_t, SBThreadPlan, GetStopReasonDataAtIndex,
                     (uint32_t), idx);

  return 0;
}

SBThread SBThreadPlan::GetThread() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBThread, SBThreadPlan, GetThread);

  if (m_opaque_sp) {
    return LLDB_RECORD_RESULT(
        SBThread(m_opaque_sp->GetThread().shared_from_this()));
  } else
    return LLDB_RECORD_RESULT(SBThread());
}

bool SBThreadPlan::GetDescription(lldb::SBStream &description) const {
  LLDB_RECORD_METHOD_CONST(bool, SBThreadPlan, GetDescription,
                           (lldb::SBStream &), description);

  if (m_opaque_sp) {
    m_opaque_sp->GetDescription(description.get(), eDescriptionLevelFull);
  } else {
    description.Printf("Empty SBThreadPlan");
  }
  return true;
}

void SBThreadPlan::SetThreadPlan(const ThreadPlanSP &lldb_object_sp) {
  m_opaque_sp = lldb_object_sp;
}

void SBThreadPlan::SetPlanComplete(bool success) {
  LLDB_RECORD_METHOD(void, SBThreadPlan, SetPlanComplete, (bool), success);

  if (m_opaque_sp)
    m_opaque_sp->SetPlanComplete(success);
}

bool SBThreadPlan::IsPlanComplete() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBThreadPlan, IsPlanComplete);

  if (m_opaque_sp)
    return m_opaque_sp->IsPlanComplete();
  else
    return true;
}

bool SBThreadPlan::IsPlanStale() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBThreadPlan, IsPlanStale);

  if (m_opaque_sp)
    return m_opaque_sp->IsPlanStale();
  else
    return true;
}

bool SBThreadPlan::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBThreadPlan, IsValid);

  if (m_opaque_sp)
    return m_opaque_sp->ValidatePlan(nullptr);
  else
    return false;
}

// This section allows an SBThreadPlan to push another of the common types of
// plans...
//
// FIXME, you should only be able to queue thread plans from inside the methods
// of a Scripted Thread Plan.  Need a way to enforce that.

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepOverRange(SBAddress &sb_start_address,
                                              lldb::addr_t size) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepOverRange,
                     (lldb::SBAddress &, lldb::addr_t), sb_start_address, size);

  SBError error;
  return LLDB_RECORD_RESULT(
      QueueThreadPlanForStepOverRange(sb_start_address, size, error));
}

SBThreadPlan SBThreadPlan::QueueThreadPlanForStepOverRange(
    SBAddress &sb_start_address, lldb::addr_t size, SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepOverRange,
                     (lldb::SBAddress &, lldb::addr_t, lldb::SBError &),
                     sb_start_address, size, error);

  if (m_opaque_sp) {
    Address *start_address = sb_start_address.get();
    if (!start_address) {
      return LLDB_RECORD_RESULT(SBThreadPlan());
    }

    AddressRange range(*start_address, size);
    SymbolContext sc;
    start_address->CalculateSymbolContext(&sc);
    Status plan_status;

    SBThreadPlan plan =
        SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepOverRange(
            false, range, sc, eAllThreads, plan_status));

    if (plan_status.Fail())
      error.SetErrorString(plan_status.AsCString());

    return LLDB_RECORD_RESULT(plan);
  } else {
    return LLDB_RECORD_RESULT(SBThreadPlan());
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepInRange(SBAddress &sb_start_address,
                                            lldb::addr_t size) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepInRange,
                     (lldb::SBAddress &, lldb::addr_t), sb_start_address, size);

  SBError error;
  return LLDB_RECORD_RESULT(
      QueueThreadPlanForStepInRange(sb_start_address, size, error));
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepInRange(SBAddress &sb_start_address,
                                            lldb::addr_t size, SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepInRange,
                     (lldb::SBAddress &, lldb::addr_t, lldb::SBError &),
                     sb_start_address, size, error);

  if (m_opaque_sp) {
    Address *start_address = sb_start_address.get();
    if (!start_address) {
      return LLDB_RECORD_RESULT(SBThreadPlan());
    }

    AddressRange range(*start_address, size);
    SymbolContext sc;
    start_address->CalculateSymbolContext(&sc);

    Status plan_status;
    SBThreadPlan plan =
        SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepInRange(
            false, range, sc, NULL, eAllThreads, plan_status));

    if (plan_status.Fail())
      error.SetErrorString(plan_status.AsCString());

    return LLDB_RECORD_RESULT(plan);
  } else {
    return LLDB_RECORD_RESULT(SBThreadPlan());
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepOut(uint32_t frame_idx_to_step_to,
                                        bool first_insn) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepOut, (uint32_t, bool),
                     frame_idx_to_step_to, first_insn);

  SBError error;
  return LLDB_RECORD_RESULT(
      QueueThreadPlanForStepOut(frame_idx_to_step_to, first_insn, error));
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepOut(uint32_t frame_idx_to_step_to,
                                        bool first_insn, SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepOut,
                     (uint32_t, bool, lldb::SBError &), frame_idx_to_step_to,
                     first_insn, error);

  if (m_opaque_sp) {
    SymbolContext sc;
    sc = m_opaque_sp->GetThread().GetStackFrameAtIndex(0)->GetSymbolContext(
        lldb::eSymbolContextEverything);

    Status plan_status;
    SBThreadPlan plan =
        SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepOut(
            false, &sc, first_insn, false, eVoteYes, eVoteNoOpinion,
            frame_idx_to_step_to, plan_status));

    if (plan_status.Fail())
      error.SetErrorString(plan_status.AsCString());

    return LLDB_RECORD_RESULT(plan);
  } else {
    return LLDB_RECORD_RESULT(SBThreadPlan());
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForRunToAddress(SBAddress sb_address) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForRunToAddress, (lldb::SBAddress),
                     sb_address);

  SBError error;
  return LLDB_RECORD_RESULT(QueueThreadPlanForRunToAddress(sb_address, error));
}

SBThreadPlan SBThreadPlan::QueueThreadPlanForRunToAddress(SBAddress sb_address,
                                                          SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForRunToAddress,
                     (lldb::SBAddress, lldb::SBError &), sb_address, error);

  if (m_opaque_sp) {
    Address *address = sb_address.get();
    if (!address)
      return LLDB_RECORD_RESULT(SBThreadPlan());

    Status plan_status;
    SBThreadPlan plan =
        SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForRunToAddress(
            false, *address, false, plan_status));

    if (plan_status.Fail())
      error.SetErrorString(plan_status.AsCString());

    return LLDB_RECORD_RESULT(plan);
  } else {
    return LLDB_RECORD_RESULT(SBThreadPlan());
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepScripted(const char *script_class_name) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepScripted, (const char *),
                     script_class_name);

  SBError error;
  return LLDB_RECORD_RESULT(
      QueueThreadPlanForStepScripted(script_class_name, error));
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepScripted(const char *script_class_name,
                                             SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                     QueueThreadPlanForStepScripted,
                     (const char *, lldb::SBError &), script_class_name, error);

  if (m_opaque_sp) {
    Status plan_status;
    SBThreadPlan plan =
        SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepScripted(
            false, script_class_name, false, plan_status));

    if (plan_status.Fail())
      error.SetErrorString(plan_status.AsCString());

    return LLDB_RECORD_RESULT(plan);
  } else {
    return LLDB_RECORD_RESULT(SBThreadPlan());
  }
}
