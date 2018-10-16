//===-- SBThreadPlan.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBThread.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/State.h"
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
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StructuredData.h"

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThreadPlan.h"
#include "lldb/API/SBValue.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------
SBThreadPlan::SBThreadPlan() {}

SBThreadPlan::SBThreadPlan(const ThreadPlanSP &lldb_object_sp)
    : m_opaque_sp(lldb_object_sp) {}

SBThreadPlan::SBThreadPlan(const SBThreadPlan &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {}

SBThreadPlan::SBThreadPlan(lldb::SBThread &sb_thread, const char *class_name) {
  Thread *thread = sb_thread.get();
  if (thread)
    m_opaque_sp.reset(new ThreadPlanPython(*thread, class_name));
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------

const lldb::SBThreadPlan &SBThreadPlan::operator=(const SBThreadPlan &rhs) {
  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}
//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBThreadPlan::~SBThreadPlan() {}

lldb_private::ThreadPlan *SBThreadPlan::get() { return m_opaque_sp.get(); }

bool SBThreadPlan::IsValid() const { return m_opaque_sp.get() != NULL; }

void SBThreadPlan::Clear() { m_opaque_sp.reset(); }

lldb::StopReason SBThreadPlan::GetStopReason() { return eStopReasonNone; }

size_t SBThreadPlan::GetStopReasonDataCount() { return 0; }

uint64_t SBThreadPlan::GetStopReasonDataAtIndex(uint32_t idx) { return 0; }

SBThread SBThreadPlan::GetThread() const {
  if (m_opaque_sp) {
    return SBThread(m_opaque_sp->GetThread().shared_from_this());
  } else
    return SBThread();
}

bool SBThreadPlan::GetDescription(lldb::SBStream &description) const {
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
  if (m_opaque_sp)
    m_opaque_sp->SetPlanComplete(success);
}

bool SBThreadPlan::IsPlanComplete() {
  if (m_opaque_sp)
    return m_opaque_sp->IsPlanComplete();
  else
    return true;
}

bool SBThreadPlan::IsPlanStale() {
  if (m_opaque_sp)
    return m_opaque_sp->IsPlanStale();
  else
    return true;
}

bool SBThreadPlan::IsValid() {
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
  if (m_opaque_sp) {
    Address *start_address = sb_start_address.get();
    if (!start_address) {
      return SBThreadPlan();
    }

    AddressRange range(*start_address, size);
    SymbolContext sc;
    start_address->CalculateSymbolContext(&sc);
    return SBThreadPlan(
        m_opaque_sp->GetThread().QueueThreadPlanForStepOverRange(
            false, range, sc, eAllThreads));
  } else {
    return SBThreadPlan();
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepInRange(SBAddress &sb_start_address,
                                            lldb::addr_t size) {
  if (m_opaque_sp) {
    Address *start_address = sb_start_address.get();
    if (!start_address) {
      return SBThreadPlan();
    }

    AddressRange range(*start_address, size);
    SymbolContext sc;
    start_address->CalculateSymbolContext(&sc);
    return SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepInRange(
        false, range, sc, NULL, eAllThreads));
  } else {
    return SBThreadPlan();
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForStepOut(uint32_t frame_idx_to_step_to,
                                        bool first_insn) {
  if (m_opaque_sp) {
    SymbolContext sc;
    sc = m_opaque_sp->GetThread().GetStackFrameAtIndex(0)->GetSymbolContext(
        lldb::eSymbolContextEverything);
    return SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForStepOut(
        false, &sc, first_insn, false, eVoteYes, eVoteNoOpinion,
        frame_idx_to_step_to));
  } else {
    return SBThreadPlan();
  }
}

SBThreadPlan
SBThreadPlan::QueueThreadPlanForRunToAddress(SBAddress sb_address) {
  if (m_opaque_sp) {
    Address *address = sb_address.get();
    if (!address)
      return SBThreadPlan();

    return SBThreadPlan(m_opaque_sp->GetThread().QueueThreadPlanForRunToAddress(
        false, *address, false));
  } else {
    return SBThreadPlan();
  }
}
