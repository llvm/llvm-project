//===-- SBExecutionContext.cpp ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBExecutionContext.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

#include "lldb/Target/ExecutionContext.h"

using namespace lldb;
using namespace lldb_private;

SBExecutionContext::SBExecutionContext() : m_exe_ctx_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBExecutionContext);
}

SBExecutionContext::SBExecutionContext(const lldb::SBExecutionContext &rhs)
    : m_exe_ctx_sp(rhs.m_exe_ctx_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext,
                          (const lldb::SBExecutionContext &), rhs);
}

SBExecutionContext::SBExecutionContext(
    lldb::ExecutionContextRefSP exe_ctx_ref_sp)
    : m_exe_ctx_sp(exe_ctx_ref_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext, (lldb::ExecutionContextRefSP),
                          exe_ctx_ref_sp);
}

SBExecutionContext::SBExecutionContext(const lldb::SBTarget &target)
    : m_exe_ctx_sp(new ExecutionContextRef()) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext, (const lldb::SBTarget &), target);

  m_exe_ctx_sp->SetTargetSP(target.GetSP());
}

SBExecutionContext::SBExecutionContext(const lldb::SBProcess &process)
    : m_exe_ctx_sp(new ExecutionContextRef()) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext, (const lldb::SBProcess &),
                          process);

  m_exe_ctx_sp->SetProcessSP(process.GetSP());
}

SBExecutionContext::SBExecutionContext(lldb::SBThread thread)
    : m_exe_ctx_sp(new ExecutionContextRef()) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext, (lldb::SBThread), thread);

  m_exe_ctx_sp->SetThreadPtr(thread.get());
}

SBExecutionContext::SBExecutionContext(const lldb::SBFrame &frame)
    : m_exe_ctx_sp(new ExecutionContextRef()) {
  LLDB_RECORD_CONSTRUCTOR(SBExecutionContext, (const lldb::SBFrame &), frame);

  m_exe_ctx_sp->SetFrameSP(frame.GetFrameSP());
}

SBExecutionContext::~SBExecutionContext() {}

const SBExecutionContext &SBExecutionContext::
operator=(const lldb::SBExecutionContext &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBExecutionContext &,
      SBExecutionContext, operator=,(const lldb::SBExecutionContext &), rhs);

  m_exe_ctx_sp = rhs.m_exe_ctx_sp;
  return LLDB_RECORD_RESULT(*this);
}

ExecutionContextRef *SBExecutionContext::get() const {
  return m_exe_ctx_sp.get();
}

SBTarget SBExecutionContext::GetTarget() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBTarget, SBExecutionContext,
                                   GetTarget);

  SBTarget sb_target;
  if (m_exe_ctx_sp) {
    TargetSP target_sp(m_exe_ctx_sp->GetTargetSP());
    if (target_sp)
      sb_target.SetSP(target_sp);
  }
  return LLDB_RECORD_RESULT(sb_target);
}

SBProcess SBExecutionContext::GetProcess() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBProcess, SBExecutionContext,
                                   GetProcess);

  SBProcess sb_process;
  if (m_exe_ctx_sp) {
    ProcessSP process_sp(m_exe_ctx_sp->GetProcessSP());
    if (process_sp)
      sb_process.SetSP(process_sp);
  }
  return LLDB_RECORD_RESULT(sb_process);
}

SBThread SBExecutionContext::GetThread() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBThread, SBExecutionContext,
                                   GetThread);

  SBThread sb_thread;
  if (m_exe_ctx_sp) {
    ThreadSP thread_sp(m_exe_ctx_sp->GetThreadSP());
    if (thread_sp)
      sb_thread.SetThread(thread_sp);
  }
  return LLDB_RECORD_RESULT(sb_thread);
}

SBFrame SBExecutionContext::GetFrame() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFrame, SBExecutionContext, GetFrame);

  SBFrame sb_frame;
  if (m_exe_ctx_sp) {
    StackFrameSP frame_sp(m_exe_ctx_sp->GetFrameSP());
    if (frame_sp)
      sb_frame.SetFrameSP(frame_sp);
  }
  return LLDB_RECORD_RESULT(sb_frame);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBExecutionContext>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, ());
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext,
                            (const lldb::SBExecutionContext &));
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext,
                            (lldb::ExecutionContextRefSP));
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBTarget &));
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBProcess &));
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (lldb::SBThread));
  LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBFrame &));
  LLDB_REGISTER_METHOD(
      const lldb::SBExecutionContext &,
      SBExecutionContext, operator=,(const lldb::SBExecutionContext &));
  LLDB_REGISTER_METHOD_CONST(lldb::SBTarget, SBExecutionContext, GetTarget,
                             ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBProcess, SBExecutionContext, GetProcess,
                             ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBExecutionContext, GetThread,
                             ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBFrame, SBExecutionContext, GetFrame, ());
}

}
}
