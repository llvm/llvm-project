//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterBridge.h"
#include "API/SBCommandReturnObjectImpl.h"
#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBExecutionContext.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFrameList.h"
#include "lldb/API/SBLaunchInfo.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBThreadPlan.h"
#include "lldb/API/SBValue.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;

lldb::DataExtractorSP
ScriptInterpreterBridge::GetDataExtractor(const lldb::SBData &data) {
  return data.m_opaque_sp;
}

lldb::ThreadPlanSP
ScriptInterpreterBridge::GetThreadPlan(const lldb::SBThreadPlan &thread_plan) {
  return thread_plan.GetSP();
}

lldb::BreakpointSP
ScriptInterpreterBridge::GetBreakpoint(const lldb::SBBreakpoint &breakpoint) {
  return breakpoint.m_opaque_wp.lock();
}

lldb::BreakpointLocationSP ScriptInterpreterBridge::GetBreakpointLocation(
    const lldb::SBBreakpointLocation &break_loc) {
  return break_loc.m_opaque_wp.lock();
}

CommandReturnObject *ScriptInterpreterBridge::GetCommandReturnObject(
    const lldb::SBCommandReturnObject &cmd_retobj) {
  return cmd_retobj.m_opaque_up->get();
}

lldb::DebuggerSP
ScriptInterpreterBridge::GetDebugger(const lldb::SBDebugger &debugger) {
  return debugger.m_opaque_sp;
}

lldb::ProcessAttachInfoSP ScriptInterpreterBridge::GetProcessAttachInfo(
    const lldb::SBAttachInfo &attach_info) {
  return attach_info.m_opaque_sp;
}

lldb::ProcessLaunchInfoSP ScriptInterpreterBridge::GetProcessLaunchInfo(
    const lldb::SBLaunchInfo &launch_info) {
  return std::make_shared<ProcessLaunchInfo>(
      *reinterpret_cast<ProcessLaunchInfo *>(launch_info.m_opaque_sp.get()));
}

Status ScriptInterpreterBridge::GetStatus(const lldb::SBError &error) {
  if (error.m_opaque_up)
    return error.m_opaque_up->Clone();

  return Status();
}

lldb::ThreadSP
ScriptInterpreterBridge::GetThread(const lldb::SBThread &thread) {
  if (thread.m_opaque_sp)
    return thread.m_opaque_sp->GetThreadSP();
  return nullptr;
}

lldb::StackFrameSP
ScriptInterpreterBridge::GetStackFrame(const lldb::SBFrame &frame) {
  if (frame.m_opaque_sp)
    return frame.m_opaque_sp->GetFrameSP();
  return nullptr;
}

Event *ScriptInterpreterBridge::GetEvent(const lldb::SBEvent &event) {
  return event.m_opaque_ptr;
}

lldb::StreamSP
ScriptInterpreterBridge::GetStream(const lldb::SBStream &stream) {
  if (stream.m_opaque_up) {
    lldb::StreamSP s = std::make_shared<lldb_private::StreamString>();
    *s << reinterpret_cast<StreamString *>(stream.m_opaque_up.get())->m_packet;
    return s;
  }

  return nullptr;
}

SymbolContext ScriptInterpreterBridge::GetSymbolContext(
    const lldb::SBSymbolContext &sb_sym_ctx) {
  if (sb_sym_ctx.m_opaque_up)
    return *sb_sym_ctx.m_opaque_up;
  return {};
}

std::optional<lldb_private::MemoryRegionInfo>
ScriptInterpreterBridge::GetMemoryRegionInfo(
    const lldb::SBMemoryRegionInfo &mem_region) {
  if (!mem_region.m_opaque_up)
    return std::nullopt;
  return *mem_region.m_opaque_up.get();
}

lldb::ExecutionContextRefSP ScriptInterpreterBridge::GetExecutionContextRef(
    const lldb::SBExecutionContext &exe_ctx) {
  return exe_ctx.m_exe_ctx_sp;
}

lldb::StackFrameListSP ScriptInterpreterBridge::GetStackFrameList(
    const lldb::SBFrameList &frame_list) {
  return frame_list.m_opaque_sp;
}

lldb::TargetSP
ScriptInterpreterBridge::GetTarget(const lldb::SBTarget &target) {
  return target.m_opaque_sp;
}

lldb::ValueObjectSP
ScriptInterpreterBridge::GetValueObject(const lldb::SBValue &value) {
  if (!value.m_opaque_sp)
    return lldb::ValueObjectSP();

  lldb_private::ValueLocker locker;
  return locker.GetLockedSP(*value.m_opaque_sp);
}
