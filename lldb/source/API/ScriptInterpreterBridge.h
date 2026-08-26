//===-- ScriptInterpreterBridge.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_API_SCRIPTINTERPRETERBRIDGE_H
#define LLDB_SOURCE_API_SCRIPTINTERPRETERBRIDGE_H

#include "lldb/API/SBDefines.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include <optional>

namespace lldb_private {

class CommandReturnObject;
class Event;

/// Unwraps the opaque internal object held by an SB (public API) instance,
/// for scripting-language plugins that need to convert values passed across
/// the script callback boundary back to their lldb_private form. Every SB
/// class this needs to reach into grants friendship to this class alone, so
/// that access to API internals stays confined to this single bridge rather
/// than spreading across the Interpreter layer.
class ScriptInterpreterBridge {
public:
  static lldb::DataExtractorSP GetDataExtractor(const lldb::SBData &data);

  static lldb::ThreadPlanSP GetThreadPlan(const lldb::SBThreadPlan &thread_plan);

  static Status GetStatus(const lldb::SBError &error);

  static Event *GetEvent(const lldb::SBEvent &event);

  static lldb::StreamSP GetStream(const lldb::SBStream &stream);

  static lldb::ThreadSP GetThread(const lldb::SBThread &thread);

  static lldb::StackFrameSP GetStackFrame(const lldb::SBFrame &frame);

  static SymbolContext GetSymbolContext(const lldb::SBSymbolContext &sym_ctx);

  static lldb::BreakpointSP GetBreakpoint(const lldb::SBBreakpoint &breakpoint);

  static lldb::BreakpointLocationSP
  GetBreakpointLocation(const lldb::SBBreakpointLocation &break_loc);

  static CommandReturnObject *
  GetCommandReturnObject(const lldb::SBCommandReturnObject &cmd_retobj);

  static lldb::DebuggerSP GetDebugger(const lldb::SBDebugger &debugger);

  static lldb::ProcessAttachInfoSP
  GetProcessAttachInfo(const lldb::SBAttachInfo &attach_info);

  static lldb::ProcessLaunchInfoSP
  GetProcessLaunchInfo(const lldb::SBLaunchInfo &launch_info);

  static std::optional<MemoryRegionInfo>
  GetMemoryRegionInfo(const lldb::SBMemoryRegionInfo &mem_region);

  static lldb::ExecutionContextRefSP
  GetExecutionContextRef(const lldb::SBExecutionContext &exe_ctx);

  static lldb::StackFrameListSP
  GetStackFrameList(const lldb::SBFrameList &frame_list);

  static lldb::ValueObjectSP GetValueObject(const lldb::SBValue &value);

  static lldb::TargetSP GetTarget(const lldb::SBTarget &target);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_API_SCRIPTINTERPRETERBRIDGE_H
