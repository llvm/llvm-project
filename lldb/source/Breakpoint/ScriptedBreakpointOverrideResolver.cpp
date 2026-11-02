//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/ScriptedBreakpointOverrideResolver.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointResolverScripted.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "llvm/Support/ErrorExtras.h"

namespace lldb_private {
lldb::BreakpointResolverSP ScriptedBreakpointResolverOverride::CheckForOverride(
    Target &target, lldb::BreakpointResolverSP initial_sp) {
  lldb::BreakpointResolverSP candidate_sp(new BreakpointResolverScripted(
      {}, m_class_name, initial_sp->GetDepth(), m_args_data));
  if (candidate_sp->OverridesResolver(target, initial_sp))
    return candidate_sp;
  return {};
}

llvm::Error ScriptedBreakpointResolverOverride::HasErrors() {
  ScriptInterpreter *interpreter =
      m_target.GetDebugger().GetScriptInterpreter();
  if (!interpreter || !interpreter->CheckObjectExists(m_class_name.c_str()))
    return llvm::createStringErrorV(
        "class for breakpoint override: \"{0}\" does not exist", m_class_name);
  return llvm::Error::success();
}
} // namespace lldb_private
