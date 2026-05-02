//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Interfaces/ScriptedBreakpointInterface.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointResolverScripted.h"

namespace lldb_private {
lldb::BreakpointResolverSP ScriptedBreakpointResolverOverride::CheckForOverride(
    lldb::BreakpointResolverSP initial_sp) {
  lldb::BreakpointResolverSP candidate_sp(
      new BreakpointResolverScripted(initial_sp->GetBreakpoint(), m_class_name,
                                     initial_sp->GetDepth(), m_args_data));
  if (candidate_sp->OverridesResolver(initial_sp))
    return candidate_sp;
  return {};
}
} // namespace lldb_private
