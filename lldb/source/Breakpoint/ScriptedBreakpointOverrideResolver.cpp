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

namespace lldb_private {
lldb::BreakpointResolverSP ScriptedBreakpointResolverOverride::CheckForOverride(
    Target &target, lldb::BreakpointResolverSP initial_sp) {
  lldb::BreakpointResolverSP candidate_sp(new BreakpointResolverScripted(
      {}, m_class_name, initial_sp->GetDepth(), m_args_data));
  if (candidate_sp->OverridesResolver(target, initial_sp))
    return candidate_sp;
  return {};
}

llvm::Error ScriptedBreakpointResolverOverride::Validate() {
  // FIXME: we should make sure the module and class exist, though that will
  // to happen in a scripting language specific function.
  return llvm::Error::success();
}
} // namespace lldb_private
