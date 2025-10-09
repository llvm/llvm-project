//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDBREAKPOINTINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDBREAKPOINTINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
class ScriptedBreakpointInterface : public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::BreakpointSP break_sp,
                     const StructuredDataImpl &args_sp) = 0;

  /// "ResolverCallback" will get called when a new module is loaded.  The
  /// new module information is passed in sym_ctx.  The Resolver will add
  /// any breakpoint locations it found in that module.
  virtual bool ResolverCallback(SymbolContext sym_ctx) { return true; }
  virtual lldb::SearchDepth GetDepth() { return lldb::eSearchDepthModule; }
  virtual std::optional<std::string> GetShortHelp() { return nullptr; }
  /// WasHit returns the breakpoint location SP for the location that was "hit".
  virtual lldb::BreakpointLocationSP
  WasHit(lldb::StackFrameSP frame_sp, lldb::BreakpointLocationSP bp_loc_sp) {
    return LLDB_INVALID_BREAK_ID;
  }
  virtual std::optional<std::string>
  GetLocationDescription(lldb::BreakpointLocationSP bp_loc_sp,
                         lldb::DescriptionLevel level) {
    return {};
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTOPHOOKINTERFACE_H
