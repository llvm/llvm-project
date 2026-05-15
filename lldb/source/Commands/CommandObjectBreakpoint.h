//===-- CommandObjectBreakpoint.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTBREAKPOINT_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTBREAKPOINT_H

#include "lldb/Breakpoint/BreakpointName.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectMultiwordBreakpoint

class CommandObjectMultiwordBreakpoint : public CommandObjectMultiword {
public:
  CommandObjectMultiwordBreakpoint(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordBreakpoint() override;

  static void VerifyBreakpointOrLocationIDs(
      Args &args, const ExecutionContext &exe_ctx, CommandReturnObject &result,
      BreakpointIDList *valid_ids,
      BreakpointName::Permissions ::PermissionKinds purpose) {
    VerifyIDs(args, exe_ctx, true, result, valid_ids, purpose);
  }

  static void
  VerifyBreakpointIDs(Args &args, const ExecutionContext &exe_ctx,
                      CommandReturnObject &result, BreakpointIDList *valid_ids,
                      BreakpointName::Permissions::PermissionKinds purpose) {
    VerifyIDs(args, exe_ctx, false, result, valid_ids, purpose);
  }

private:
  static void VerifyIDs(Args &args, const ExecutionContext &exe_ctx,
                        bool allow_locations, CommandReturnObject &result,
                        BreakpointIDList *valid_ids,
                        BreakpointName::Permissions::PermissionKinds purpose);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTBREAKPOINT_H
