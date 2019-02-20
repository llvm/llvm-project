//===-- CommandObjectBreakpointCommand.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectBreakpointCommand_h_
#define liblldb_CommandObjectBreakpointCommand_h_



#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------

class CommandObjectBreakpointCommand : public CommandObjectMultiword {
public:
  CommandObjectBreakpointCommand(CommandInterpreter &interpreter);

  ~CommandObjectBreakpointCommand() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectBreakpointCommand_h_
