//===-- CommandObjectWatchpointCommand.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectWatchpointCommand_h_
#define liblldb_CommandObjectWatchpointCommand_h_



#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

// CommandObjectMultiwordWatchpoint

class CommandObjectWatchpointCommand : public CommandObjectMultiword {
public:
  CommandObjectWatchpointCommand(CommandInterpreter &interpreter);

  ~CommandObjectWatchpointCommand() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectWatchpointCommand_h_
