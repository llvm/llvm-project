//===-- CommandObjectLog.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTLOG_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTLOG_H

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectLog

class CommandObjectLog : public CommandObjectMultiword {
public:
  // Constructors and Destructors
  CommandObjectLog(CommandInterpreter &interpreter);

  ~CommandObjectLog() override;

private:
  // For CommandObjectLog only
  DISALLOW_COPY_AND_ASSIGN(CommandObjectLog);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTLOG_H
