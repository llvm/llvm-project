//===-- CommandObjectCommands.h -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectCommands_h_
#define liblldb_CommandObjectCommands_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectMultiwordCommands

class CommandObjectMultiwordCommands : public CommandObjectMultiword {
public:
  CommandObjectMultiwordCommands(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordCommands() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectCommands_h_
