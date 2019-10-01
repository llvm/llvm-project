//===-- CommandObjectQuit.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectQuit_h_
#define liblldb_CommandObjectQuit_h_

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

// CommandObjectQuit

class CommandObjectQuit : public CommandObjectParsed {
public:
  CommandObjectQuit(CommandInterpreter &interpreter);

  ~CommandObjectQuit() override;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override;

  bool ShouldAskForConfirmation(bool &is_a_detach);
};

} // namespace lldb_private

#endif // liblldb_CommandObjectQuit_h_
