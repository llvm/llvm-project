//===-- CommandObjectGUI.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectGUI_h_
#define liblldb_CommandObjectGUI_h_

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

// CommandObjectGUI

class CommandObjectGUI : public CommandObjectParsed {
public:
  CommandObjectGUI(CommandInterpreter &interpreter);

  ~CommandObjectGUI() override;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectGUI_h_
