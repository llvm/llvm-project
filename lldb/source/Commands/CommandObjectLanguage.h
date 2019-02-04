//===-- CommandObjectLanguage.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectLanguage_h_
#define liblldb_CommandObjectLanguage_h_



#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/lldb-types.h"

namespace lldb_private {
class CommandObjectLanguage : public CommandObjectMultiword {
public:
  CommandObjectLanguage(CommandInterpreter &interpreter);

  ~CommandObjectLanguage() override;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result);
};
} // namespace lldb_private

#endif // liblldb_CommandObjectLanguage_h_
