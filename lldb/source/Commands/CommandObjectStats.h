//===-- CommandObjectStats.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectStats_h_
#define liblldb_CommandObjectStats_h_

#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {
class CommandObjectStats : public CommandObjectMultiword {
public:
  CommandObjectStats(CommandInterpreter &interpreter);

  ~CommandObjectStats() override;
};
} // namespace lldb_private

#endif // liblldb_CommandObjectLanguage_h_
