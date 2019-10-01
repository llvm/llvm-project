//===-- CommandObjectTarget.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectTarget_h_
#define liblldb_CommandObjectTarget_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

// CommandObjectMultiwordTarget

class CommandObjectMultiwordTarget : public CommandObjectMultiword {
public:
  CommandObjectMultiwordTarget(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordTarget() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectTarget_h_
