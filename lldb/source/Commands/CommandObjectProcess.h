//===-- CommandObjectProcess.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectProcess_h_
#define liblldb_CommandObjectProcess_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordProcess
//-------------------------------------------------------------------------

class CommandObjectMultiwordProcess : public CommandObjectMultiword {
public:
  CommandObjectMultiwordProcess(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordProcess() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectProcess_h_
