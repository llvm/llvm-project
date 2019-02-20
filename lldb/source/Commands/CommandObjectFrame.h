//===-- CommandObjectFrame.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectFrame_h_
#define liblldb_CommandObjectFrame_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordFrame
//-------------------------------------------------------------------------

class CommandObjectMultiwordFrame : public CommandObjectMultiword {
public:
  CommandObjectMultiwordFrame(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordFrame() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectFrame_h_
