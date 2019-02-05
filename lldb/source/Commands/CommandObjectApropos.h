//===-- CommandObjectApropos.h -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectApropos_h_
#define liblldb_CommandObjectApropos_h_

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectApropos
//-------------------------------------------------------------------------

class CommandObjectApropos : public CommandObjectParsed {
public:
  CommandObjectApropos(CommandInterpreter &interpreter);

  ~CommandObjectApropos() override;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectApropos_h_
