//===-- CommandObjectRegister.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectRegister_h_
#define liblldb_CommandObjectRegister_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectRegister
//-------------------------------------------------------------------------

class CommandObjectRegister : public CommandObjectMultiword {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommandObjectRegister(CommandInterpreter &interpreter);

  ~CommandObjectRegister() override;

private:
  //------------------------------------------------------------------
  // For CommandObjectRegister only
  //------------------------------------------------------------------
  DISALLOW_COPY_AND_ASSIGN(CommandObjectRegister);
};

} // namespace lldb_private

#endif // liblldb_CommandObjectRegister_h_
