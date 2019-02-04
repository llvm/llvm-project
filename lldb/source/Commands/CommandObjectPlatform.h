//===-- CommandObjectPlatform.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectPlatform_h_
#define liblldb_CommandObjectPlatform_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectPlatform
//-------------------------------------------------------------------------

class CommandObjectPlatform : public CommandObjectMultiword {
public:
  CommandObjectPlatform(CommandInterpreter &interpreter);

  ~CommandObjectPlatform() override;

private:
  DISALLOW_COPY_AND_ASSIGN(CommandObjectPlatform);
};

} // namespace lldb_private

#endif // liblldb_CommandObjectPlatform_h_
