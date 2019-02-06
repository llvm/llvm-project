//===-- CommandObjectSettings.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSettings_h_
#define liblldb_CommandObjectSettings_h_

#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordSettings
//-------------------------------------------------------------------------

class CommandObjectMultiwordSettings : public CommandObjectMultiword {
public:
  CommandObjectMultiwordSettings(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordSettings() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectSettings_h_
