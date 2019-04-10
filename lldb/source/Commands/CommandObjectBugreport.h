//===-- CommandObjectBugreport.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectBugreport_h_
#define liblldb_CommandObjectBugreport_h_

#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectMultiwordBugreport

class CommandObjectMultiwordBugreport : public CommandObjectMultiword {
public:
  CommandObjectMultiwordBugreport(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordBugreport() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectBugreport_h_
