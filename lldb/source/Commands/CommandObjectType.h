//===-- CommandObjectType.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectType_h_
#define liblldb_CommandObjectType_h_



#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class CommandObjectType : public CommandObjectMultiword {
public:
  CommandObjectType(CommandInterpreter &interpreter);

  ~CommandObjectType() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectType_h_
