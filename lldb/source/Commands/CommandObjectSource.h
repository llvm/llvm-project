//===-- CommandObjectSource.h.h -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSource_h_
#define liblldb_CommandObjectSource_h_

#include "lldb/Core/STLUtils.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

// CommandObjectMultiwordSource

class CommandObjectMultiwordSource : public CommandObjectMultiword {
public:
  CommandObjectMultiwordSource(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordSource() override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectSource_h_
