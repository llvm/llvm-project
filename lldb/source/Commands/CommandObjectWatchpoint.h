//===-- CommandObjectWatchpoint.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectWatchpoint_h_
#define liblldb_CommandObjectWatchpoint_h_


#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/OptionGroupWatchpoint.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

// CommandObjectMultiwordWatchpoint

class CommandObjectMultiwordWatchpoint : public CommandObjectMultiword {
public:
  CommandObjectMultiwordWatchpoint(CommandInterpreter &interpreter);

  ~CommandObjectMultiwordWatchpoint() override;

  static bool VerifyWatchpointIDs(Target *target, Args &args,
                                  std::vector<uint32_t> &wp_ids);
};

} // namespace lldb_private

#endif // liblldb_CommandObjectWatchpoint_h_
