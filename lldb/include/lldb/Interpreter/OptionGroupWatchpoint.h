//===-- OptionGroupWatchpoint.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONGROUPWATCHPOINT_H
#define LLDB_INTERPRETER_OPTIONGROUPWATCHPOINT_H

#include "lldb/Interpreter/Options.h"

namespace lldb_private {

// OptionGroupWatchpoint

class OptionGroupWatchpoint : public OptionGroup {
public:
  OptionGroupWatchpoint();

  ~OptionGroupWatchpoint() override;

  static bool IsWatchSizeSupported(uint32_t watch_size);

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

  Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_value,
                        ExecutionContext *execution_context) override;

  void OptionParsingStarting(ExecutionContext *execution_context) override;

  // Note:
  // eWatchRead == LLDB_WATCH_TYPE_READ; and
  // eWatchWrite == LLDB_WATCH_TYPE_WRITE
  enum WatchType {
    eWatchInvalid = 0,
    eWatchRead,
    eWatchWrite,
    eWatchReadWrite
  };

  WatchType watch_type;
  uint32_t watch_size;
  bool watch_type_specified;

private:
  OptionGroupWatchpoint(const OptionGroupWatchpoint &) = delete;
  const OptionGroupWatchpoint &
  operator=(const OptionGroupWatchpoint &) = delete;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONGROUPWATCHPOINT_H
