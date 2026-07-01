//===-- OptionGroupDirection.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONGROUPDIRECTION_H
#define LLDB_INTERPRETER_OPTIONGROUPDIRECTION_H

#include "lldb/Interpreter/Options.h"

#include <optional>

namespace lldb_private {

// OptionGroupDirection

class OptionGroupDirection : public OptionGroup {
public:
  OptionGroupDirection();

  ~OptionGroupDirection() override = default;

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

  Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                        ExecutionContext *execution_context) override;

  void OptionParsingStarting(ExecutionContext *execution_context) override;

  std::optional<lldb::RunDirection> &GetDirection() { return m_direction; }

protected:
  std::optional<lldb::RunDirection> m_direction;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONGROUPDIRECTION_H
