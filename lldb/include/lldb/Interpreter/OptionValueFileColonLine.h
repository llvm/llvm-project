//===-- OptionValueFileColonLine.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONVALUEFILECOLONLINE_H
#define LLDB_INTERPRETER_OPTIONVALUEFILECOLONLINE_H

#include "lldb/Interpreter/OptionValue.h"

#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/Chrono.h"

namespace lldb_private {

class OptionValueFileColonLine : public OptionValue {
public:
  OptionValueFileColonLine();
  OptionValueFileColonLine(const llvm::StringRef input);

  ~OptionValueFileColonLine() override {}

  OptionValue::Type GetType() const override { return eTypeFileLineColumn; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;
  Status
  SetValueFromString(const char *,
                     VarSetOperationType = eVarSetOperationAssign) = delete;

  bool Clear() override {
    m_file_spec.Clear();
    m_line_number = LLDB_INVALID_LINE_NUMBER;
    m_column_number = LLDB_INVALID_COLUMN_NUMBER;
    return true;
  }

  lldb::OptionValueSP DeepCopy() const override;

  void AutoComplete(CommandInterpreter &interpreter,
                    CompletionRequest &request) override;

  FileSpec &GetFileSpec() { return m_file_spec; }
  uint32_t GetLineNumber() { return m_line_number; }
  uint32_t GetColumnNumber() { return m_column_number; }

  void SetCompletionMask(uint32_t mask) { m_completion_mask = mask; }

protected:
  FileSpec m_file_spec;
  uint32_t m_line_number;
  uint32_t m_column_number;
  uint32_t m_completion_mask;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONVALUEFILECOLONLINE_H
