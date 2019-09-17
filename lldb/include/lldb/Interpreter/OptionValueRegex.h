//===-- OptionValueRegex.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueRegex_h_
#define liblldb_OptionValueRegex_h_

#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Utility/RegularExpression.h"

namespace lldb_private {

class OptionValueRegex : public OptionValue {
public:
  OptionValueRegex(const char *value = nullptr)
      : OptionValue(), m_regex(llvm::StringRef::withNullAsEmpty(value)) {}

  ~OptionValueRegex() override = default;

  // Virtual subclass pure virtual overrides

  OptionValue::Type GetType() const override { return eTypeRegex; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;
  Status
  SetValueFromString(const char *,
                     VarSetOperationType = eVarSetOperationAssign) = delete;

  bool Clear() override {
    m_regex = RegularExpression();
    m_value_was_set = false;
    return true;
  }

  lldb::OptionValueSP DeepCopy() const override;

  // Subclass specific functions
  const RegularExpression *GetCurrentValue() const {
    return (m_regex.IsValid() ? &m_regex : nullptr);
  }

  void SetCurrentValue(const char *value) {
    if (value && value[0])
      m_regex = RegularExpression(llvm::StringRef(value));
    else
      m_regex = RegularExpression();
  }

  bool IsValid() const { return m_regex.IsValid(); }

protected:
  RegularExpression m_regex;
};

} // namespace lldb_private

#endif // liblldb_OptionValueRegex_h_
