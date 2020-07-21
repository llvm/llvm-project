//===-- OptionValueUInt64.h --------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONVALUEUINT64_H
#define LLDB_INTERPRETER_OPTIONVALUEUINT64_H

#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueUInt64 : public OptionValue {
public:
  OptionValueUInt64() : OptionValue(), m_current_value(0), m_default_value(0) {}

  OptionValueUInt64(uint64_t value)
      : OptionValue(), m_current_value(value), m_default_value(value) {}

  OptionValueUInt64(uint64_t current_value, uint64_t default_value)
      : OptionValue(), m_current_value(current_value),
        m_default_value(default_value) {}

  ~OptionValueUInt64() override {}

  // Decode a uint64_t from "value_cstr" return a OptionValueUInt64 object
  // inside of a lldb::OptionValueSP object if all goes well. If the string
  // isn't a uint64_t value or any other error occurs, return an empty
  // lldb::OptionValueSP and fill error in with the correct stuff.
  static lldb::OptionValueSP Create(const char *, Status &) = delete;
  static lldb::OptionValueSP Create(llvm::StringRef value_str, Status &error);
  // Virtual subclass pure virtual overrides

  OptionValue::Type GetType() const override { return eTypeUInt64; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;
  Status
  SetValueFromString(const char *,
                     VarSetOperationType = eVarSetOperationAssign) = delete;

  void Clear() override {
    m_current_value = m_default_value;
    m_value_was_set = false;
  }

  lldb::OptionValueSP DeepCopy() const override;

  // Subclass specific functions

  const uint64_t &operator=(uint64_t value) {
    m_current_value = value;
    return m_current_value;
  }

  operator uint64_t() const { return m_current_value; }

  uint64_t GetCurrentValue() const { return m_current_value; }

  uint64_t GetDefaultValue() const { return m_default_value; }

  void SetCurrentValue(uint64_t value) { m_current_value = value; }

  void SetDefaultValue(uint64_t value) { m_default_value = value; }

protected:
  uint64_t m_current_value;
  uint64_t m_default_value;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONVALUEUINT64_H
