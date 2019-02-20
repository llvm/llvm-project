//===-- OptionValueFileSpecList.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueFileSpecList_h_
#define liblldb_OptionValueFileSpecList_h_

#include "lldb/Core/FileSpecList.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueFileSpecList : public OptionValue {
public:
  OptionValueFileSpecList() : OptionValue(), m_current_value() {}

  OptionValueFileSpecList(const FileSpecList &current_value)
      : OptionValue(), m_current_value(current_value) {}

  ~OptionValueFileSpecList() override {}

  //---------------------------------------------------------------------
  // Virtual subclass pure virtual overrides
  //---------------------------------------------------------------------

  OptionValue::Type GetType() const override { return eTypeFileSpecList; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;
  Status
  SetValueFromString(const char *,
                     VarSetOperationType = eVarSetOperationAssign) = delete;

  bool Clear() override {
    m_current_value.Clear();
    m_value_was_set = false;
    return true;
  }

  lldb::OptionValueSP DeepCopy() const override;

  bool IsAggregateValue() const override { return true; }

  //---------------------------------------------------------------------
  // Subclass specific functions
  //---------------------------------------------------------------------

  FileSpecList &GetCurrentValue() { return m_current_value; }

  const FileSpecList &GetCurrentValue() const { return m_current_value; }

  void SetCurrentValue(const FileSpecList &value) { m_current_value = value; }

protected:
  FileSpecList m_current_value;
};

} // namespace lldb_private

#endif // liblldb_OptionValueFileSpecList_h_
