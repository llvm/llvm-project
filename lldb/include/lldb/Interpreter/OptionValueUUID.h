//===-- OptionValueUUID.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONVALUEUUID_H
#define LLDB_INTERPRETER_OPTIONVALUEUUID_H

#include "lldb/Utility/UUID.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueUUID : public OptionValue {
public:
  OptionValueUUID() : OptionValue(), m_uuid() {}

  OptionValueUUID(const UUID &uuid) : OptionValue(), m_uuid(uuid) {}

  ~OptionValueUUID() override {}

  // Virtual subclass pure virtual overrides

  OptionValue::Type GetType() const override { return eTypeUUID; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;
  Status
  SetValueFromString(const char *,
                     VarSetOperationType = eVarSetOperationAssign) = delete;

  void Clear() override {
    m_uuid.Clear();
    m_value_was_set = false;
  }

  lldb::OptionValueSP DeepCopy() const override;

  // Subclass specific functions

  UUID &GetCurrentValue() { return m_uuid; }

  const UUID &GetCurrentValue() const { return m_uuid; }

  void SetCurrentValue(const UUID &value) { m_uuid = value; }

  void AutoComplete(CommandInterpreter &interpreter,
                    CompletionRequest &request) override;

protected:
  UUID m_uuid;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONVALUEUUID_H
