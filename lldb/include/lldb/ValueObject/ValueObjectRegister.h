//===-- ValueObjectRegister.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_VALUEOBJECTREGISTER_H
#define LLDB_VALUEOBJECT_VALUEOBJECTREGISTER_H

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-types.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace lldb_private {
class DataExtractor;
class Status;
class ExecutionContextScope;
class Scalar;
class Stream;

class ValueObjectRegisterSet : public ValueObject {
public:
  ~ValueObjectRegisterSet() override;

  static lldb::ValueObjectSP Create(ExecutionContextScope *exe_scope,
                                    lldb::ThreadRegisterContextSP &reg_ctx_sp,
                                    uint32_t set_idx);

  llvm::Expected<uint64_t> GetByteSize() override;

  lldb::ValueType GetValueType() const override {
    return lldb::eValueTypeRegisterSet;
  }

  ConstString GetTypeName() override;

  ConstString GetQualifiedTypeName() override;

  llvm::Expected<uint32_t> CalculateNumChildren(uint32_t max) override;

  lldb::ValueObjectSP GetChildMemberWithName(llvm::StringRef name,
                                             bool can_create = true) override;

  llvm::Expected<size_t> GetIndexOfChildWithName(llvm::StringRef name) override;

protected:
  bool UpdateValue() override;

  CompilerType GetCompilerTypeImpl() override;

  lldb::ThreadRegisterContextSP m_reg_ctx_sp;
  std::shared_ptr<const RegisterDescription> m_description;
  const RegisterGroup *m_reg_set;
  uint32_t m_reg_set_idx;

private:
  friend class ValueObjectRegisterContext;

  ValueObjectRegisterSet(ExecutionContextScope *exe_scope,
                         ValueObjectManager &manager,
                         lldb::ThreadRegisterContextSP &reg_ctx_sp, uint32_t set_idx);

  ValueObject *CreateChildAtIndex(size_t idx) override;
  ValueObject *CreateSyntheticArrayMember(size_t idx) override {
    return nullptr;
  }

  std::optional<std::pair<size_t, const RegisterDescriptor *>>
  LookupChildWithName(llvm::StringRef name);

  static bool IsSameRegister(const RegisterDescriptor *a,
                             const RegisterDescriptor *b);

  // For ValueObject only
  ValueObjectRegisterSet(const ValueObjectRegisterSet &) = delete;
  const ValueObjectRegisterSet &
  operator=(const ValueObjectRegisterSet &) = delete;
};

class ValueObjectRegister : public ValueObject {
public:
  ~ValueObjectRegister() override;

  static lldb::ValueObjectSP Create(ExecutionContextScope *exe_scope,
                                    lldb::ThreadRegisterContextSP &reg_ctx_sp,
                                    const RegisterDescriptor *reg_info);

  llvm ::Expected<uint64_t> GetByteSize() override;

  lldb::ValueType GetValueType() const override {
    return lldb::eValueTypeRegister;
  }

  ConstString GetTypeName() override;

  llvm::Expected<uint32_t> CalculateNumChildren(uint32_t max) override;

  bool SetValueFromCString(const char *value_str, Status &error) override;

  bool SetData(DataExtractor &data, Status &error) override;

  bool ResolveValue(Scalar &scalar) override;

  void
  GetExpressionPath(Stream &s,
                    GetExpressionPathFormat epformat =
                        eGetExpressionPathFormatDereferencePointers) override;

protected:
  bool UpdateValue() override;

  CompilerType GetCompilerTypeImpl() override;

  lldb::ThreadRegisterContextSP m_reg_ctx_sp;
  std::shared_ptr<const RegisterDescription> m_description;
  RegisterID m_register_id = LLDB_INVALID_REGNUM;
  const RegisterDescriptor &GetDescriptor() const {
    return *m_description->Find(m_register_id);
  }
  RegisterValue m_reg_value;
  ConstString m_type_name;
  CompilerType m_compiler_type;

private:
  void ConstructObject(const RegisterDescriptor *reg_info);

  friend class ValueObjectRegisterSet;

  ValueObjectRegister(ValueObject &parent,
                      lldb::ThreadRegisterContextSP &reg_ctx_sp,
                      const RegisterDescriptor *reg_info);
  ValueObjectRegister(ExecutionContextScope *exe_scope,
                      ValueObjectManager &manager,
                      lldb::ThreadRegisterContextSP &reg_ctx_sp,
                      const RegisterDescriptor *reg_info);

  // For ValueObject only
  ValueObjectRegister(const ValueObjectRegister &) = delete;
  const ValueObjectRegister &operator=(const ValueObjectRegister &) = delete;
};

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_VALUEOBJECTREGISTER_H
