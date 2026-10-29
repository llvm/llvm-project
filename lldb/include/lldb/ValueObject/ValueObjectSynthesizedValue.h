//===-- ValueObjectSynthesizedValue.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_VALUEOBJECTSYNTHESIZEDVALUE_H
#define LLDB_VALUEOBJECT_VALUEOBJECTSYNTHESIZEDVALUE_H

#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

/// \class ValueObjectSynthesizedValue
///
/// Presents another ValueObject under the given ValueType.
///
/// A ValueObject reports the ValueType that follows from how it was produced,
/// which is not always what it is being presented as: a frame that synthesizes
/// variables, or a frame recognizer naming an argument, knows a ValueType the
/// ValueObject cannot report for itself. This ValueObject subclass reports that
/// ValueType and forwards everything else to the ValueObject it presents.
///
/// The presented ValueObject is not a parent. It is the same ValueObject seen
/// differently, so this is a root and reports no parent.
class ValueObjectSynthesizedValue : public ValueObject {
public:
  static lldb::ValueObjectSP Create(ValueObject &valobj, lldb::ValueType type);

  ~ValueObjectSynthesizedValue() override;

  lldb::ValueType GetValueType() const override { return m_type; }

  llvm::Expected<uint64_t> GetByteSize() override;

  llvm::Expected<uint32_t> CalculateNumChildren(uint32_t max) override;

  bool IsInScope() override;

protected:
  ValueObjectSynthesizedValue(ExecutionContextScope *exe_scope,
                              ValueObjectManager &manager,
                              const lldb::ValueObjectSP &valobj_sp,
                              lldb::ValueType type);

  bool UpdateValue() override;

  CompilerType GetCompilerTypeImpl() override;

  lldb::ValueObjectSP m_valobj_sp;
  lldb::ValueType m_type;

private:
  ValueObjectSynthesizedValue(const ValueObjectSynthesizedValue &) = delete;
  const ValueObjectSynthesizedValue &
  operator=(const ValueObjectSynthesizedValue &) = delete;
};

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_VALUEOBJECTSYNTHESIZEDVALUE_H
