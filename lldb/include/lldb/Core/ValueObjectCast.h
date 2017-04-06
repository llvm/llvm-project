//===-- ValueObjectDynamicValue.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectCast_h_
#define liblldb_ValueObjectCast_h_

#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/CompilerType.h" // for CompilerType
#include "lldb/lldb-defines.h"        // for DISALLOW_COPY_AND_ASSIGN
#include "lldb/lldb-enumerations.h"   // for ValueType
#include "lldb/lldb-forward.h"        // for ValueObjectSP

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint64_t

namespace lldb_private {
class ConstString;
}

namespace lldb_private {

//---------------------------------------------------------------------------------
// A ValueObject that represents a given value represented as a different type.
//---------------------------------------------------------------------------------
class ValueObjectCast : public ValueObject {
public:
  ~ValueObjectCast() override;

  static lldb::ValueObjectSP Create(ValueObject &parent,
                                    const ConstString &name,
                                    const CompilerType &cast_type);

  uint64_t GetByteSize() override;

  size_t CalculateNumChildren(uint32_t max) override;

  lldb::ValueType GetValueType() const override;

  bool IsInScope() override;

  ValueObject *GetParent() override {
    return ((m_parent != nullptr) ? m_parent->GetParent() : nullptr);
  }

  const ValueObject *GetParent() const override {
    return ((m_parent != nullptr) ? m_parent->GetParent() : nullptr);
  }

protected:
  ValueObjectCast(ValueObject &parent, const ConstString &name,
                  const CompilerType &cast_type);

  bool UpdateValue() override;

  CompilerType GetCompilerTypeImpl() override;

  CompilerType m_cast_type;

private:
  DISALLOW_COPY_AND_ASSIGN(ValueObjectCast);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectCast_h_
