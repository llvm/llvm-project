//===-- ValueObjectConstResultCast.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResultCast_h_
#define liblldb_ValueObjectConstResultCast_h_

#include "lldb/Core/ValueObjectCast.h"
#include "lldb/Core/ValueObjectConstResultImpl.h"
#include "lldb/Symbol/CompilerType.h" // for CompilerType
#include "lldb/Utility/ConstString.h" // for ConstString
#include "lldb/lldb-defines.h"        // for DISALLOW_COPY_AND_...
#include "lldb/lldb-forward.h"        // for ValueObjectSP
#include "lldb/lldb-types.h"          // for addr_t

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, int32_t

namespace lldb_private {
class DataExtractor;
}
namespace lldb_private {
class Error;
}
namespace lldb_private {
class ValueObject;
}

namespace lldb_private {

class ValueObjectConstResultCast : public ValueObjectCast {
public:
  ValueObjectConstResultCast(ValueObject &parent, const ConstString &name,
                             const CompilerType &cast_type,
                             lldb::addr_t live_address = LLDB_INVALID_ADDRESS);

  ~ValueObjectConstResultCast() override;

  lldb::ValueObjectSP Dereference(Error &error) override;

  ValueObject *CreateChildAtIndex(size_t idx, bool synthetic_array_member,
                                  int32_t synthetic_index) override;

  virtual CompilerType GetCompilerType() {
    return ValueObjectCast::GetCompilerType();
  }

  lldb::ValueObjectSP GetSyntheticChildAtOffset(
      uint32_t offset, const CompilerType &type, bool can_create,
      ConstString name_const_str = ConstString()) override;

  lldb::ValueObjectSP AddressOf(Error &error) override;

  size_t GetPointeeData(DataExtractor &data, uint32_t item_idx = 0,
                        uint32_t item_count = 1) override;

  lldb::ValueObjectSP Cast(const CompilerType &compiler_type) override;

protected:
  ValueObjectConstResultImpl m_impl;

private:
  friend class ValueObject;
  friend class ValueObjectConstResult;
  friend class ValueObjectConstResultImpl;

  DISALLOW_COPY_AND_ASSIGN(ValueObjectConstResultCast);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectConstResultCast_h_
