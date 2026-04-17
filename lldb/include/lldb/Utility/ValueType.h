//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_VALUETYPE_H
#define LLDB_UTILITY_VALUETYPE_H

#include "lldb/lldb-enumerations.h"

namespace lldb_private {
/// Get the base value type - for when we don't care if the value is synthetic
/// or not, or when we've already handled that case.
constexpr lldb::ValueType GetBaseValueType(lldb::ValueType vt) {
  return lldb::ValueType(vt & ~lldb::ValueTypeSyntheticMask);
}

/// Given a base value type, return a version that carries the synthetic bit.
constexpr lldb::ValueType GetSyntheticValueType(lldb::ValueType base) {
  return lldb::ValueType(base | lldb::ValueTypeSyntheticMask);
}

/// Return true if vt represents a synthetic value, false if not.
constexpr bool IsSyntheticValueType(lldb::ValueType vt) {
  return vt & lldb::ValueTypeSyntheticMask;
}
} // namespace lldb_private

#endif // LLDB_UTILITY_VALUETYPE_H
