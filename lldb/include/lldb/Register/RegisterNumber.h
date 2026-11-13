//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Register-number conversion and lookup through an immutable description.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERNUMBER_H
#define LLDB_REGISTER_REGISTERNUMBER_H

#include "lldb/Register/RegisterDescription.h"

namespace lldb_private {

/// A number is meaningful only with its numbering scheme. Conversion and name
/// lookup are queries of a description, never state retained from a Thread.
struct RegisterNumber {
  lldb::RegisterKind kind = lldb::kNumRegisterKinds;
  uint32_t number = LLDB_INVALID_REGNUM;

  RegisterNumber() = default;
  RegisterNumber(lldb::RegisterKind kind, uint32_t number)
      : kind(kind), number(number) {}

  /// Check the number and kind sentinels, without consulting target metadata. A
  /// structurally valid number can still be absent from a description.
  bool IsValid() const {
    return kind < lldb::kNumRegisterKinds && number != LLDB_INVALID_REGNUM;
  }

  /// Find the logical register in a description, or return nullptr if unmapped.
  const RegisterDescriptor *
  Resolve(const RegisterDescription &description) const {
    return description.FindByNumber(kind, number);
  }

  /// Convert to another numbering scheme, returning LLDB_INVALID_REGNUM if
  /// either the source number or the target mapping is unavailable.
  uint32_t GetAsKind(const RegisterDescription &description,
                     lldb::RegisterKind target) const {
    return description.ConvertNumber(kind, number, target);
  }

  /// Return a borrowed primary name, or an empty view if resolution fails.
  llvm::StringRef GetName(const RegisterDescription &description) const {
    const auto *reg = Resolve(description);
    return reg ? reg->name : llvm::StringRef();
  }

  /// Return true only when both numbers resolve to the same logical ID. Two
  /// unresolved numbers are not considered the same register.
  bool RefersToSameRegister(const RegisterDescription &description,
                            RegisterNumber other) const {
    const auto *reg = Resolve(description);
    const auto *rhs = other.Resolve(description);
    return reg && rhs && reg->id == rhs->id;
  }
};

} // namespace lldb_private
#endif
