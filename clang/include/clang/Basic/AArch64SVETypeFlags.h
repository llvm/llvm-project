//===- AArch64SVETypeFlags.h - Flags used to generate ACLE builtins- C++ -*-===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_AARCH64SVETYPEFLAGS_H
#define LLVM_CLANG_BASIC_AARCH64SVETYPEFLAGS_H

#include <stdint.h>

namespace clang {

/// Flags to identify the types for overloaded SVE builtins.
class SVETypeFlags {
  uint64_t Flags;

public:
  /// These must be kept in sync with the flags in
  /// include/clang/Basic/arm_sve.td.
  static const uint64_t MemEltTypeOffset = 4; // Bit offset of MemEltTypeMask
  static const uint64_t EltTypeMask      = 0x00000000000f;
  static const uint64_t MemEltTypeMask   = 0x000000000070;
  static const uint64_t IsLoad           = 0x000000000080;

  enum EltType {
    Invalid,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bool8,
    Bool16,
    Bool32,
    Bool64
  };

  enum MemEltTy {
    MemEltTyDefault,
    MemEltTyInt8,
    MemEltTyInt16,
    MemEltTyInt32,
    MemEltTyInt64
  };

  SVETypeFlags(uint64_t F) : Flags(F) { }
  SVETypeFlags(EltType ET, bool IsUnsigned) : Flags(ET) { }

  EltType getEltType() const { return (EltType)(Flags & EltTypeMask); }
  MemEltTy getMemEltType() const {
    return (MemEltTy)((Flags & MemEltTypeMask) >> MemEltTypeOffset);
  }

  bool isLoad() const { return Flags & IsLoad; }

  uint64_t getBits() const { return Flags; }
  bool isFlagSet(uint64_t Flag) const { return Flags & Flag; }
};

} // end namespace clang

#endif
