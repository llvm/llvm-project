//===----- LowerCall.h - Encapsulate calling convention details -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CGCall.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERCALL_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERCALL_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace cir {

/// Contains the address where the return value of a function can be stored, and
/// whether the address is volatile or not.
class ReturnValueSlot {
  Value Addr = {};

  // Return value slot flags
  unsigned IsVolatile : 1;
  unsigned IsUnused : 1;
  unsigned IsExternallyDestructed : 1;

public:
  ReturnValueSlot()
      : IsVolatile(false), IsUnused(false), IsExternallyDestructed(false) {}
  ReturnValueSlot(Value Addr, bool IsVolatile, bool IsUnused = false,
                  bool IsExternallyDestructed = false)
      : Addr(Addr), IsVolatile(IsVolatile), IsUnused(IsUnused),
        IsExternallyDestructed(IsExternallyDestructed) {}

  bool isNull() const { return !Addr; }
  bool isVolatile() const { return IsVolatile; }
  Value getValue() const { return Addr; }
  bool isUnused() const { return IsUnused; }
  bool isExternallyDestructed() const { return IsExternallyDestructed; }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERCALL_H
