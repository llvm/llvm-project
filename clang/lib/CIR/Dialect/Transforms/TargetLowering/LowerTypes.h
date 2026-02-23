//===--- LowerTypes.cpp - Type lowering for CIR dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H

#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

namespace cir {

// Forward declarations.
class LowerModule;

/// This class organizes lowering to ABI-specific types in CIR.
class LowerTypes {
  // FIXME(cir): This abstraction could likely be replaced by a MLIR interface
  // or direct queries to CIR types. It's here mostly for code parity.

private:
  LowerModule &lm;
  cir::CIRDataLayout dataLayout;

public:
  LowerTypes(LowerModule &lm);
  ~LowerTypes() = default;

  LowerModule &getLm() const { return lm; }
  const cir::CIRDataLayout &getDataLayout() const { return dataLayout; }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
