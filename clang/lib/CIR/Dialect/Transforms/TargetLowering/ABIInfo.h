//===----- ABIInfo.h - CIR's ABI information --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the CodeGen/ABIInfo.h class. The main difference
// is that this is adapted to operate on the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H

#include "llvm/IR/CallingConv.h"

namespace mlir {
namespace cir {

// Forward declarations.
class LowerTypes;

/// Target specific hooks for defining how a type should be passed or returned
/// from functions.
class ABIInfo {
protected:
  LowerTypes &LT;
  llvm::CallingConv::ID RuntimeCC;

public:
  ABIInfo(LowerTypes &LT) : LT(LT), RuntimeCC(llvm::CallingConv::C) {}
  virtual ~ABIInfo();
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H
