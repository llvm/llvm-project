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

namespace cir {

// Forward declarations.
class LowerTypes;

/// Target specific hooks for defining how a type should be passed or returned
/// from functions.
/// FIXME(cir): this needs to be merged with clang/lib/CIR/CodeGen/ABIInfo.h
class ABIInfo {
protected:
  LowerTypes &lt;
  llvm::CallingConv::ID RuntimeCC;

public:
  ABIInfo(LowerTypes &lt) : lt(lt), RuntimeCC(llvm::CallingConv::C) {}
  virtual ~ABIInfo();
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H