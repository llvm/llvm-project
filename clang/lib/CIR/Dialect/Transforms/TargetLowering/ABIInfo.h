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

#include "CIRCXXABI.h"
#include "CIRLowerContext.h"
#include "LowerFunctionInfo.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "llvm/IR/CallingConv.h"

namespace cir {

// Forward declarations.
class LowerTypes;

/// Target specific hooks for defining how a type should be passed or returned
/// from functions.
/// FIXME(cir): this needs to be merged with clang/lib/CIR/CodeGen/ABIInfo.h
class ABIInfo {
protected:
  LowerTypes &LT;
  llvm::CallingConv::ID RuntimeCC;

public:
  ABIInfo(LowerTypes &LT) : LT(LT), RuntimeCC(llvm::CallingConv::C) {}
  virtual ~ABIInfo();

  CIRCXXABI &getCXXABI() const;

  CIRLowerContext &getContext() const;

  const clang::TargetInfo &getTarget() const;

  const cir::CIRDataLayout &getDataLayout() const;

  virtual void computeInfo(LowerFunctionInfo &FI) const = 0;

  // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
  // only difference is that this considers bit-precise integer types as well.
  bool isPromotableIntegerTypeForABI(mlir::Type Ty) const;

  cir::ABIArgInfo getNaturalAlignIndirect(mlir::Type Ty, bool ByVal = true,
                                          bool Realign = false,
                                          mlir::Type Padding = {}) const;
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H
