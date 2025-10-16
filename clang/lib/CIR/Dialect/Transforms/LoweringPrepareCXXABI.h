//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the LoweringPrepareCXXABI class, which is the base class
// for ABI specific functionalities that are required during LLVM lowering
// prepare.
//
//===----------------------------------------------------------------------===//

#ifndef CIR_DIALECT_TRANSFORMS__LOWERINGPREPARECXXABI_H
#define CIR_DIALECT_TRANSFORMS__LOWERINGPREPARECXXABI_H

#include "mlir/IR/Value.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

class LoweringPrepareCXXABI {
public:
  static LoweringPrepareCXXABI *createItaniumABI();

  virtual ~LoweringPrepareCXXABI() {}

  virtual mlir::Value lowerDynamicCast(CIRBaseBuilderTy &builder,
                                       clang::ASTContext &astCtx,
                                       cir::DynamicCastOp op) = 0;
};

} // namespace cir

#endif // CIR_DIALECT_TRANSFORMS__LOWERINGPREPARECXXABI_H
