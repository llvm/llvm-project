//====- LoweringPrepareCXXABI.h -------------------------------------------===//
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

#ifndef LLVM_CLANG_LIB_CIR_LOWERING_PREPARE_CXX_ABI_H
#define LLVM_CLANG_LIB_CIR_LOWERING_PREPARE_CXX_ABI_H

#include "mlir/IR/Value.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Target/AArch64.h"

namespace cir {

class LoweringPrepareCXXABI {
public:
  static LoweringPrepareCXXABI *createItaniumABI();
  static LoweringPrepareCXXABI *createAArch64ABI(::cir::AArch64ABIKind k);

  virtual mlir::Value lowerVAArg(CIRBaseBuilderTy &builder,
                                 mlir::cir::VAArgOp op,
                                 const cir::CIRDataLayout &datalayout) = 0;
  virtual ~LoweringPrepareCXXABI() {}

  virtual mlir::Value lowerDynamicCast(CIRBaseBuilderTy &builder,
                                       clang::ASTContext &astCtx,
                                       mlir::cir::DynamicCastOp op) = 0;
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_LOWERING_PREPARE_CXX_ABI_H
