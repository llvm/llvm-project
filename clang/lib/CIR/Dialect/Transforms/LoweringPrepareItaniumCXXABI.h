//====- LoweringPrepareItaniumCXXABI.h - Itanium ABI specific code --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Itanium C++ ABI specific code that is used during LLVMIR
// lowering prepare.
//
//===----------------------------------------------------------------------===//

#include "LoweringPrepareCXXABI.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

class LoweringPrepareItaniumCXXABI : public cir::LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                               clang::ASTContext &astCtx,
                               mlir::cir::DynamicCastOp op) override;
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, mlir::cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override;
};
