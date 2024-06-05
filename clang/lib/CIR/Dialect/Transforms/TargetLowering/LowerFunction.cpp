//===--- LowerFunction.cpp - Lower CIR Function Code ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenFunction.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerFunction.h"
#include "LowerCall.h"
#include "LowerModule.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

// FIXME(cir): Pass SrcFn and NewFn around instead of having then as attributes.
LowerFunction::LowerFunction(LowerModule &LM, PatternRewriter &rewriter,
                             FuncOp srcFn, FuncOp newFn)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), NewFn(newFn),
      LM(LM) {}

LowerFunction::LowerFunction(LowerModule &LM, PatternRewriter &rewriter,
                             FuncOp srcFn, CallOp callOp)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), callOp(callOp),
      LM(LM) {}

} // namespace cir
} // namespace mlir
