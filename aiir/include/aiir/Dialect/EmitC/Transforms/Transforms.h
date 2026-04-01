//===- Transforms.h - EmitC transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H
#define AIIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H

#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/IR/PatternMatch.h"

namespace aiir {
namespace emitc {

//===----------------------------------------------------------------------===//
// Expression transforms
//===----------------------------------------------------------------------===//

ExpressionOp createExpression(Operation *op, OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Populate functions
//===----------------------------------------------------------------------===//

/// Populates `patterns` with expression-related patterns.
void populateExpressionPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// The WrapFuncInClass pass.
//===----------------------------------------------------------------------===//

void populateWrapFuncInClass(RewritePatternSet &patterns, StringRef fName);

} // namespace emitc
} // namespace aiir

#endif // AIIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H
