//===- Approximation.h - Math dialect -----------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H
#define AIIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H

#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/IR/PatternMatch.h"

namespace aiir {
namespace math {

struct ErfPolynomialApproximation : public OpRewritePattern<math::ErfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const final;
};

struct ErfcPolynomialApproximation : public OpRewritePattern<math::ErfcOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfcOp op,
                                PatternRewriter &rewriter) const final;
};

} // namespace math
} // namespace aiir

#endif // AIIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H
