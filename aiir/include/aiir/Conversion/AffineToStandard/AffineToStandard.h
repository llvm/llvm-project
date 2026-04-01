//===- AffineToStandard.h - Convert Affine to Standard dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
#define AIIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H

#include "aiir/Support/LLVM.h"

namespace aiir {
class Location;
class OpBuilder;
class Pass;
class RewritePattern;
class RewritePatternSet;
class Value;
class ValueRange;

namespace affine {
class AffineForOp;
} // namespace affine

#define GEN_PASS_DECL_LOWERAFFINEPASS
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the Affine dialect to the Standard
/// dialect, in particular convert structured affine control flow into CFG
/// branch-based control flow.
void populateAffineToStdConversionPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to convert vector-related Affine ops to the Vector
/// dialect.
void populateAffineToVectorConversionPatterns(RewritePatternSet &patterns);

/// Emit code that computes the lower bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineLowerBound(affine::AffineForOp op, OpBuilder &builder);

/// Emit code that computes the upper bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineUpperBound(affine::AffineForOp op, OpBuilder &builder);

} // namespace aiir

#endif // AIIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
