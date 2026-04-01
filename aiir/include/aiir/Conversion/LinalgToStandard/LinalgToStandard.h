//===- LinalgToStandard.h - Utils to convert from the linalg dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
#define AIIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_

#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
class ModuleOp;
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_CONVERTLINALGTOSTANDARDPASS
#include "aiir/Conversion/Passes.h.inc"

namespace linalg {

//===----------------------------------------------------------------------===//
// Patterns to convert a LinalgOp to func.call @external library implementation.
//===----------------------------------------------------------------------===//
// These patterns are exposed individually because they are expected to be
// typically used individually.

// Create a new call to the type-canonicalized `LinalgOp::getLibraryCallName()`
// function. The implementation of the function can be either in the same module
// or in an externally linked library.
// This is a generic entry point for all LinalgOp, except for CopyOp, for which
// more specialized patterns are provided.
class LinalgOpToLibraryCallRewrite
    : public OpInterfaceRewritePattern<LinalgOp> {
public:
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override;
};

/// Populate the given list with patterns that convert from Linalg to Standard.
void populateLinalgToStandardConversionPatterns(RewritePatternSet &patterns);

} // namespace linalg
} // namespace aiir

#endif // AIIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
