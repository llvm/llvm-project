//===- ComplexToLibm.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_
#define AIIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_

#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOLIBM
#include "aiir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to Libm
/// calls.
void populateComplexToLibmConversionPatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit);

} // namespace aiir

#endif // AIIR_CONVERSION_COMPLEXTOLIBM_COMPLEXTOLIBM_H_
