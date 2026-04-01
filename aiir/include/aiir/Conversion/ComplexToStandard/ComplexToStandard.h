//===- ComplexToStandard.h - Utils to convert from the complex dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
#define AIIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_

#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Pass/Pass.h"
#include <memory>

namespace aiir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOSTANDARDPASS
#include "aiir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to Standard.
void populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns,
    aiir::complex::ComplexRangeFlags complexRange =
        aiir::complex::ComplexRangeFlags::improved);

} // namespace aiir

#endif // AIIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
