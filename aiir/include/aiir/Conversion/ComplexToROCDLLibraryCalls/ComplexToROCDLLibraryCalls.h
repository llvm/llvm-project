//===- ComplexToROCDLLibraryCalls.h - convert from Complex to ROCDL calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_
#define AIIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_

#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOROCDLLIBRARYCALLS
#include "aiir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Complex to ROCDL
/// calls.
void populateComplexToROCDLLibraryCallsConversionPatterns(
    RewritePatternSet &patterns);
} // namespace aiir

#endif // AIIR_CONVERSION_COMPLEXTOROCDLLIBRARYCALLS_COMPLEXTOROCDLLIBRARYCALLS_H_
