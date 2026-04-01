//===-- TosaToArith.h - TOSA optimization pass declarations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to Standard Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TOSATOARITH_TOSATOARITH_H
#define AIIR_CONVERSION_TOSATOARITH_TOSATOARITH_H

#include "aiir/Pass/Pass.h"

namespace aiir {

#define GEN_PASS_DECL_TOSATOARITHPASS
#include "aiir/Conversion/Passes.h.inc"

namespace tosa {

void populateTosaToArithConversionPatterns(RewritePatternSet *patterns);

void populateTosaRescaleToArithConversionPatterns(RewritePatternSet *patterns,
                                                  bool include32Bit = false);

} // namespace tosa
} // namespace aiir

#endif // AIIR_CONVERSION_TOSATOARITH_TOSATOARITH_H
