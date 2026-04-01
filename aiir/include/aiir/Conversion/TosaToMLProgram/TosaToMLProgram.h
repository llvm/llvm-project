//===-- TosaToMLProgram.h - TOSA to MLProgram dialect lowerings-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to MLProgram Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H
#define AIIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H

#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {

#define GEN_PASS_DECL_TOSATOMLPROGRAM

namespace tosa {

void populateTosaToMLProgramConversionPatterns(RewritePatternSet *patterns);

} // namespace tosa
} // namespace aiir

#endif // AIIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H
