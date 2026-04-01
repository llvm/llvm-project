//===-- TosaToSCF.h - TOSA to SCF dialect lowerings -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to SCF Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TOSATOSCF_TOSATOSCF_H
#define AIIR_CONVERSION_TOSATOSCF_TOSATOSCF_H

#include "aiir/Pass/Pass.h"

namespace aiir {

#define GEN_PASS_DECL_TOSATOSCFPASS
#include "aiir/Conversion/Passes.h.inc"

namespace tosa {

void populateTosaToSCFConversionPatterns(RewritePatternSet *patterns);

/// Populates passes to convert from TOSA to SCF.
void addTosaToSCFPasses(OpPassManager &pm);

} // namespace tosa
} // namespace aiir

#endif // AIIR_CONVERSION_TOSATOSCF_TOSATOSCF_H
