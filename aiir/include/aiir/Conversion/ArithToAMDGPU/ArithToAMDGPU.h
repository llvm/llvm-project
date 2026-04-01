//===- ArithToAMDGPU.h - Arith to AMDGPU dialect conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H
#define AIIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H

#include "aiir/Dialect/AMDGPU/Utils/Chipset.h"
#include "aiir/IR/PatternMatch.h"
#include <memory>
#include <string>

namespace aiir {

class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOAMDGPUCONVERSIONPASS
#include "aiir/Conversion/Passes.h.inc"

namespace arith {
/// Add patterns for rewriting `arith.extf` and `arith.truncf` on FP8 types
/// to wrappers around AMDGPU--specific intrinsics. If `saturateFP8TruncF`
/// is set, values outside the range of the destination type are clamped
/// to the largest value of that type instead of being rewritten to Inf (aka
/// NaN).
void populateArithToAMDGPUConversionPatterns(
    RewritePatternSet &patterns, bool convertFP8Arithmetic,
    bool saturateFP8Truncf, bool allowPackedF16Rtz, bool supportsScaledExtTrunc,
    amdgpu::Chipset chipset, PatternBenefit benefit = 1);
} // namespace arith
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHTOAMDGPU_ARITHTOAMDGPU_H
