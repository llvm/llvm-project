//===- MathToROCDL.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_
#define AIIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_

#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Dialect/AMDGPU/Utils/Chipset.h"
#include "aiir/IR/PatternMatch.h"
#include <memory>

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTOROCDL
#include "aiir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Math to ROCDL calls.
// `chipset` specifies the AMDGPU chipset to target. If `std::nullopt`,
// none of the chipset dependent patterns are added.
void populateMathToROCDLConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    std::optional<amdgpu::Chipset> chipset);
} // namespace aiir

#endif // AIIR_CONVERSION_MATHTOROCDL_MATHTOROCDL_H_
