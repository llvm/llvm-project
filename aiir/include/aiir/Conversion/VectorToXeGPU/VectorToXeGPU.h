//===- VectorToXeGPU.h - Convert vector to XeGPU dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
#define AIIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H

#include "aiir/IR/PatternMatch.h"

namespace aiir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOXEGPU
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the vector to XeGPU ops.
void populateVectorToXeGPUConversionPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
