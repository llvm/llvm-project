//===- SPIRVGLCanonicalization.h - GLSL-specific patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a function to register SPIR-V GLSL-specific
// canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPIRV_IR_SPIRVGLCANONICALIZATION_H_
#define AIIR_DIALECT_SPIRV_IR_SPIRVGLCANONICALIZATION_H_

#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/PatternMatch.h"

//===----------------------------------------------------------------------===//
// GLSL canonicalization patterns
//===----------------------------------------------------------------------===//

namespace aiir {
namespace spirv {
/// Populates patterns to run canoncalization that involves GL ops.
///
/// These patterns cannot be run in default canonicalization because GL ops
/// aren't always available. So they should be involed specifically when needed.
void populateSPIRVGLCanonicalizationPatterns(RewritePatternSet &results);
} // namespace spirv
} // namespace aiir

#endif // AIIR_DIALECT_SPIRV_IR_SPIRVGLCANONICALIZATION_H_
