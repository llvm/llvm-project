//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_
#define AIIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_

#include "aiir/Pass/Pass.h"

namespace aiir {
namespace quant {

#define GEN_PASS_DECL
#include "aiir/Dialect/Quant/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/Quant/Transforms/Passes.h.inc"

void populateLowerQuantOpsPatterns(RewritePatternSet &patterns);

} // namespace quant
} // namespace aiir

#endif // AIIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_
