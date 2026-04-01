//===-- Passes.h - TOSA optimization pass declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the TOSA Dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_TOSA_TRANSFORMS_PASSES_H

#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
class TypeConverter;
namespace tosa {

#define GEN_PASS_DECL
#include "aiir/Dialect/Tosa/Transforms/Passes.h.inc"

// Expose Rewrite Functions that decompose TOSA Ops into further TOSA Ops.
// The rewrites can be selectively added to a conversion pass.
void populateTosaDecomposeTransposeConv(AIIRContext *ctx,
                                        RewritePatternSet &patterns);
void populateTosaDecomposeDepthwise(AIIRContext *ctx,
                                    RewritePatternSet &patterns);
void populateTosaFoldConstantReciprocalPatterns(AIIRContext *ctx,
                                                RewritePatternSet &patterns);
void populateTosaFoldConstantTransposePatterns(AIIRContext *ctx,
                                               RewritePatternSet &patterns);
void populateTosaConstantReduction(AIIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   bool aggressiveReduceConstant);

void populateTosaTypeConversion(TypeConverter &converter);

std::unique_ptr<Pass> createTosaTestQuantUtilAPIPass();
std::unique_ptr<Pass>
createTosaInputShapePass(std::vector<std::string> args = {});

#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/Tosa/Transforms/Passes.h.inc"

} // namespace tosa
} // namespace aiir

#endif // AIIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
