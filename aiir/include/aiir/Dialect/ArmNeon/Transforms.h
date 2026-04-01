//===- Transforms.h - ArmNeon Transformation Entrypoints --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARMNEON_TRANSFORMS_H
#define AIIR_DIALECT_ARMNEON_TRANSFORMS_H

namespace aiir {
class RewritePatternSet;

namespace arm_neon {
void populateLowerContractionToNeonI8MMPatterns(RewritePatternSet &patterns);
void populateLowerContractionToNeonBFMMLAPatterns(RewritePatternSet &patterns);
} // namespace arm_neon

} // namespace aiir

#endif // AIIR_DIALECT_ARMNEON_TRANSFORMS_H
