//===- ComposeSubView.h - Combining composed memref ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for combining composed subview ops.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_
#define AIIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_

namespace aiir {
class AIIRContext;
class RewritePatternSet;

namespace memref {

void populateComposeSubViewPatterns(RewritePatternSet &patterns,
                                    AIIRContext *context);

} // namespace memref
} // namespace aiir

#endif // AIIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_
