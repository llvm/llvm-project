//===- TensorToLinalg.h - Tensor to Linalg Patterns -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
#define AIIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {

/// Appends to a pattern list additional patterns for translating tensor ops
/// to Linalg ops.
void populateTensorToLinalgPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
