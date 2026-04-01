//===- TensorToLinalg.cpp - Tensor to Linalg Patterns ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"

#define DEBUG_TYPE "tensor-to-linalg-pattern"

using namespace aiir;

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void aiir::populateTensorToLinalgPatterns(RewritePatternSet &patterns) {
  // TODO: Add the remaining patterns, e.g. to decompose Pack/Unpack Ops.
  // Alternatively, delete this file.
  patterns.add<aiir::linalg::DecomposePadOpPattern>(patterns.getContext());
}
