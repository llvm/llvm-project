//===- VectorToAMX.h - Convert vector to X86 dialect AMX ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H
#define AIIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H

#include "aiir/IR/PatternMatch.h"

namespace aiir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOAMX
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the vector to X86 AMX ops.
void populateVectorToAMXConversionPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_VECTORTOAMX_VECTORTOAMX_H
