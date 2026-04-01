//===- SCFToEmitC.h - SCF to EmitC Pass entrypoint --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H
#define AIIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H

#include "aiir/Transforms/DialectConversion.h"
#include <memory>

namespace aiir {
class DialectRegistry;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_SCFTOEMITC
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert SCF operations to the EmitC dialect.
void populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns,
                                          TypeConverter &typeConverter);

void registerConvertSCFToEmitCInterface(DialectRegistry &registry);
} // namespace aiir

#endif // AIIR_CONVERSION_SCFTOEMITC_SCFTOEMITC_H
