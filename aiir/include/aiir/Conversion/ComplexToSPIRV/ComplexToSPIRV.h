//===- ComplexToSPIRV.h - Complex to SPIR-V Patterns ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRV_H
#define AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRV_H

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Complex ops
/// to SPIR-V ops.
void populateComplexToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                    RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRV_H
