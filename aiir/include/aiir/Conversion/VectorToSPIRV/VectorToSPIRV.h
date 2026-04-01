//=- VectorToSPIRV.h - Vector to SPIR-V Patterns ------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Vector dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H
#define AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Vector Ops to
/// SPIR-V ops.
void populateVectorToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                   RewritePatternSet &patterns);

/// Appends patterns to convert vector reduction of the form:
/// ```
///   vector.reduction <add>, (muli (ext %lhs), (ext %rhs)), [%acc]
/// ```
///
/// to SPIR-V integer dot product ops.
void populateVectorReductionToSPIRVDotProductPatterns(
    RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H
