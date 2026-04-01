//===- ConvertVectorToLLVM.h - Utils to convert from the vector dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
#define AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
class LLVMTypeConverter;

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool reassociateFPReductions = false, bool force32BitVectorIndices = false,
    bool useVectorAlignment = false);

namespace vector {
void registerConvertVectorToLLVMInterface(DialectRegistry &registry);
}
} // namespace aiir

#endif // AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
