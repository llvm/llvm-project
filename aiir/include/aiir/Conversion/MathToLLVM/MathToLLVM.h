//===- MathToLLVM.h - Math to LLVM dialect conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H
#define AIIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H

#include "aiir/IR/PatternMatch.h"
#include <memory>

namespace aiir {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"

void populateMathToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          bool approximateLog1p = true,
                                          PatternBenefit benefit = 1);

void registerConvertMathToLLVMInterface(DialectRegistry &registry);

} // namespace aiir

#endif // AIIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H
