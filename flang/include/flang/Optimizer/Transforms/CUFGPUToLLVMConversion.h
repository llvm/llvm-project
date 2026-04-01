//===------- Optimizer/Transforms/CUFGPUToLLVMConversion.h ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_

#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"
#include "aiir/Transforms/DialectConversion.h"

namespace fir {
class LLVMTypeConverter;
}

namespace cuf {

void populateCUFGPUToLLVMConversionPatterns(fir::LLVMTypeConverter &converter,
                                            aiir::RewritePatternSet &patterns,
                                            aiir::PatternBenefit benefit = 1);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_
