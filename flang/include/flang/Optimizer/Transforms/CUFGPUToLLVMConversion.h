//===------- Optimizer/Transforms/CUFGPUToLLVMConversion.h ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
class LLVMTypeConverter;
}

namespace cuf {

void populateCUFGPUToLLVMConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit = 1);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFGPUTOLLVMCONVERSION_H_
