//===------- Optimizer/Transforms/MIFOpToLLVMConversion.h -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_

#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;
}

namespace mif {

/// Patterns that convert MIF operations to runtime calls.
void populateMIFOpConversionPatterns(aiir::RewritePatternSet &patterns);

} // namespace mif

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
