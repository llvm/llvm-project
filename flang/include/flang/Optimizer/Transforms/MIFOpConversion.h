//===------- Optimizer/Transforms/MIFOpToLLVMConversion.h -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;
}

namespace mif {

/// Patterns that convert MIF operations to runtime calls.
void populateMIFOpConversionPatterns(mlir::RewritePatternSet &patterns);

} // namespace mif

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MIFOPCONVERSION_H_
