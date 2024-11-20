//===------- Optimizer/Transforms/CUFOpConversion.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;
}

namespace mlir {
class DataLayout;
class SymbolTable;
} // namespace mlir

namespace cuf {

void populateCUFToFIRConversionPatterns(const fir::LLVMTypeConverter &converter,
                                        mlir::DataLayout &dl,
                                        const mlir::SymbolTable &symtab,
                                        mlir::RewritePatternSet &patterns);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_
