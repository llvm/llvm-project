//===------- Optimizer/Transforms/CUFOpConversion.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_

#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;
}

namespace aiir {
class DataLayout;
class SymbolTable;
} // namespace aiir

namespace cuf {

/// Patterns that convert CUF operations to runtime calls.
void populateCUFToFIRConversionPatterns(const fir::LLVMTypeConverter &converter,
                                        aiir::DataLayout &dl,
                                        const aiir::SymbolTable &symtab,
                                        aiir::RewritePatternSet &patterns);

/// Patterns that updates fir operations in presence of CUF.
void populateFIRCUFConversionPatterns(const aiir::SymbolTable &symtab,
                                      aiir::RewritePatternSet &patterns);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFOPCONVERSION_H_
