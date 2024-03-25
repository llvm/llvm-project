//===------- Optimizer/CodeGen/CodeGenOpenMP.h - OpenMP codegen -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENMP_H
#define FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENMP_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;

/// Specialised conversion patterns of OpenMP operations for FIR to LLVM
/// dialect, utilised in cases where the default OpenMP dialect handling cannot
/// handle all cases for intermingled fir types and operations.
void populateOpenMPFIRToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENMP_H
