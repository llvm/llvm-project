//===- Passes.h - OpenACC pass entry points -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the OpenACC passes specific to Fortran and FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_PASSES_H
#define FORTRAN_OPTIMIZER_OPENACC_PASSES_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace fir {
namespace acc {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/OpenACC/Passes.h.inc"

std::unique_ptr<mlir::Pass> createACCInitializeFIRAnalysesPass();
std::unique_ptr<mlir::Pass> createACCRecipeBufferizationPass();
std::unique_ptr<mlir::Pass> createACCUseDeviceCanonicalizerPass();

} // namespace acc
} // namespace fir

#endif // FORTRAN_OPTIMIZER_OPENACC_PASSES_H
