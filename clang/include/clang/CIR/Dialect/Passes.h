//===- Passes.h - CIR pass entry points -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_PASSES_H
#define CLANG_CIR_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/Transforms/CallConvTarget.h"
#include "llvm/ABI/TargetInfo.h"

namespace mlir {

std::unique_ptr<Pass> createCIRCanonicalizePass();
std::unique_ptr<Pass> createCIRFoldEmptyBranchesPass();
std::unique_ptr<Pass> createCIRFlattenCFGPass();
std::unique_ptr<Pass> createCIRSimplifyPass();
std::unique_ptr<Pass> createCIREHABILoweringPass();
std::unique_ptr<Pass> createCXXABILoweringPass();
std::unique_ptr<Pass> createTargetLoweringPass();
std::unique_ptr<Pass> createCallConvLoweringPass();
std::unique_ptr<Pass>
createCallConvLoweringPass(cir::CallConvTarget target,
                           llvm::abi::X86AVXABILevel x86AvxAbiLevel,
                           bool allowsX86TargetAttrAvx,
                           const llvm::abi::X86ABICompatInfo &x86AbiCompat);
std::unique_ptr<Pass> createHoistAllocasPass();
std::unique_ptr<Pass> createLoweringPreparePass();
std::unique_ptr<Pass> createGotoSolverPass();
std::unique_ptr<Pass> createIdiomRecognizerPass();
std::unique_ptr<Pass> createLibOptPass();

void populateCIRPreLoweringPasses(mlir::OpPassManager &pm);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerCIRDialectTranslation(mlir::MLIRContext &context);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "clang/CIR/Dialect/Passes.h.inc"

} // namespace mlir

#endif // CLANG_CIR_DIALECT_PASSES_H
