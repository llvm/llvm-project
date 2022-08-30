//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class BlockAndValueMapping;
class GreedyRewriteConfig;
class Operation;
class Pass;
class Region;
} // namespace mlir

namespace fir {

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_ABSTRACTRESULTONFUNCOPTPASS
#define GEN_PASS_DECL_ABSTRACTRESULTONGLOBALOPTPASS
#define GEN_PASS_DECL_AFFINEDIALECTPROMOTIONPASS
#define GEN_PASS_DECL_AFFINEDIALECTDEMOTIONPASS
#define GEN_PASS_DECL_ANNOTATECONSTANTOPERANDSPASS
#define GEN_PASS_DECL_ARRAYVALUECOPYPASS
#define GEN_PASS_DECL_CHARACTERCONVERSIONPASS
#define GEN_PASS_DECL_CFGCONVERSIONPASS
#define GEN_PASS_DECL_EXTERNALNAMECONVERSIONPASS
#define GEN_PASS_DECL_MEMREFDATAFLOWOPTPASS
#define GEN_PASS_DECL_SIMPLIFYINTRINSICSPASS
#define GEN_PASS_DECL_MEMORYALLOCATIONOPTPASS
#define GEN_PASS_DECL_SIMPLIFYREGIONLITEPASS
#define GEN_PASS_DECL_ALGEBRAICSIMPLIFICATIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAbstractResultOnFuncOptPass();
std::unique_ptr<mlir::Pass> createAbstractResultOnGlobalOptPass();
std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass> createArrayValueCopyPass();
std::unique_ptr<mlir::Pass> createFirToCfgPass();
std::unique_ptr<mlir::Pass> createCharacterConversionPass();
std::unique_ptr<mlir::Pass> createExternalNameConversionPass();
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
std::unique_ptr<mlir::Pass> createMemoryAllocationPass();
std::unique_ptr<mlir::Pass> createSimplifyIntrinsicsPass();

std::unique_ptr<mlir::Pass>
createMemoryAllocationPass(bool dynOnHeap, std::size_t maxStackSize);
std::unique_ptr<mlir::Pass> createAnnotateConstantOperandsPass();
std::unique_ptr<mlir::Pass> createSimplifyRegionLitePass();
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();
std::unique_ptr<mlir::Pass>
createAlgebraicSimplificationPass(const mlir::GreedyRewriteConfig &config);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
