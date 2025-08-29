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
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class IRMapping;
class GreedyRewriteConfig;
class Operation;
class Pass;
class Region;
class ModuleOp;
} // namespace mlir

namespace fir {

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL

#include "flang/Optimizer/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass>
createArrayValueCopyPass(fir::ArrayValueCopyOptions options = {});
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
std::unique_ptr<mlir::Pass> createFIRToSCFPass();
std::unique_ptr<mlir::Pass>
createAddDebugInfoPass(fir::AddDebugInfoOptions options = {});

std::unique_ptr<mlir::Pass> createAnnotateConstantOperandsPass();
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();
std::unique_ptr<mlir::Pass>
createAlgebraicSimplificationPass(const mlir::GreedyRewriteConfig &config);

std::unique_ptr<mlir::Pass> createVScaleAttrPass();
std::unique_ptr<mlir::Pass>
createVScaleAttrPass(std::pair<unsigned, unsigned> vscaleAttr);

void populateCfgConversionRewrites(mlir::RewritePatternSet &patterns,
                                   bool forceLoopToExecuteOnce = false,
                                   bool setNSW = true);

void populateSimplifyFIROperationsPatterns(mlir::RewritePatternSet &patterns,
                                           bool preferInlineImplementation);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
