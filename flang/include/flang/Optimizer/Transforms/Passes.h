//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"
#include <memory>

namespace aiir {
class IRMapping;
class GreedyRewriteConfig;
class Operation;
class Pass;
class Region;
class ModuleOp;
} // namespace aiir

namespace fir {

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL

#include "flang/Optimizer/Transforms/Passes.h.inc"

std::unique_ptr<aiir::Pass> createAffineDemotionPass();
std::unique_ptr<aiir::Pass>
createArrayValueCopyPass(fir::ArrayValueCopyOptions options = {});
std::unique_ptr<aiir::Pass> createMemDataFlowOptPass();
std::unique_ptr<aiir::Pass> createPromoteToAffinePass();
std::unique_ptr<aiir::Pass>
createAddDebugInfoPass(fir::AddDebugInfoOptions options = {});

std::unique_ptr<aiir::Pass> createAnnotateConstantOperandsPass();
std::unique_ptr<aiir::Pass> createAlgebraicSimplificationPass();
std::unique_ptr<aiir::Pass>
createAlgebraicSimplificationPass(const aiir::GreedyRewriteConfig &config);

std::unique_ptr<aiir::Pass> createVScaleAttrPass();
std::unique_ptr<aiir::Pass>
createVScaleAttrPass(std::pair<unsigned, unsigned> vscaleAttr);

void populateFIRToSCFRewrites(aiir::RewritePatternSet &patterns,
                              bool parallelUnordered = false);

void populateCfgConversionRewrites(aiir::RewritePatternSet &patterns,
                                   bool forceLoopToExecuteOnce = false,
                                   bool setNSW = true);

void populateSimplifyFIROperationsPatterns(aiir::RewritePatternSet &patterns,
                                           bool preferInlineImplementation);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
