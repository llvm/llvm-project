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

#define GEN_PASS_DECL_ABSTRACTRESULTOPT
#define GEN_PASS_DECL_AFFINEDIALECTPROMOTION
#define GEN_PASS_DECL_AFFINEDIALECTDEMOTION
#define GEN_PASS_DECL_ANNOTATECONSTANTOPERANDS
#define GEN_PASS_DECL_ARRAYVALUECOPY
#define GEN_PASS_DECL_ASSUMEDRANKOPCONVERSION
#define GEN_PASS_DECL_CHARACTERCONVERSION
#define GEN_PASS_DECL_CFGCONVERSION
#define GEN_PASS_DECL_EXTERNALNAMECONVERSION
#define GEN_PASS_DECL_MEMREFDATAFLOWOPT
#define GEN_PASS_DECL_SIMPLIFYINTRINSICS
#define GEN_PASS_DECL_MEMORYALLOCATIONOPT
#define GEN_PASS_DECL_SIMPLIFYREGIONLITE
#define GEN_PASS_DECL_ALGEBRAICSIMPLIFICATION
#define GEN_PASS_DECL_POLYMORPHICOPCONVERSION
#define GEN_PASS_DECL_OPENACCDATAOPERANDCONVERSION
#define GEN_PASS_DECL_ADDDEBUGINFO
#define GEN_PASS_DECL_STACKARRAYS
#define GEN_PASS_DECL_LOOPVERSIONING
#define GEN_PASS_DECL_ADDALIASTAGS
#define GEN_PASS_DECL_OMPMAPINFOFINALIZATIONPASS
#define GEN_PASS_DECL_OMPMARKDECLARETARGETPASS
#define GEN_PASS_DECL_OMPFUNCTIONFILTERING
#define GEN_PASS_DECL_VSCALEATTR
#define GEN_PASS_DECL_FUNCTIONATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass>
createArrayValueCopyPass(fir::ArrayValueCopyOptions options = {});
std::unique_ptr<mlir::Pass> createCFGConversionPassWithNSW();
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
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
                                   bool setNSW = false);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
