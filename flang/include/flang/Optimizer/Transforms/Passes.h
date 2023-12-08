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

#define GEN_PASS_DECL_ABSTRACTRESULTONFUNCOPT
#define GEN_PASS_DECL_ABSTRACTRESULTONGLOBALOPT
#define GEN_PASS_DECL_AFFINEDIALECTPROMOTION
#define GEN_PASS_DECL_AFFINEDIALECTDEMOTION
#define GEN_PASS_DECL_ANNOTATECONSTANTOPERANDS
#define GEN_PASS_DECL_ARRAYVALUECOPY
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
#include "flang/Optimizer/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAbstractResultOnFuncOptPass();
std::unique_ptr<mlir::Pass> createAbstractResultOnGlobalOptPass();
std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass>
createArrayValueCopyPass(fir::ArrayValueCopyOptions options = {});
std::unique_ptr<mlir::Pass> createFirToCfgPass();
std::unique_ptr<mlir::Pass> createCharacterConversionPass();
std::unique_ptr<mlir::Pass> createExternalNameConversionPass();
std::unique_ptr<mlir::Pass>
createExternalNameConversionPass(bool appendUnderscore);
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
std::unique_ptr<mlir::Pass> createMemoryAllocationPass();
std::unique_ptr<mlir::Pass> createStackArraysPass();
std::unique_ptr<mlir::Pass> createAliasTagsPass();
std::unique_ptr<mlir::Pass> createSimplifyIntrinsicsPass();
std::unique_ptr<mlir::Pass> createAddDebugFoundationPass();
std::unique_ptr<mlir::Pass> createLoopVersioningPass();

std::unique_ptr<mlir::Pass>
createMemoryAllocationPass(bool dynOnHeap, std::size_t maxStackSize);
std::unique_ptr<mlir::Pass> createAnnotateConstantOperandsPass();
std::unique_ptr<mlir::Pass> createSimplifyRegionLitePass();
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();
std::unique_ptr<mlir::Pass>
createAlgebraicSimplificationPass(const mlir::GreedyRewriteConfig &config);
std::unique_ptr<mlir::Pass> createPolymorphicOpConversionPass();

std::unique_ptr<mlir::Pass> createOMPFunctionFilteringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOMPMarkDeclareTargetPass();

std::unique_ptr<mlir::Pass> createVScaleAttrPass();
std::unique_ptr<mlir::Pass>
createVScaleAttrPass(std::pair<unsigned, unsigned> vscaleAttr);

struct FunctionAttrTypes {
  mlir::LLVM::framePointerKind::FramePointerKind framePointerKind =
      mlir::LLVM::framePointerKind::FramePointerKind::None;
};

std::unique_ptr<mlir::Pass> createFunctionAttrPass();
std::unique_ptr<mlir::Pass>
createFunctionAttrPass(FunctionAttrTypes &functionAttr);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
