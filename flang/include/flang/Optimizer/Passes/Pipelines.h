//===-- Pipelines.h -- FIR pass pipelines -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This file declares some utilties to setup FIR pass pipelines. These are
/// common to flang and the test tools.

#ifndef FORTRAN_OPTIMIZER_PASSES_PIPELINES_H
#define FORTRAN_OPTIMIZER_PASSES_PIPELINES_H

#include "flang/Common/CommandLineOpts.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"

namespace {

using PassConstructor = std::unique_ptr<mlir::Pass>();

template <typename OP>
void addNestedPassToOps(mlir::PassManager &pm, PassConstructor ctor) {
  pm.addNestedPass<OP>(ctor());
}

template <typename OP, typename... OPS,
          typename = std::enable_if_t<sizeof...(OPS) != 0>>
void addNestedPassToOps(mlir::PassManager &pm, PassConstructor ctor) {
  addNestedPassToOps<OP>(pm, ctor);
  addNestedPassToOps<OPS...>(pm, ctor);
}

/// Generic for adding a pass to the pass manager if it is not disabled.
template <typename F>
void addPassConditionally(mlir::PassManager &pm, llvm::cl::opt<bool> &disabled,
                          F ctor) {
  if (!disabled)
    pm.addPass(ctor());
}

template <typename OP, typename F>
void addNestedPassConditionally(mlir::PassManager &pm,
                                llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    pm.addNestedPass<OP>(ctor());
}

} // namespace

namespace fir {

void addDebugInfoPass(mlir::PassManager &pm,
                      llvm::codegenoptions::DebugInfoKind debugLevel,
                      llvm::OptimizationLevel optLevel,
                      llvm::StringRef inputFilename);

void addFIRToLLVMPass(mlir::PassManager &pm,
                      const MLIRToLLVMPassPipelineConfig &config);

void addLLVMDialectToLLVMPass(mlir::PassManager &pm, llvm::raw_ostream &output);

/// Use inliner extension point callback to register the default inliner pass.
void registerDefaultInlinerPass(MLIRToLLVMPassPipelineConfig &config);

/// Create a pass pipeline for running default optimization passes for
/// incremental conversion of FIR.
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
void createDefaultFIROptimizerPassPipeline(mlir::PassManager &pm,
                                           MLIRToLLVMPassPipelineConfig &pc);

/// Create a pass pipeline for lowering from HLFIR to FIR
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createHLFIRToFIRPassPipeline(
    mlir::PassManager &pm, llvm::OptimizationLevel optLevel = defaultOptLevel);

/// Create a pass pipeline for handling certain OpenMP transformations needed
/// prior to FIR lowering.
///
/// WARNING: These passes must be run immediately after the lowering to ensure
/// that the FIR is correct with respect to OpenMP operations/attributes.
///
/// \param pm - MLIR pass manager that will hold the pipeline definition.
/// \param isTargetDevice - Whether code is being generated for a target device
/// rather than the host device.
void createOpenMPFIRPassPipeline(mlir::PassManager &pm, bool isTargetDevice);

#if !defined(FLANG_EXCLUDE_CODEGEN)
void createDebugPasses(mlir::PassManager &pm,
                       llvm::codegenoptions::DebugInfoKind debugLevel,
                       llvm::OptimizationLevel OptLevel,
                       llvm::StringRef inputFilename);

void createDefaultFIRCodeGenPassPipeline(mlir::PassManager &pm,
                                         MLIRToLLVMPassPipelineConfig config,
                                         llvm::StringRef inputFilename = {});

/// Create a pass pipeline for lowering from MLIR to LLVM IR
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createMLIRToLLVMPassPipeline(mlir::PassManager &pm,
                                  MLIRToLLVMPassPipelineConfig &config,
                                  llvm::StringRef inputFilename = {});
#undef FLANG_EXCLUDE_CODEGEN
#endif

} // namespace fir

#endif // FORTRAN_OPTIMIZER_PASSES_PIPELINES_H
