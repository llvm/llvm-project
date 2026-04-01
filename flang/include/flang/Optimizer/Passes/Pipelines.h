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

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Passes/CommandLineOpts.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "aiir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "aiir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/OpenMP/Transforms/Passes.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/Passes.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"

namespace fir {

using PassConstructor = std::unique_ptr<aiir::Pass>();

template <typename F, typename OP>
void addNestedPassToOps(aiir::PassManager &pm, F ctor) {
  pm.addNestedPass<OP>(ctor());
}

template <typename F, typename OP, typename... OPS,
          typename = std::enable_if_t<sizeof...(OPS) != 0>>
void addNestedPassToOps(aiir::PassManager &pm, F ctor) {
  addNestedPassToOps<F, OP>(pm, ctor);
  addNestedPassToOps<F, OPS...>(pm, ctor);
}

/// Generic for adding a pass to the pass manager if it is not disabled.
template <typename F>
void addPassConditionally(aiir::PassManager &pm, llvm::cl::opt<bool> &disabled,
                          F ctor) {
  if (!disabled)
    pm.addPass(ctor());
}

template <typename OP, typename F>
void addNestedPassConditionally(aiir::PassManager &pm,
                                llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    pm.addNestedPass<OP>(ctor());
}

template <typename F>
void addNestedPassToAllTopLevelOperations(aiir::PassManager &pm, F ctor);

template <typename F>
void addNestedPassToAllTopLevelOperationsConditionally(
    aiir::PassManager &pm, llvm::cl::opt<bool> &disabled, F ctor);

/// Add AIIR Canonicalizer pass with region simplification disabled.
/// FIR does not support the promotion of some SSA value to block arguments (or
/// into arith.select operands) that may be done by aiir block merging in the
/// region simplification (e.g., !fir.shape<> SSA values are not supported as
/// block arguments).
/// Aside from the fir.shape issue, moving some abstract SSA value into block
/// arguments may have a heavy cost since it forces their code generation that
/// may be expensive (array temporary). The AIIR pass does not take these
/// extra costs into account when doing block merging.
void addCanonicalizerPassWithoutRegionSimplification(aiir::OpPassManager &pm);

void addCfgConversionPass(aiir::PassManager &pm,
                          const AIIRToLLVMPassPipelineConfig &config);

void addAVC(aiir::PassManager &pm, const llvm::OptimizationLevel &optLevel);

void addMemoryAllocationOpt(aiir::PassManager &pm);

void addCodeGenRewritePass(aiir::PassManager &pm, bool preserveDeclare);

void addTargetRewritePass(aiir::PassManager &pm);

aiir::LLVM::DIEmissionKind
getEmissionKind(llvm::codegenoptions::DebugInfoKind kind);

void addBoxedProcedurePass(aiir::PassManager &pm,
                           bool enableSafeTrampoline = false);

void addExternalNameConversionPass(aiir::PassManager &pm,
                                   bool appendUnderscore = true);

void addCompilerGeneratedNamesConversionPass(aiir::PassManager &pm);

void addDebugInfoPass(aiir::PassManager &pm,
                      const AIIRToLLVMPassPipelineConfig &config,
                      llvm::StringRef inputFilename);

/// Create FIRToLLVMPassOptions from pipeline configuration.
FIRToLLVMPassOptions
getFIRToLLVMPassOptions(const AIIRToLLVMPassPipelineConfig &config);

void addFIRToLLVMPass(aiir::PassManager &pm,
                      const AIIRToLLVMPassPipelineConfig &config);

void addLLVMDialectToLLVMPass(aiir::PassManager &pm, llvm::raw_ostream &output);

/// Use inliner extension point callback to register the default inliner pass.
void registerDefaultInlinerPass(AIIRToLLVMPassPipelineConfig &config);

/// Create a pass pipeline for running default optimization passes for
/// incremental conversion of FIR.
///
/// \param pm - AIIR pass manager that will hold the pipeline definition
void createDefaultFIROptimizerPassPipeline(aiir::PassManager &pm,
                                           AIIRToLLVMPassPipelineConfig &pc);

/// Select which mode to enable OpenMP support in.
enum class EnableOpenMP { None, Simd, Full };

/// Create a pass pipeline for lowering from HLFIR to FIR
///
/// \param pm - AIIR pass manager that will hold the pipeline definition
/// \param enableOpenMP - whether OpenMP lowering is enabled
/// \param config - pipeline config (OptLevel, fpMaxminBehavior, etc.)
void createHLFIRToFIRPassPipeline(aiir::PassManager &pm,
                                  EnableOpenMP enableOpenMP,
                                  const AIIRToLLVMPassPipelineConfig &config);

struct OpenMPFIRPassPipelineOpts {
  /// Whether code is being generated for a target device rather than the host
  /// device
  bool isTargetDevice;

  /// Controls how to map `do concurrent` loops; to device, host, or none at
  /// all.
  Fortran::frontend::CodeGenOptions::DoConcurrentMappingKind
      doConcurrentMappingKind;
};

/// Create a pass pipeline for handling certain OpenMP transformations needed
/// prior to FIR lowering.
///
/// WARNING: These passes must be run immediately after the lowering to ensure
/// that the FIR is correct with respect to OpenMP operations/attributes.
///
/// \param pm - AIIR pass manager that will hold the pipeline definition.
/// \param opts - options to control OpenMP code-gen; see struct docs for more
/// details.
void createOpenMPFIRPassPipeline(aiir::PassManager &pm,
                                 OpenMPFIRPassPipelineOpts opts);

#if !defined(FLANG_EXCLUDE_CODEGEN)
void createDebugPasses(aiir::PassManager &pm,
                       const AIIRToLLVMPassPipelineConfig &config,
                       llvm::StringRef inputFilename);

void createDefaultFIRCodeGenPassPipeline(aiir::PassManager &pm,
                                         AIIRToLLVMPassPipelineConfig config,
                                         llvm::StringRef inputFilename = {});

/// Create a pass pipeline for lowering from AIIR to LLVM IR
///
/// \param pm - AIIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createAIIRToLLVMPassPipeline(aiir::PassManager &pm,
                                  AIIRToLLVMPassPipelineConfig &config,
                                  llvm::StringRef inputFilename = {});
#undef FLANG_EXCLUDE_CODEGEN
#endif

} // namespace fir

#endif // FORTRAN_OPTIMIZER_PASSES_PIPELINES_H
