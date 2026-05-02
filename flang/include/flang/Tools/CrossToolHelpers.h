//===-- Tools/CrossToolHelpers.h --------------------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A header file for containing functionallity that is used across Flang tools,
// such as helper functions which apply or generate information needed accross
// tools like bbc and flang.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
#define FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H

#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Support/FPMaxminBehavior.h"
#include "flang/Support/LangOptions.h"
#include "flang/Support/MathOptionsBase.h"
#include <cstdint>

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Passes/OptimizationLevel.h"

// Flang Extension Point Callbacks
class FlangEPCallBacks {
public:
  void registerFIROptEarlyEPCallbacks(
      const std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>
          &C) {
    FIROptEarlyEPCallbacks.push_back(C);
  }

  void registerFIRInlinerCallback(
      const std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>
          &C) {
    FIRInlinerCallback.push_back(C);
  }

  void registerFIROptLastEPCallbacks(
      const std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>
          &C) {
    FIROptLastEPCallbacks.push_back(C);
  }

  void invokeFIROptEarlyEPCallbacks(
      mlir::PassManager &pm, llvm::OptimizationLevel optLevel) {
    for (auto &C : FIROptEarlyEPCallbacks)
      C(pm, optLevel);
  };

  void invokeFIRInlinerCallback(
      mlir::PassManager &pm, llvm::OptimizationLevel optLevel) {
    for (auto &C : FIRInlinerCallback)
      C(pm, optLevel);
  };

  void invokeFIROptLastEPCallbacks(
      mlir::PassManager &pm, llvm::OptimizationLevel optLevel) {
    for (auto &C : FIROptLastEPCallbacks)
      C(pm, optLevel);
  };

private:
  llvm::SmallVector<
      std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>, 1>
      FIROptEarlyEPCallbacks;

  llvm::SmallVector<
      std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>, 1>
      FIRInlinerCallback;

  llvm::SmallVector<
      std::function<void(mlir::PassManager &, llvm::OptimizationLevel)>, 1>
      FIROptLastEPCallbacks;
};

/// Configuriation for the MLIR to LLVM pass pipeline.
struct MLIRToLLVMPassPipelineConfig : public FlangEPCallBacks {
  explicit MLIRToLLVMPassPipelineConfig(llvm::OptimizationLevel level) {
    OptLevel = level;
  }
  explicit MLIRToLLVMPassPipelineConfig(llvm::OptimizationLevel level,
      const Fortran::frontend::CodeGenOptions &opts,
      const Fortran::common::MathOptionsBase &mathOpts) {
    OptLevel = level;
    StackArrays = opts.StackArrays;
    EnableSafeTrampoline = opts.EnableSafeTrampoline;
    Underscoring = opts.Underscoring;
    LoopVersioning = opts.LoopVersioning;
    DebugInfo = opts.getDebugInfo();
    AliasAnalysis = opts.AliasAnalysis;
    FramePointerKind = opts.getFramePointer();
    // The logic for setting these attributes is intended to match the logic
    // used in Clang.
    NoInfsFPMath = mathOpts.getNoHonorInfs();
    NoNaNsFPMath = mathOpts.getNoHonorNaNs();
    ApproxFuncFPMath = mathOpts.getApproxFunc();
    NoSignedZerosFPMath = mathOpts.getNoSignedZeros();
    UnsafeFPMath = mathOpts.getAssociativeMath() &&
        mathOpts.getReciprocalMath() && NoSignedZerosFPMath &&
        ApproxFuncFPMath && mathOpts.getFPContractEnabled();
    Reciprocals = opts.Reciprocals;
    PreferVectorWidth = opts.PreferVectorWidth;
    UseSampleProfile = !opts.SampleProfileFile.empty();
    DebugInfoForProfiling = opts.DebugInfoForProfiling;
    if (opts.InstrumentFunctions) {
      InstrumentFunctionEntry = "__cyg_profile_func_enter";
      InstrumentFunctionExit = "__cyg_profile_func_exit";
    }
    DwarfVersion = opts.DwarfVersion;
    SplitDwarfFile = opts.SplitDwarfFile;
    DwarfDebugFlags = opts.DwarfDebugFlags;
  }

  llvm::OptimizationLevel OptLevel; ///< optimisation level
  bool StackArrays = false; ///< convert memory allocations to alloca.
  bool EnableSafeTrampoline{false}; ///< Use runtime trampoline pool (W^X).
  bool Underscoring = true; ///< add underscores to function names.
  bool LoopVersioning = false; ///< Run the version loop pass.
  bool AliasAnalysis = false; ///< Add TBAA tags to generated LLVMIR.
  llvm::codegenoptions::DebugInfoKind DebugInfo =
      llvm::codegenoptions::NoDebugInfo; ///< Debug info generation.
  llvm::FramePointerKind FramePointerKind =
      llvm::FramePointerKind::None; ///< Add frame pointer to functions.
  unsigned VScaleMin = 0; ///< SVE vector range minimum.
  unsigned VScaleMax = 0; ///< SVE vector range maximum.
  bool NoInfsFPMath = false; ///< Set ninf flag for instructions.
  bool NoNaNsFPMath = false; ///< Set no-nans-fp-math attribute for functions.
  bool ApproxFuncFPMath = false; ///< Set afn flag for instructions.
  bool NoSignedZerosFPMath =
      false; ///< Set no-signed-zeros-fp-math attribute for functions.
  bool UnsafeFPMath = false; ///< Set all fast-math flags for instructions.
  std::string Reciprocals = ""; ///< Set reciprocal-estimate attribute for
                                ///< functions.
  std::string PreferVectorWidth = ""; ///< Set prefer-vector-width attribute for
                                      ///< functions.
  bool NSWOnLoopVarInc = true; ///< Add nsw flag to loop variable increments.
  bool EnableOpenMP = false; ///< Enable OpenMP lowering.
  bool UseSampleProfile = false; ///< Enable sample based profiling
  bool DebugInfoForProfiling = false; ///< Enable extra debugging info
  bool EnableOpenMPSimd = false; ///< Enable OpenMP simd-only mode.
  bool SkipConvertComplexPow = false; ///< Do not run complex pow conversion.
  std::string InstrumentFunctionEntry =
      ""; ///< Name of the instrument-function that is called on each
          ///< function-entry
  std::string InstrumentFunctionExit =
      ""; ///< Name of the instrument-function that is called on each
          ///< function-exit
  Fortran::frontend::CodeGenOptions::ComplexRangeKind ComplexRange =
      Fortran::frontend::CodeGenOptions::ComplexRangeKind::
          CX_Full; ///< Method for calculating complex number division
  int32_t DwarfVersion = 0; ///< Version of DWARF debug info to generate
  std::string SplitDwarfFile = ""; ///< File name for the split debug info
  std::string DwarfDebugFlags = ""; ///< Debug flags to append to DWARF producer
  Fortran::common::FPMaxminBehavior fpMaxminBehavior =
      Fortran::common::FPMaxminBehavior::Legacy;
};

/// Create OffloadModuleOpts from Flang LangOptions.
[[maybe_unused]] static mlir::omp::OffloadModuleOpts makeOffloadModuleOpts(
    Fortran::common::LangOptions &Opts) {
  return mlir::omp::OffloadModuleOpts(Opts.OpenMPTargetDebug,
      Opts.OpenMPTeamSubscription, Opts.OpenMPThreadSubscription,
      Opts.OpenMPNoThreadState, Opts.OpenMPNoNestedParallelism,
      Opts.OpenMPIsTargetDevice, Opts.OpenMPIsGPU, Opts.OpenMPForceUSM,
      Opts.OpenMPVersion, Opts.OMPHostIRFile, Opts.OMPTargetTriples,
      Opts.NoGPULib);
}

#endif // FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
