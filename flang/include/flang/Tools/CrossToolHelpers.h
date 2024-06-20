//===-- Tools/CrossToolHelpers.h --------------------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A header file for containing functionallity that is used across Flang tools,
// such as helper functions which apply or generate information needed accross
// tools like bbc and flang-new.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
#define FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H

#include "flang/Common/MathOptionsBase.h"
#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/LangOptions.h"
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
  }

  llvm::OptimizationLevel OptLevel; ///< optimisation level
  bool StackArrays = false; ///< convert memory allocations to alloca.
  bool Underscoring = true; ///< add underscores to function names.
  bool LoopVersioning = false; ///< Run the version loop pass.
  bool AliasAnalysis = false; ///< Add TBAA tags to generated LLVMIR
  llvm::codegenoptions::DebugInfoKind DebugInfo =
      llvm::codegenoptions::NoDebugInfo; ///< Debug info generation.
  llvm::FramePointerKind FramePointerKind =
      llvm::FramePointerKind::None; ///< Add frame pointer to functions.
  unsigned VScaleMin = 0; ///< SVE vector range minimum.
  unsigned VScaleMax = 0; ///< SVE vector range maximum.
  bool NoInfsFPMath = false; ///< Set no-infs-fp-math attribute for functions.
  bool NoNaNsFPMath = false; ///< Set no-nans-fp-math attribute for functions.
  bool ApproxFuncFPMath =
      false; ///< Set approx-func-fp-math attribute for functions.
  bool NoSignedZerosFPMath =
      false; ///< Set no-signed-zeros-fp-math attribute for functions.
  bool UnsafeFPMath = false; ///< Set unsafe-fp-math attribute for functions.
  bool NSWOnLoopVarInc = false; ///< Add nsw flag to loop variable increments.
};

struct OffloadModuleOpts {
  OffloadModuleOpts() {}
  OffloadModuleOpts(uint32_t OpenMPTargetDebug, bool OpenMPTeamSubscription,
      bool OpenMPThreadSubscription, bool OpenMPNoThreadState,
      bool OpenMPNoNestedParallelism, bool OpenMPIsTargetDevice,
      bool OpenMPIsGPU, uint32_t OpenMPVersion, std::string OMPHostIRFile = {},
      bool NoGPULib = false)
      : OpenMPTargetDebug(OpenMPTargetDebug),
        OpenMPTeamSubscription(OpenMPTeamSubscription),
        OpenMPThreadSubscription(OpenMPThreadSubscription),
        OpenMPNoThreadState(OpenMPNoThreadState),
        OpenMPNoNestedParallelism(OpenMPNoNestedParallelism),
        OpenMPIsTargetDevice(OpenMPIsTargetDevice), OpenMPIsGPU(OpenMPIsGPU),
        OpenMPVersion(OpenMPVersion), OMPHostIRFile(OMPHostIRFile),
        NoGPULib(NoGPULib) {}

  OffloadModuleOpts(Fortran::frontend::LangOptions &Opts)
      : OpenMPTargetDebug(Opts.OpenMPTargetDebug),
        OpenMPTeamSubscription(Opts.OpenMPTeamSubscription),
        OpenMPThreadSubscription(Opts.OpenMPThreadSubscription),
        OpenMPNoThreadState(Opts.OpenMPNoThreadState),
        OpenMPNoNestedParallelism(Opts.OpenMPNoNestedParallelism),
        OpenMPIsTargetDevice(Opts.OpenMPIsTargetDevice),
        OpenMPIsGPU(Opts.OpenMPIsGPU), OpenMPVersion(Opts.OpenMPVersion),
        OMPHostIRFile(Opts.OMPHostIRFile), NoGPULib(Opts.NoGPULib) {}

  uint32_t OpenMPTargetDebug = 0;
  bool OpenMPTeamSubscription = false;
  bool OpenMPThreadSubscription = false;
  bool OpenMPNoThreadState = false;
  bool OpenMPNoNestedParallelism = false;
  bool OpenMPIsTargetDevice = false;
  bool OpenMPIsGPU = false;
  uint32_t OpenMPVersion = 11;
  std::string OMPHostIRFile = {};
  bool NoGPULib = false;
};

//  Shares assinging of the OpenMP OffloadModuleInterface and its assorted
//  attributes accross Flang tools (bbc/flang)
[[maybe_unused]] static void setOffloadModuleInterfaceAttributes(
    mlir::ModuleOp &module, OffloadModuleOpts Opts) {
  // Should be registered by the OpenMPDialect
  if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
          module.getOperation())) {
    offloadMod.setIsTargetDevice(Opts.OpenMPIsTargetDevice);
    offloadMod.setIsGPU(Opts.OpenMPIsGPU);
    if (Opts.OpenMPIsTargetDevice) {
      offloadMod.setFlags(Opts.OpenMPTargetDebug, Opts.OpenMPTeamSubscription,
          Opts.OpenMPThreadSubscription, Opts.OpenMPNoThreadState,
          Opts.OpenMPNoNestedParallelism, Opts.OpenMPVersion, Opts.NoGPULib);

      if (!Opts.OMPHostIRFile.empty())
        offloadMod.setHostIRFilePath(Opts.OMPHostIRFile);
    }
  }
}

[[maybe_unused]] static void setOpenMPVersionAttribute(
    mlir::ModuleOp &module, int64_t version) {
  module.getOperation()->setAttr(
      mlir::StringAttr::get(module.getContext(), llvm::Twine{"omp.version"}),
      mlir::omp::VersionAttr::get(module.getContext(), version));
}

[[maybe_unused]] static int64_t getOpenMPVersionAttribute(
    mlir::ModuleOp module, int64_t fallback = -1) {
  if (mlir::Attribute verAttr = module->getAttr("omp.version"))
    return llvm::cast<mlir::omp::VersionAttr>(verAttr).getVersion();
  return fallback;
}

#endif // FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
