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

#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/LangOptions.h"
#include <cstdint>

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Passes/OptimizationLevel.h"

/// Configuriation for the MLIR to LLVM pass pipeline.
struct MLIRToLLVMPassPipelineConfig {
  explicit MLIRToLLVMPassPipelineConfig(llvm::OptimizationLevel level) {
    OptLevel = level;
  }
  explicit MLIRToLLVMPassPipelineConfig(llvm::OptimizationLevel level,
      const Fortran::frontend::CodeGenOptions &opts) {
    OptLevel = level;
    StackArrays = opts.StackArrays;
    Underscoring = opts.Underscoring;
    LoopVersioning = opts.LoopVersioning;
    DebugInfo = opts.getDebugInfo();
  }

  llvm::OptimizationLevel OptLevel; ///< optimisation level
  bool StackArrays = false; ///< convert memory allocations to alloca.
  bool Underscoring = true; ///< add underscores to function names.
  bool LoopVersioning = false; ///< Run the version loop pass.
  llvm::codegenoptions::DebugInfoKind DebugInfo =
      llvm::codegenoptions::NoDebugInfo; ///< Debug info generation.
};

struct OffloadModuleOpts {
  OffloadModuleOpts() {}
  OffloadModuleOpts(uint32_t OpenMPTargetDebug, bool OpenMPTeamSubscription,
      bool OpenMPThreadSubscription, bool OpenMPNoThreadState,
      bool OpenMPNoNestedParallelism, bool OpenMPIsTargetDevice,
      bool OpenMPIsGPU, uint32_t OpenMPVersion, std::string OMPHostIRFile = {})
      : OpenMPTargetDebug(OpenMPTargetDebug),
        OpenMPTeamSubscription(OpenMPTeamSubscription),
        OpenMPThreadSubscription(OpenMPThreadSubscription),
        OpenMPNoThreadState(OpenMPNoThreadState),
        OpenMPNoNestedParallelism(OpenMPNoNestedParallelism),
        OpenMPIsTargetDevice(OpenMPIsTargetDevice), OpenMPIsGPU(OpenMPIsGPU),
        OpenMPVersion(OpenMPVersion), OMPHostIRFile(OMPHostIRFile) {}

  OffloadModuleOpts(Fortran::frontend::LangOptions &Opts)
      : OpenMPTargetDebug(Opts.OpenMPTargetDebug),
        OpenMPTeamSubscription(Opts.OpenMPTeamSubscription),
        OpenMPThreadSubscription(Opts.OpenMPThreadSubscription),
        OpenMPNoThreadState(Opts.OpenMPNoThreadState),
        OpenMPNoNestedParallelism(Opts.OpenMPNoNestedParallelism),
        OpenMPIsTargetDevice(Opts.OpenMPIsTargetDevice),
        OpenMPIsGPU(Opts.OpenMPIsGPU), OpenMPVersion(Opts.OpenMPVersion),
        OMPHostIRFile(Opts.OMPHostIRFile) {}

  uint32_t OpenMPTargetDebug = 0;
  bool OpenMPTeamSubscription = false;
  bool OpenMPThreadSubscription = false;
  bool OpenMPNoThreadState = false;
  bool OpenMPNoNestedParallelism = false;
  bool OpenMPIsTargetDevice = false;
  bool OpenMPIsGPU = false;
  uint32_t OpenMPVersion = 11;
  std::string OMPHostIRFile = {};
};

//  Shares assinging of the OpenMP OffloadModuleInterface and its assorted
//  attributes accross Flang tools (bbc/flang)
void setOffloadModuleInterfaceAttributes(
    mlir::ModuleOp &module, OffloadModuleOpts Opts) {
  // Should be registered by the OpenMPDialect
  if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
          module.getOperation())) {
    offloadMod.setIsTargetDevice(Opts.OpenMPIsTargetDevice);
    offloadMod.setIsGPU(Opts.OpenMPIsGPU);
    if (Opts.OpenMPIsTargetDevice) {
      offloadMod.setFlags(Opts.OpenMPTargetDebug, Opts.OpenMPTeamSubscription,
          Opts.OpenMPThreadSubscription, Opts.OpenMPNoThreadState,
          Opts.OpenMPNoNestedParallelism, Opts.OpenMPVersion);

      if (!Opts.OMPHostIRFile.empty())
        offloadMod.setHostIRFilePath(Opts.OMPHostIRFile);
    }
  }
}

//  Shares assinging of the OpenMP OffloadModuleInterface and its TargetCPU
//  attribute accross Flang tools (bbc/flang)
void setOffloadModuleInterfaceTargetAttribute(mlir::ModuleOp &module,
    llvm::StringRef targetCPU, llvm::StringRef targetFeatures) {
  // Should be registered by the OpenMPDialect
  if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
          module.getOperation())) {
    offloadMod.setTarget(targetCPU, targetFeatures);
  }
}

void setOpenMPVersionAttribute(mlir::ModuleOp &module, int64_t version) {
  module.getOperation()->setAttr(
      mlir::StringAttr::get(module.getContext(), llvm::Twine{"omp.version"}),
      mlir::omp::VersionAttr::get(module.getContext(), version));
}

#endif // FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
