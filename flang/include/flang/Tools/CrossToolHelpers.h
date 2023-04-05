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

#include "flang/Frontend/LangOptions.h"
#include <cstdint>

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"

struct OffloadModuleOpts {
  OffloadModuleOpts() {}
  OffloadModuleOpts(uint32_t OpenMPTargetDebug, bool OpenMPTeamSubscription,
      bool OpenMPThreadSubscription, bool OpenMPNoThreadState,
      bool OpenMPNoNestedParallelism, bool OpenMPIsDevice)
      : OpenMPTargetDebug(OpenMPTargetDebug),
        OpenMPTeamSubscription(OpenMPTeamSubscription),
        OpenMPThreadSubscription(OpenMPThreadSubscription),
        OpenMPNoThreadState(OpenMPNoThreadState),
        OpenMPNoNestedParallelism(OpenMPNoNestedParallelism),
        OpenMPIsDevice(OpenMPIsDevice) {}

  OffloadModuleOpts(Fortran::frontend::LangOptions &Opts)
      : OpenMPTargetDebug(Opts.OpenMPTargetDebug),
        OpenMPTeamSubscription(Opts.OpenMPTeamSubscription),
        OpenMPThreadSubscription(Opts.OpenMPThreadSubscription),
        OpenMPNoThreadState(Opts.OpenMPNoThreadState),
        OpenMPNoNestedParallelism(Opts.OpenMPNoNestedParallelism),
        OpenMPIsDevice(Opts.OpenMPIsDevice) {}

  uint32_t OpenMPTargetDebug = 0;
  bool OpenMPTeamSubscription = false;
  bool OpenMPThreadSubscription = false;
  bool OpenMPNoThreadState = false;
  bool OpenMPNoNestedParallelism = false;
  bool OpenMPIsDevice = false;
};

//  Shares assinging of the OpenMP OffloadModuleInterface and its assorted
//  attributes accross Flang tools (bbc/flang)
void setOffloadModuleInterfaceAttributes(
    mlir::ModuleOp &module, OffloadModuleOpts Opts) {
  // Should be registered by the OpenMPDialect
  if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
          module.getOperation())) {
    offloadMod.setIsDevice(Opts.OpenMPIsDevice);
    if (Opts.OpenMPIsDevice) {
      offloadMod.setFlags(Opts.OpenMPTargetDebug, Opts.OpenMPTeamSubscription,
          Opts.OpenMPThreadSubscription, Opts.OpenMPNoThreadState,
          Opts.OpenMPNoNestedParallelism);
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

#endif // FORTRAN_TOOLS_CROSS_TOOL_HELPERS_H
