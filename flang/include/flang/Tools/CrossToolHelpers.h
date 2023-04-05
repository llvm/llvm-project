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

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"

//  Shares assinging of the OpenMP OffloadModuleInterface and its assorted
//  attributes accross Flang tools (bbc/flang)
void setOffloadModuleInterfaceAttributes(
    mlir::ModuleOp &module, bool isDevice) {
  // Should be registered by the OpenMPDialect
  if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
          module.getOperation())) {
    offloadMod.setIsDevice(isDevice);
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
