//===-- CUFCommon.cpp - Shared functions between passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUFCommon.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

/// Retrieve or create the CUDA Fortran GPU module in the give in \p mod.
mlir::gpu::GPUModuleOp cuf::getOrCreateGPUModule(mlir::ModuleOp mod,
                                                 mlir::SymbolTable &symTab) {
  if (auto gpuMod = symTab.lookup<mlir::gpu::GPUModuleOp>(cudaDeviceModuleName))
    return gpuMod;

  auto *ctx = mod.getContext();
  mod->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
               mlir::UnitAttr::get(ctx));

  mlir::OpBuilder builder(ctx);
  auto gpuMod = builder.create<mlir::gpu::GPUModuleOp>(mod.getLoc(),
                                                       cudaDeviceModuleName);
  llvm::SmallVector<mlir::Attribute> targets;
  targets.push_back(mlir::NVVM::NVVMTargetAttr::get(ctx));
  gpuMod.setTargetsAttr(builder.getArrayAttr(targets));
  mlir::Block::iterator insertPt(mod.getBodyRegion().front().end());
  symTab.insert(gpuMod, insertPt);
  return gpuMod;
}
