//===-- CUFCommon.cpp - Shared functions between passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  mlir::Block::iterator insertPt(mod.getBodyRegion().front().end());
  symTab.insert(gpuMod, insertPt);
  return gpuMod;
}

bool cuf::isInCUDADeviceContext(mlir::Operation *op) {
  if (!op)
    return false;
  if (op->getParentOfType<cuf::KernelOp>() ||
      op->getParentOfType<mlir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    if (auto cudaProcAttr = funcOp->getAttrOfType<cuf::ProcAttributeAttr>(
            cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host;
    }
  }
  return false;
}

bool cuf::isRegisteredDeviceGlobal(fir::GlobalOp op) {
  if (op.getConstant())
    return false;
  auto attr = op.getDataAttr();
  if (attr && (*attr == cuf::DataAttribute::Device ||
               *attr == cuf::DataAttribute::Managed ||
               *attr == cuf::DataAttribute::Constant))
    return true;
  return false;
}

void cuf::genPointerSync(const mlir::Value box, fir::FirOpBuilder &builder) {
  if (auto declareOp = box.getDefiningOp<hlfir::DeclareOp>()) {
    if (auto addrOfOp = declareOp.getMemref().getDefiningOp<fir::AddrOfOp>()) {
      auto mod = addrOfOp->getParentOfType<mlir::ModuleOp>();
      if (auto globalOp =
              mod.lookupSymbol<fir::GlobalOp>(addrOfOp.getSymbol())) {
        if (cuf::isRegisteredDeviceGlobal(globalOp)) {
          builder.create<cuf::SyncDescriptorOp>(box.getLoc(),
                                                addrOfOp.getSymbol());
        }
      }
    }
  }
}
