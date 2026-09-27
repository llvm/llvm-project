//===-- CUFDuplicateDeviceFunc.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFDUPLICATEDEVICEFUNC
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

class CUFDuplicateDeviceFunc
    : public fir::impl::CUFDuplicateDeviceFuncBase<CUFDuplicateDeviceFunc> {
  using CUFDuplicateDeviceFuncBase<
      CUFDuplicateDeviceFunc>::CUFDuplicateDeviceFuncBase;

  static bool isHostDevice(mlir::func::FuncOp funcOp) {
    auto procAttr =
        funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName());
    return procAttr && procAttr.getValue() == cuf::ProcAttribute::HostDevice;
  }

  /// Create the device copy of \p funcOp: same body, device proc attribute, the
  /// device allocation policy, and a marker naming the original.
  static mlir::func::FuncOp createDeviceCopy(mlir::func::FuncOp funcOp,
                                             llvm::StringRef copyName,
                                             mlir::SymbolTable &symTab) {
    auto copy = mlir::cast<mlir::func::FuncOp>(funcOp->clone());
    copy.setSymName(copyName);
    copy->setAttr(cuf::getProcAttrName(),
                  cuf::ProcAttributeAttr::get(funcOp.getContext(),
                                              cuf::ProcAttribute::Device));
    cuf::setDeviceCopyOf(copy, funcOp.getSymName());
    cuf::setDeviceAllocationPolicy(copy.getOperation());
    symTab.insert(copy,
                  std::next(mlir::Block::iterator(funcOp.getOperation())));
    return copy;
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::SymbolTable symTab(mod);
    cuf::DeviceCodeSet code = cuf::collectDeviceCode(mod, symTab);

    // Procedures that need a copy of their own on the device: host_device ones,
    // whose original stays host code, and the procedures without a device
    // attribute that device code reaches. Declarations have nothing to
    // optimize; cuf-transform-device-func clones them by name as before.
    llvm::SetVector<mlir::func::FuncOp> toCopy;
    for (mlir::func::FuncOp funcOp : code.deviceFuncs)
      if (isHostDevice(funcOp) && !funcOp.isDeclaration())
        toCopy.insert(funcOp);
    for (mlir::func::FuncOp funcOp : code.calledFromDevice)
      if (!funcOp.isDeclaration())
        toCopy.insert(funcOp);
    if (toCopy.empty())
      return;

    llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> copyOf;
    llvm::SmallVector<mlir::func::FuncOp> copies;
    for (mlir::func::FuncOp funcOp : toCopy) {
      std::string copyName = (funcOp.getSymName() + cudaDeviceCopySuffix).str();
      auto copy = symTab.lookup<mlir::func::FuncOp>(copyName);
      if (copy && cuf::getDeviceCopyOf(copy) != funcOp.getSymName()) {
        funcOp.emitError("cannot create the device copy of this procedure: "
                         "the symbol '")
            << copyName << "' is already taken";
        signalPassFailure();
        return;
      }
      if (!copy)
        copy = createDeviceCopy(funcOp, copyName, symTab);
      copies.push_back(copy);
      copyOf[funcOp.getSymNameAttr()] =
          mlir::FlatSymbolRefAttr::get(copy.getSymNameAttr());
    }

    // Device code refers to the copies, host code keeps the originals. Device
    // code is every device procedure except the host_device originals, the
    // copies themselves, and the cuf.kernel regions.
    for (mlir::func::FuncOp funcOp : code.deviceFuncs)
      if (!isHostDevice(funcOp))
        cuf::remapProcedureSymbols(funcOp, copyOf);
    for (mlir::func::FuncOp copy : copies)
      cuf::remapProcedureSymbols(copy, copyOf);
    mod.walk([&](cuf::KernelOp kernelOp) {
      cuf::remapProcedureSymbols(kernelOp, copyOf);
    });
  }
};

} // end anonymous namespace
