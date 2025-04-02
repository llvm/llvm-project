//===-- CUFDeviceGlobal.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/allocatable.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"

namespace fir {
#define GEN_PASS_DEF_CUFDEVICEGLOBAL
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static void processAddrOfOp(fir::AddrOfOp addrOfOp,
                            mlir::SymbolTable &symbolTable,
                            llvm::DenseSet<fir::GlobalOp> &candidates,
                            bool recurseInGlobal) {
  if (auto globalOp = symbolTable.lookup<fir::GlobalOp>(
          addrOfOp.getSymbol().getRootReference().getValue())) {
    // TO DO: limit candidates to non-scalars. Scalars appear to have been
    // folded in already.
    if (recurseInGlobal)
      globalOp.walk([&](fir::AddrOfOp op) {
        processAddrOfOp(op, symbolTable, candidates, recurseInGlobal);
      });
    candidates.insert(globalOp);
  }
}

static void processEmboxOp(fir::EmboxOp emboxOp, mlir::SymbolTable &symbolTable,
                           llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(
          fir::unwrapRefType(emboxOp.getMemref().getType()))) {
    if (auto globalOp = symbolTable.lookup<fir::GlobalOp>(
            fir::NameUniquer::getTypeDescriptorName(recTy.getName()))) {
      if (!candidates.contains(globalOp)) {
        globalOp.walk([&](fir::AddrOfOp op) {
          processAddrOfOp(op, symbolTable, candidates,
                          /*recurseInGlobal=*/true);
        });
        candidates.insert(globalOp);
      }
    }
  }
}

static void
prepareImplicitDeviceGlobals(mlir::func::FuncOp funcOp,
                             mlir::SymbolTable &symbolTable,
                             llvm::DenseSet<fir::GlobalOp> &candidates) {
  auto cudaProcAttr{
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName())};
  if (cudaProcAttr && cudaProcAttr.getValue() != cuf::ProcAttribute::Host) {
    funcOp.walk([&](fir::AddrOfOp op) {
      processAddrOfOp(op, symbolTable, candidates, /*recurseInGlobal=*/false);
    });
    funcOp.walk(
        [&](fir::EmboxOp op) { processEmboxOp(op, symbolTable, candidates); });
  }
}

class CUFDeviceGlobal : public fir::impl::CUFDeviceGlobalBase<CUFDeviceGlobal> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::ModuleOp mod = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!mod)
      return signalPassFailure();

    llvm::DenseSet<fir::GlobalOp> candidates;
    mlir::SymbolTable symTable(mod);
    mod.walk([&](mlir::func::FuncOp funcOp) {
      prepareImplicitDeviceGlobals(funcOp, symTable, candidates);
      return mlir::WalkResult::advance();
    });
    mod.walk([&](cuf::KernelOp kernelOp) {
      kernelOp.walk([&](fir::AddrOfOp addrOfOp) {
        processAddrOfOp(addrOfOp, symTable, candidates,
                        /*recurseInGlobal=*/false);
      });
    });

    // Copying the device global variable into the gpu module
    mlir::SymbolTable parentSymTable(mod);
    auto gpuMod = cuf::getOrCreateGPUModule(mod, parentSymTable);
    if (!gpuMod)
      return signalPassFailure();
    mlir::SymbolTable gpuSymTable(gpuMod);
    for (auto globalOp : mod.getOps<fir::GlobalOp>()) {
      if (cuf::isRegisteredDeviceGlobal(globalOp))
        candidates.insert(globalOp);
    }
    for (auto globalOp : candidates) {
      auto globalName{globalOp.getSymbol().getValue()};
      if (gpuSymTable.lookup<fir::GlobalOp>(globalName)) {
        break;
      }
      gpuSymTable.insert(globalOp->clone());
    }
  }
};
} // namespace
