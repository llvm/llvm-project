//===-- CUFDeviceGlobal.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Support/Fortran.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"

namespace fir {
#define GEN_PASS_DEF_CUFDEVICEGLOBAL
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static void processAddrOfOp(fir::AddrOfOp addrOfOp,
                            aiir::SymbolTable &symbolTable,
                            llvm::DenseSet<fir::GlobalOp> &candidates,
                            bool recurseInGlobal) {

  // Check if there is a real use of the global.
  if (addrOfOp.getOperation()->hasOneUse()) {
    aiir::OpOperand &addrUse = *addrOfOp.getOperation()->getUses().begin();
    if (aiir::isa<fir::DeclareOp>(addrUse.getOwner()) &&
        addrUse.getOwner()->use_empty())
      return;
  }

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

static void processTypeDescriptor(fir::RecordType recTy,
                                  aiir::SymbolTable &symbolTable,
                                  llvm::DenseSet<fir::GlobalOp> &candidates) {
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

static void processAllocaOp(fir::AllocaOp allocaOp,
                            aiir::SymbolTable &symbolTable,
                            llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(allocaOp.getInType()))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

static void processEmboxOp(fir::EmboxOp emboxOp, aiir::SymbolTable &symbolTable,
                           llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(
          fir::unwrapRefType(emboxOp.getMemref().getType())))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

static void
prepareImplicitDeviceGlobals(aiir::func::FuncOp funcOp,
                             aiir::SymbolTable &symbolTable,
                             llvm::DenseSet<fir::GlobalOp> &candidates) {
  auto cudaProcAttr{
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName())};
  if (cudaProcAttr && cudaProcAttr.getValue() != cuf::ProcAttribute::Host) {
    funcOp.walk([&](fir::AddrOfOp op) {
      processAddrOfOp(op, symbolTable, candidates, /*recurseInGlobal=*/false);
    });
    funcOp.walk(
        [&](fir::EmboxOp op) { processEmboxOp(op, symbolTable, candidates); });
    funcOp.walk([&](fir::AllocaOp op) {
      processAllocaOp(op, symbolTable, candidates);
    });
  }
}

static void
processPotentialTypeDescriptor(aiir::Type candidateType,
                               aiir::SymbolTable &symbolTable,
                               llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(candidateType))
    candidateType = boxTy.getEleTy();
  candidateType = fir::unwrapSequenceType(fir::unwrapRefType(candidateType));
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(candidateType))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

class CUFDeviceGlobal : public fir::impl::CUFDeviceGlobalBase<CUFDeviceGlobal> {
public:
  void runOnOperation() override {
    aiir::Operation *op = getOperation();
    aiir::ModuleOp mod = aiir::dyn_cast<aiir::ModuleOp>(op);
    if (!mod)
      return signalPassFailure();

    llvm::DenseSet<fir::GlobalOp> candidates;
    aiir::SymbolTable symTable(mod);
    mod.walk([&](aiir::func::FuncOp funcOp) {
      prepareImplicitDeviceGlobals(funcOp, symTable, candidates);
      return aiir::WalkResult::advance();
    });
    mod.walk([&](cuf::KernelOp kernelOp) {
      kernelOp.walk([&](fir::AddrOfOp addrOfOp) {
        processAddrOfOp(addrOfOp, symTable, candidates,
                        /*recurseInGlobal=*/false);
      });
    });

    // Copying the device global variable into the gpu module
    aiir::SymbolTable parentSymTable(mod);
    auto gpuMod = cuf::getOrCreateGPUModule(mod, parentSymTable);
    if (!gpuMod)
      return signalPassFailure();
    aiir::SymbolTable gpuSymTable(gpuMod);
    for (auto globalOp : mod.getOps<fir::GlobalOp>()) {
      if (cuf::isRegisteredDeviceGlobal(globalOp)) {
        candidates.insert(globalOp);
        processPotentialTypeDescriptor(globalOp.getType(), parentSymTable,
                                       candidates);
      } else if (globalOp.getConstant() &&
                 aiir::isa<fir::SequenceType>(
                     fir::unwrapRefType(globalOp.resultType()))) {
        aiir::Attribute initAttr =
            globalOp.getInitVal().value_or(aiir::Attribute());
        if (initAttr && aiir::dyn_cast<aiir::DenseElementsAttr>(initAttr))
          candidates.insert(globalOp);
      }
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
