//===-- CUFDeviceGlobal.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace fir {
#define GEN_PASS_DEF_CUFDEVICEGLOBAL
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static void processAddrOfOp(fir::AddrOfOp addrOfOp,
                            mlir::SymbolTable &symbolTable,
                            llvm::DenseSet<fir::GlobalOp> &candidates,
                            bool recurseInGlobal,
                            bool skipDeadDeclares = true) {

  // Skip globals whose only reference is a dead fir.declare (no real uses).
  // This is disabled when fir.declare ops are preserved for debug info,
  // because later passes will copy the entire function body (including dead
  // references) into GPU kernels.
  if (skipDeadDeclares && addrOfOp.getOperation()->hasOneUse()) {
    mlir::OpOperand &addrUse = *addrOfOp.getOperation()->getUses().begin();
    if (mlir::isa<fir::DeclareOp>(addrUse.getOwner()) &&
        addrUse.getOwner()->use_empty())
      return;
  }

  if (auto globalOp = symbolTable.lookup<fir::GlobalOp>(
          addrOfOp.getSymbol().getRootReference().getValue())) {
    // TO DO: limit candidates to non-scalars. Scalars appear to have been
    // folded in already.
    // Insert before recursing so cycles among globals (e.g. mutually
    // referencing type descriptors) do not cause infinite recursion.
    if (!candidates.insert(globalOp).second)
      return;
    if (recurseInGlobal)
      globalOp.walk([&](fir::AddrOfOp op) {
        processAddrOfOp(op, symbolTable, candidates, recurseInGlobal);
      });
  }
}

static void processTypeDescriptor(fir::RecordType recTy,
                                  mlir::SymbolTable &symbolTable,
                                  llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto globalOp = symbolTable.lookup<fir::GlobalOp>(
          fir::NameUniquer::getTypeDescriptorName(recTy.getName()))) {
    // Insert before walking so cyclic addr_of chains terminate.
    if (!candidates.insert(globalOp).second)
      return;
    globalOp.walk([&](fir::AddrOfOp op) {
      processAddrOfOp(op, symbolTable, candidates,
                      /*recurseInGlobal=*/true);
    });
  }
}

static void processAllocaOp(fir::AllocaOp allocaOp,
                            mlir::SymbolTable &symbolTable,
                            llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(allocaOp.getInType()))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

static void processEmboxOp(fir::EmboxOp emboxOp, mlir::SymbolTable &symbolTable,
                           llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(
          fir::unwrapRefType(emboxOp.getMemref().getType())))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

static void prepareImplicitDeviceGlobals(
    mlir::func::FuncOp funcOp, mlir::SymbolTable &symbolTable,
    llvm::DenseSet<fir::GlobalOp> &candidates, bool skipDeadDeclares) {
  auto cudaProcAttr{
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName())};
  if (cudaProcAttr && cudaProcAttr.getValue() != cuf::ProcAttribute::Host) {
    funcOp.walk([&](fir::AddrOfOp op) {
      processAddrOfOp(op, symbolTable, candidates, /*recurseInGlobal=*/false,
                      skipDeadDeclares);
    });
    funcOp.walk(
        [&](fir::EmboxOp op) { processEmboxOp(op, symbolTable, candidates); });
    funcOp.walk([&](fir::AllocaOp op) {
      processAllocaOp(op, symbolTable, candidates);
    });
  }
}

static void
processPotentialTypeDescriptor(mlir::Type candidateType,
                               mlir::SymbolTable &symbolTable,
                               llvm::DenseSet<fir::GlobalOp> &candidates) {
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(candidateType))
    candidateType = boxTy.getEleTy();
  candidateType = fir::unwrapSequenceType(fir::unwrapRefType(candidateType));
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(candidateType))
    processTypeDescriptor(recTy, symbolTable, candidates);
}

/// NVPTX cannot emit global initializers that form a reference cycle (see
/// VisitGlobalVariableForEmission). Fortran type-info globals often do
/// (mutually recursive derived types). Make a subset of the GPU copies extern
/// declarations so that the remaining initializer dependency graph is
/// acyclic, while preserving as many complete initializers as possible.
static void dropCyclicGlobalInitializers(mlir::gpu::GPUModuleOp gpuMod) {
  llvm::DenseMap<llvm::StringRef, fir::GlobalOp> byName;
  llvm::SmallVector<fir::GlobalOp, 16> globals;
  for (auto global : gpuMod.getOps<fir::GlobalOp>()) {
    byName[global.getSymName()] = global;
    globals.push_back(global);
  }
  if (globals.empty())
    return;

  llvm::sort(globals, [](fir::GlobalOp lhs, fir::GlobalOp rhs) {
    return lhs.getSymName() < rhs.getSymName();
  });

  // Adjacency: global -> targets referenced via fir.address_of in its body.
  llvm::DenseMap<fir::GlobalOp, llvm::SmallVector<fir::GlobalOp, 4>> adj;
  for (fir::GlobalOp global : globals) {
    global.walk([&](fir::AddrOfOp addrOf) {
      fir::GlobalOp target =
          byName.lookup(addrOf.getSymbol().getRootReference().getValue());
      if (!target)
        return;
      adj[global].push_back(target);
    });
  }

  // Greedily construct a feedback vertex set. Dropping one initializer
  // removes all outgoing dependency edges from that global. Restart the DFS
  // after each cut until no back edge remains.
  llvm::DenseSet<fir::GlobalOp> declarationOnly;
  while (true) {
    // 0: unvisited, 1: active, 2: complete.
    llvm::DenseMap<fir::GlobalOp, unsigned> state;
    fir::GlobalOp cut;
    std::function<bool(fir::GlobalOp)> findCycle =
        [&](fir::GlobalOp global) -> bool {
      if (declarationOnly.contains(global))
        return false;
      state[global] = 1;
      for (fir::GlobalOp target : adj.lookup(global)) {
        if (declarationOnly.contains(target))
          continue;
        if (state.lookup(target) == 1) {
          cut = global;
          return true;
        }
        if (state.lookup(target) == 0 && findCycle(target))
          return true;
      }
      state[global] = 2;
      return false;
    };

    for (fir::GlobalOp global : globals)
      if (state.lookup(global) == 0 && findCycle(global))
        break;
    if (!cut)
      break;
    declarationOnly.insert(cut);
  }

  for (fir::GlobalOp global : declarationOnly) {
    global.getRegion().getBlocks().clear();
    global.removeInitValAttr();
    // No initializer: use default external linkage so NVPTX emits
    // `.extern .global` with no initializer dependency edges.
    global.removeLinkNameAttr();
  }
}

class CUFDeviceGlobal : public fir::impl::CUFDeviceGlobalBase<CUFDeviceGlobal> {
public:
  using CUFDeviceGlobalBase::CUFDeviceGlobalBase;

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::ModuleOp mod = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!mod)
      return signalPassFailure();

    llvm::DenseSet<fir::GlobalOp> candidates;
    mlir::SymbolTable symTable(mod);
    mod.walk([&](mlir::func::FuncOp funcOp) {
      prepareImplicitDeviceGlobals(funcOp, symTable, candidates,
                                   skipDeadDeclares);
      return mlir::WalkResult::advance();
    });
    mod.walk([&](cuf::KernelOp kernelOp) {
      kernelOp.walk([&](fir::AddrOfOp addrOfOp) {
        processAddrOfOp(addrOfOp, symTable, candidates,
                        /*recurseInGlobal=*/false, skipDeadDeclares);
      });
    });

    // Copying the device global variable into the gpu module
    mlir::SymbolTable parentSymTable(mod);
    auto gpuMod = cuf::getOrCreateGPUModule(mod, parentSymTable);
    if (!gpuMod)
      return signalPassFailure();
    mlir::SymbolTable gpuSymTable(gpuMod);
    for (auto globalOp : mod.getOps<fir::GlobalOp>()) {
      if (cuf::isRegisteredDeviceGlobal(globalOp)) {
        candidates.insert(globalOp);
        processPotentialTypeDescriptor(globalOp.getType(), parentSymTable,
                                       candidates);
      } else if (globalOp.getConstant() &&
                 mlir::isa<fir::SequenceType>(
                     fir::unwrapRefType(globalOp.resultType()))) {
        mlir::Attribute initAttr =
            globalOp.getInitVal().value_or(mlir::Attribute());
        if (initAttr && mlir::dyn_cast<mlir::DenseElementsAttr>(initAttr))
          candidates.insert(globalOp);
      }
    }
    for (auto globalOp : candidates) {
      auto globalName{globalOp.getSymbol().getValue()};
      if (gpuSymTable.lookup<fir::GlobalOp>(globalName)) {
        continue;
      }
      auto *cloned = globalOp->clone();
      auto clonedGlobal = mlir::cast<fir::GlobalOp>(cloned);
      // Under -gpu=mem:unified, plain host module-scope variables (no
      // explicit CUF data attribute, not a constant) get a no-body
      // declaration in the GPU module: clear the body, init value, and
      // linkName. With no linkName, the LLVM lowering uses the default
      // External linkage (see convertLinkage in CodeGen.cpp), so an
      // initializer-less global emits as `.extern .global ...` in PTX.
      // The host-side definition stays. CUFAddConstructor will emit
      // CUFRegisterExternalVariable (= __cudaRegisterHostVar) so the CUDA
      // runtime maps the device extern to the host pointer at module-load
      // time, and HMM/ATS handles migration.
      if (cudaUnified && !globalOp.getConstant() &&
          !globalOp.getDataAttrAttr()) {
        clonedGlobal.getRegion().getBlocks().clear();
        clonedGlobal.removeInitValAttr();
        clonedGlobal.removeLinkNameAttr();
      }
      // Registered CUDA globals with internal linkage must have a visible
      // device symbol so runtime lookups (cudaGetSymbolAddress) can resolve
      // them. Drop internal linkage from the GPU clone so it uses default
      // external linkage.
      if (cuf::isRegisteredDeviceGlobal(globalOp) &&
          globalOp.getLinkName() == "internal")
        clonedGlobal.removeLinkNameAttr();
      gpuSymTable.insert(cloned);
    }
    // Type-info globals for mutually recursive derived types form initializer
    // cycles; NVPTX rejects those. Drop initializers from cyclic GPU copies.
    dropCyclicGlobalInitializers(gpuMod);
  }
};
} // namespace
