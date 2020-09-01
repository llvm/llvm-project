//===- MemRefDataFlowOpt.cpp - Memory DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "flang-memref-dataflow-opt"

namespace {

template <typename Trait>
llvm::SmallVector<mlir::Operation *, 8>
getParentOpsWithTrait(mlir::Operation *op) {
  llvm::SmallVector<mlir::Operation *, 8> parentLoops;
  while ((op = op->getParentOp())) {
    if (op->hasTrait<Trait>())
      parentLoops.push_back(op);
  }
  return parentLoops;
}

#if 0
unsigned getNumCommonSurroundingOps(
    const llvm::SmallVectorImpl<mlir::Operation *> OpsA,
    const llvm::SmallVectorImpl<mlir::Operation *> OpsB) {
  unsigned numCommonOps = 0;
  unsigned minNumOps = std::min(OpsA.size(), OpsB.size());
  for (unsigned i = 0; i < minNumOps; ++i) {
    if (OpsA[i] != OpsB[i])
      break;
    numCommonOps++;
  }
  return numCommonOps;
}
#endif

/// This is based on MLIR's MemRefDataFlowOpt which is specialized on AffineRead
/// and AffineWrite interface
template <typename ReadOp, typename WriteOp>
class LoadStoreForwarding {
public:
  LoadStoreForwarding(mlir::DominanceInfo *di, mlir::PostDominanceInfo *pdi)
      : domInfo(di), postDomInfo(pdi) {}
  llvm::Optional<WriteOp>
  findStoreToForward(ReadOp loadOp, llvm::SmallVectorImpl<WriteOp> &&storeOps) {
    llvm::SmallVector<Operation *, 8> forwadingCandidates;
    llvm::SmallVector<Operation *, 8> storesWithDependence;

    for (auto &storeOp : storeOps) {
      if (accessDependence(loadOp, storeOp))
        storesWithDependence.push_back(storeOp.getOperation());
      if (equivalentAccess(loadOp, storeOp) &&
          domInfo->dominates(storeOp.getOperation(), loadOp.getOperation()))
        forwadingCandidates.push_back(storeOp.getOperation());
    }

    llvm::Optional<WriteOp> lastWriteStoreOp;
    for (auto *storeOp : forwadingCandidates) {
      if (llvm::all_of(storesWithDependence, [&](mlir::Operation *depStore) {
            return postDomInfo->postDominates(storeOp, depStore);
          })) {
        lastWriteStoreOp = cast<WriteOp>(storeOp);
        break;
      }
    }
    return lastWriteStoreOp;
  }
  llvm::Optional<ReadOp>
  findReadForWrite(WriteOp storeOp, llvm::SmallVectorImpl<ReadOp> &&loadOps) {
    llvm::SmallVector<Operation *, 8> useCandidates;
    llvm::SmallVector<Operation *, 8> dependences;
    for (auto &loadOp : loadOps) {
      if (equivalentAccess(loadOp, storeOp) &&
          postDomInfo->postDominates(loadOp, storeOp))
        return {loadOp};
    }
    return {};
  }
  bool equivalentAccess(ReadOp loadOp, WriteOp storeOp) { return true; }
  bool accessDependence(ReadOp loadOp, WriteOp storeOp) { return true; }

private:
  mlir::DominanceInfo *domInfo;
  mlir::PostDominanceInfo *postDomInfo;
};

template <typename OpT>
llvm::SmallVector<OpT, 8> getSpecificUsers(mlir::Value v) {
  llvm::SmallVector<OpT, 8> ops;
  for (auto *user : v.getUsers()) {
    if (auto op = dyn_cast<OpT>(user))
      ops.push_back(op);
  }
  return ops;
}

class MemDataFlowOpt : public fir::MemRefDataFlowOptBase<MemDataFlowOpt> {
public:
  void runOnFunction() override {
    mlir::FuncOp f = getFunction();

    auto domInfo = &getAnalysis<mlir::DominanceInfo>();
    auto postDomInfo = &getAnalysis<mlir::PostDominanceInfo>();
    LoadStoreForwarding<fir::LoadOp, fir::StoreOp> lsf(domInfo, postDomInfo);
    f.walk([&](fir::LoadOp loadOp) {
      auto maybeStore = lsf.findStoreToForward(
          loadOp, getSpecificUsers<fir::StoreOp>(loadOp.memref()));
      if (maybeStore) {
        LLVM_DEBUG(llvm::dbgs() << "FlangMemDataFlowOpt: erasing loadOp with "
                                   "value from store\n";
                   loadOp.dump(); maybeStore.getValue().dump(););
        loadOp.getResult().replaceAllUsesWith(maybeStore.getValue().value());
        loadOp.erase();
      }
    });
    f.walk([&](fir::AllocaOp alloca) {
      for (auto &storeOp : getSpecificUsers<fir::StoreOp>(alloca.getResult())) {
        if (!lsf.findReadForWrite(
                storeOp, getSpecificUsers<fir::LoadOp>(storeOp.memref())))
          storeOp.erase();
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createMemDataFlowOptPass() {
  return std::make_unique<MemDataFlowOpt>();
}
