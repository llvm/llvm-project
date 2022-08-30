//===- MemRefDataFlowOpt.cpp - Memory DataFlow Optimization pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_MEMREFDATAFLOWOPTPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fir-memref-dataflow-opt"

using namespace mlir;

namespace {

template <typename OpT>
static std::vector<OpT> getSpecificUsers(mlir::Value v) {
  std::vector<OpT> ops;
  for (mlir::Operation *user : v.getUsers())
    if (auto op = dyn_cast<OpT>(user))
      ops.push_back(op);
  return ops;
}

/// This is based on MLIR's MemRefDataFlowOpt which is specialized on AffineRead
/// and AffineWrite interface
template <typename ReadOp, typename WriteOp>
class LoadStoreForwarding {
public:
  LoadStoreForwarding(mlir::DominanceInfo *di) : domInfo(di) {}

  // FIXME: This algorithm has a bug. It ignores escaping references between a
  // store and a load.
  llvm::Optional<WriteOp> findStoreToForward(ReadOp loadOp,
                                             std::vector<WriteOp> &&storeOps) {
    llvm::SmallVector<WriteOp> candidateSet;

    for (auto storeOp : storeOps)
      if (domInfo->dominates(storeOp, loadOp))
        candidateSet.push_back(storeOp);

    if (candidateSet.empty())
      return {};

    llvm::Optional<WriteOp> nearestStore;
    for (auto candidate : candidateSet) {
      auto nearerThan = [&](WriteOp otherStore) {
        if (candidate == otherStore)
          return false;
        bool rv = domInfo->properlyDominates(candidate, otherStore);
        if (rv) {
          LLVM_DEBUG(llvm::dbgs()
                     << "candidate " << candidate << " is not the nearest to "
                     << loadOp << " because " << otherStore << " is closer\n");
        }
        return rv;
      };
      if (!llvm::any_of(candidateSet, nearerThan)) {
        nearestStore = mlir::cast<WriteOp>(candidate);
        break;
      }
    }
    if (!nearestStore) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "load " << loadOp << " has " << candidateSet.size()
          << " store candidates, but this algorithm can't find a best.\n");
    }
    return nearestStore;
  }

  llvm::Optional<ReadOp> findReadForWrite(WriteOp storeOp,
                                          std::vector<ReadOp> &&loadOps) {
    for (auto &loadOp : loadOps) {
      if (domInfo->dominates(storeOp, loadOp))
        return loadOp;
    }
    return {};
  }

private:
  mlir::DominanceInfo *domInfo;
};

class MemDataFlowOpt
    : public fir::impl::MemRefDataFlowOptPassBase<MemDataFlowOpt> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();

    auto *domInfo = &getAnalysis<mlir::DominanceInfo>();
    LoadStoreForwarding<fir::LoadOp, fir::StoreOp> lsf(domInfo);
    f.walk([&](fir::LoadOp loadOp) {
      auto maybeStore = lsf.findStoreToForward(
          loadOp, getSpecificUsers<fir::StoreOp>(loadOp.getMemref()));
      if (maybeStore) {
        auto storeOp = *maybeStore;
        LLVM_DEBUG(llvm::dbgs() << "FlangMemDataFlowOpt: In " << f.getName()
                                << " erasing load " << loadOp
                                << " with value from " << storeOp << '\n');
        loadOp.getResult().replaceAllUsesWith(storeOp.getValue());
        loadOp.erase();
      }
    });
    f.walk([&](fir::AllocaOp alloca) {
      for (auto &storeOp : getSpecificUsers<fir::StoreOp>(alloca.getResult())) {
        if (!lsf.findReadForWrite(
                storeOp, getSpecificUsers<fir::LoadOp>(storeOp.getMemref()))) {
          LLVM_DEBUG(llvm::dbgs() << "FlangMemDataFlowOpt: In " << f.getName()
                                  << " erasing store " << storeOp << '\n');
          storeOp.erase();
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createMemDataFlowOptPass() {
  return std::make_unique<MemDataFlowOpt>();
}
