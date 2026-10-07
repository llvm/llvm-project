//===-- CUFAllocDelay.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Delay cuf.alloc of descriptor (box) types from function entry to just before
// their first use, possibly in a later block that dominates every use. This
// defers cudaMallocManaged calls so that users can call cudaSetDevice before
// any CUDA context is created.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFALLOCDELAY
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

/// Return the coordinate_of producing the host association tuple slot that
/// \p storeOp writes \p descriptor into, or null if this is not such a capture.
static fir::CoordinateOp getHostAssocTupleSlot(fir::StoreOp storeOp,
                                               mlir::Value descriptor) {
  if (storeOp.getValue() != descriptor ||
      !mlir::isa<fir::LLVMPointerType>(storeOp.getMemref().getType()))
    return nullptr;
  auto coord = storeOp.getMemref().getDefiningOp<fir::CoordinateOp>();
  if (!coord ||
      !mlir::isa<mlir::TupleType>(fir::unwrapRefType(coord.getRef().getType())))
    return nullptr;
  return coord;
}

/// Find the point before which the cuf.alloc group should be placed: the
/// earliest use in the block that dominates all uses, or that block's
/// terminator if it holds no use itself. Uses in nested regions resolve to
/// their enclosing top-level op.
///
/// Host association stores go to \p hostAssocStores and sink with the group
/// instead of constraining it; the tuple's readers constrain it instead.
static mlir::Operation *
findDelayTarget(fir::DeclareOp declareOp, mlir::Block *entryBlock,
                mlir::DominanceInfo &domInfo,
                llvm::SmallVectorImpl<fir::StoreOp> &hostAssocStores) {
  mlir::Region *funcRegion = entryBlock->getParent();

  // Uses resolved to an op that sits directly in a block of the function.
  llvm::SmallVector<mlir::Operation *> uses;

  auto recordRealUse = [&](mlir::Operation *user) {
    mlir::Operation *op = user;
    while (op && op->getBlock() && op->getBlock()->getParent() != funcRegion)
      op = op->getParentOp();
    if (op && op->getBlock())
      uses.push_back(op);
  };

  for (mlir::Value result : declareOp->getResults()) {
    for (mlir::Operation *user : result.getUsers()) {
      auto storeOp = mlir::dyn_cast<fir::StoreOp>(user);
      fir::CoordinateOp slot =
          storeOp ? getHostAssocTupleSlot(storeOp, result) : nullptr;
      if (!slot) {
        recordRealUse(user);
        continue;
      }
      // Whoever reads the tuple must still see a populated slot.
      hostAssocStores.push_back(storeOp);
      for (mlir::Operation *tupleUser : slot.getRef().getUsers()) {
        auto coord = mlir::dyn_cast<fir::CoordinateOp>(tupleUser);
        if (!coord) {
          recordRealUse(tupleUser);
          continue;
        }
        // A coordinate_of only computes the slot address. Stores through that
        // address populate the tuple; other users actually consume the slot.
        for (mlir::Operation *coordUser : coord->getUsers()) {
          auto slotStore = mlir::dyn_cast<fir::StoreOp>(coordUser);
          if (slotStore && slotStore.getMemref() == coord.getResult())
            continue;
          recordRealUse(coordUser);
        }
      }
    }
  }

  if (uses.empty())
    return nullptr;

  mlir::Block *common = uses.front()->getBlock();
  for (mlir::Operation *use : uses) {
    common = domInfo.findNearestCommonDominator(common, use->getBlock());
    if (!common)
      return nullptr;
  }

  mlir::Operation *earliest = nullptr;
  for (mlir::Operation *use : uses)
    if (use->getBlock() == common &&
        (!earliest || use->isBeforeInBlock(earliest)))
      earliest = use;

  return earliest ? earliest : common->getTerminator();
}

struct CUFAllocDelay : public fir::impl::CUFAllocDelayBase<CUFAllocDelay> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (func.empty())
      return;

    mlir::Block &entryBlock = func.front();
    mlir::DominanceInfo domInfo(func);

    // Collect box-type cuf.alloc ops in the entry block.
    llvm::SmallVector<cuf::AllocOp> boxAllocOps;
    for (auto &op : entryBlock)
      if (auto allocOp = mlir::dyn_cast<cuf::AllocOp>(op))
        if (mlir::isa<fir::BaseBoxType>(allocOp.getInType()))
          boxAllocOps.push_back(allocOp);

    for (cuf::AllocOp allocOp : boxAllocOps) {
      // Find the fir.declare and fir.store using this cuf.alloc; bail on any
      // unexpected user.
      fir::DeclareOp declareOp = nullptr;
      fir::StoreOp storeOp = nullptr;
      bool hasUnknownUser = false;
      for (mlir::Operation *user : allocOp->getUsers()) {
        if (auto decl = mlir::dyn_cast<fir::DeclareOp>(user))
          declareOp = decl;
        else if (auto store = mlir::dyn_cast<fir::StoreOp>(user))
          storeOp = store;
        else
          hasUnknownUser = true;
      }
      if (!declareOp || hasUnknownUser)
        continue;

      llvm::SmallVector<fir::StoreOp> hostAssocStores;
      mlir::Operation *delayTarget =
          findDelayTarget(declareOp, &entryBlock, domInfo, hostAssocStores);
      if (!delayTarget)
        continue;

      // Skip if the target is at or before the alloc, or is the declare.
      if (delayTarget->getBlock() == allocOp->getBlock() &&
          (delayTarget->isBeforeInBlock(allocOp) || delayTarget == allocOp))
        continue;
      if (delayTarget == declareOp)
        continue;

      // Ops that move together, keeping their relative order.
      llvm::SmallVector<mlir::Operation *> group;
      group.push_back(allocOp);
      if (storeOp)
        group.push_back(storeOp);
      group.push_back(declareOp);
      for (fir::StoreOp hostAssocStore : hostAssocStores)
        group.push_back(hostAssocStore);

      // Whatever the group reads from outside itself stays put, so it must
      // already dominate the new position.
      llvm::SmallPtrSet<mlir::Operation *, 8> groupSet(group.begin(),
                                                       group.end());
      auto readsDominateTarget = [&](mlir::Operation *op) {
        return llvm::all_of(op->getOperands(), [&](mlir::Value operand) {
          mlir::Operation *def = operand.getDefiningOp();
          return (def && groupSet.contains(def)) ||
                 domInfo.properlyDominates(operand, delayTarget);
        });
      };
      if (!llvm::all_of(group, readsDominateTarget))
        continue;

      // Sink the group before the target, preserving its relative order.
      group.front()->moveBefore(delayTarget);
      mlir::Operation *last = group.front();
      for (mlir::Operation *op : llvm::drop_begin(group)) {
        op->moveAfter(last);
        last = op;
      }
    }
  }
};

} // end anonymous namespace
