//===-- CUFAllocDelay.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Delay cuf.alloc of descriptor (box) types from function entry to just before
// their first use. This defers cudaMallocManaged calls so that users can call
// cudaSetDevice before any CUDA context is created.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFALLOCDELAY
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

/// Find the earliest use of the descriptor and return the op before which the
/// cuf.alloc group should be placed. Uses in nested regions (fir.if,
/// fir.do_loop, ...) resolve to the enclosing entry-block op; uses confined to
/// a single successor block resolve to that block.
static mlir::Operation *findDelayTarget(fir::DeclareOp declareOp,
                                        mlir::Block *entryBlock) {
  mlir::Operation *earliest = nullptr;
  mlir::Region *funcRegion = entryBlock->getParent();

  // Uses per successor block, with the earliest op in each.
  llvm::SmallDenseMap<mlir::Block *, mlir::Operation *> successorEarliest;

  // Resolve a use in a nested region or successor block to a target in/after
  // the entry block.
  auto recordRealUse = [&](mlir::Operation *user) {
    mlir::Operation *target = user;
    while (target->getBlock() != entryBlock) {
      // User in another block of the same function.
      if (target->getBlock() && target->getBlock()->getParent() == funcRegion) {
        mlir::Block *blk = target->getBlock();
        auto it = successorEarliest.find(blk);
        if (it == successorEarliest.end() ||
            target->isBeforeInBlock(it->second))
          successorEarliest[blk] = target;
        return;
      }
      target = target->getParentOp();
      if (!target)
        return;
    }
    if (!earliest || target->isBeforeInBlock(earliest))
      earliest = target;
  };

  for (mlir::Value result : declareOp->getResults()) {
    for (mlir::Operation *user : result.getUsers())
      recordRealUse(user);
  }

  if (earliest)
    return earliest;

  // No entry-block uses.  If all successor uses are in a single block,
  // delay directly into that block (before the earliest use there).
  // Otherwise fall back to the entry block's terminator.
  if (successorEarliest.size() == 1)
    return successorEarliest.begin()->second;
  if (!successorEarliest.empty())
    return entryBlock->getTerminator();
  return nullptr;
}

struct CUFAllocDelay : public fir::impl::CUFAllocDelayBase<CUFAllocDelay> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (func.empty())
      return;

    mlir::Block &entryBlock = func.front();

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

      mlir::Operation *delayTarget = findDelayTarget(declareOp, &entryBlock);
      if (!delayTarget)
        continue;

      // Skip if the target is at or before the alloc, or is the declare.
      if (delayTarget->getBlock() == allocOp->getBlock() &&
          (delayTarget->isBeforeInBlock(allocOp) || delayTarget == allocOp))
        continue;
      if (delayTarget == declareOp)
        continue;

      // Sink {cuf.alloc, fir.store, fir.declare} before the target; the
      // embox/shape/constants stay put and still dominate the new position.
      allocOp->moveBefore(delayTarget);
      if (storeOp)
        storeOp->moveAfter(allocOp);
      if (storeOp)
        declareOp->moveAfter(storeOp);
      else
        declareOp->moveAfter(allocOp);
    }
  }
};

} // end anonymous namespace
