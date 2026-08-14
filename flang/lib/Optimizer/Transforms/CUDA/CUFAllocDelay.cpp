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

/// Find the earliest use of any of the declare results, returning the
/// operation before which the cuf.alloc group should be placed.
///
/// Uses inside nested regions (fir.if, fir.do_loop, etc.) are resolved to
/// the parent op in the entry block.  When all real uses reside in a single
/// successor block of the entry block, the target is placed in that block
/// (just before the earliest use there) so the alloc is deferred past any
/// setup code that precedes the branch.
///
/// Stores to fir.llvm_ptr destinations (host-association tuple slots) are
/// skipped as "uses" and collected in \p hostAssocStores so the caller can
/// move them along with the sunk group.
static mlir::Operation *
findDelayTarget(fir::DeclareOp declareOp, mlir::Block *entryBlock,
                llvm::SmallVectorImpl<fir::StoreOp> &hostAssocStores) {
  mlir::Operation *earliest = nullptr;
  mlir::Region *funcRegion = entryBlock->getParent();

  // Track successor-block uses: which blocks, and the earliest op in each.
  llvm::SmallDenseMap<mlir::Block *, mlir::Operation *> successorEarliest;

  auto updateEarliest = [&](mlir::Operation *user) {
    if (auto store = mlir::dyn_cast<fir::StoreOp>(user)) {
      if (mlir::isa<fir::LLVMPointerType>(store.getMemref().getType())) {
        hostAssocStores.push_back(store);
        return;
      }
    }
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
      updateEarliest(user);
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
      // Find the fir.declare and fir.store that use this cuf.alloc.
      // Bail out if the alloc has any unexpected users to avoid breaking
      // dominance for patterns we don't track.
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
          findDelayTarget(declareOp, &entryBlock, hostAssocStores);
      if (!delayTarget)
        continue;

      // Don't move if target is in the same block and at or before current pos.
      if (delayTarget->getBlock() == allocOp->getBlock() &&
          (delayTarget->isBeforeInBlock(allocOp) || delayTarget == allocOp))
        continue;
      // Don't move to the declare itself.
      if (delayTarget == declareOp)
        continue;

      // Move {cuf.alloc, fir.store, fir.declare} before the delay target.
      // The embox/zero_bits/shape/constants stay at their original positions
      // since they still dominate the new locations.
      allocOp->moveBefore(delayTarget);
      if (storeOp)
        storeOp->moveAfter(allocOp);
      if (storeOp)
        declareOp->moveAfter(storeOp);
      else
        declareOp->moveAfter(allocOp);

      // Move host-association tuple stores after the declare.
      mlir::Operation *insertAfter = declareOp;
      for (fir::StoreOp hostStore : hostAssocStores) {
        hostStore->moveAfter(insertAfter);
        insertAfter = hostStore;
      }
    }
  }
};

} // end anonymous namespace
