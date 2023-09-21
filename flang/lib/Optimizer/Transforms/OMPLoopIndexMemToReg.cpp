//===- OMPWsLoopIndexMem2Reg.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to push allocations into an OpenMP loop
// operation region when they are used to store loop indices. Then, they are
// removed together with any associated load or store operations if their
// address is not needed, in which case uses of their values are replaced for
// the block argument from which they were originally initialized.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace fir {
#define GEN_PASS_DEF_OMPLOOPINDEXMEMTOREG
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

template <typename LoopOpTy>
class LoopProcessorHelper {
  LoopOpTy loop;

  bool allUsesInLoop(ValueRange stores) {
    for (Value store : stores) {
      for (OpOperand &use : store.getUses()) {
        Operation *owner = use.getOwner();
        if (owner->getParentOfType<LoopOpTy>() != loop.getOperation())
          return false;
      }
    }
    return true;
  }

  /// Check whether a given hlfir.declare known to only be used inside of the
  /// loop and initialized by a fir.alloca operation also only used inside of
  /// the loop can be removed and replaced by the block argument representing
  /// the corresponding loop index.
  static bool isDeclareRemovable(hlfir::DeclareOp declareOp) {
    fir::AllocaOp allocaOp = llvm::dyn_cast_if_present<fir::AllocaOp>(
        declareOp.getMemref().getDefiningOp());

    // Check that the hlfir.declare is initialized by a fir.alloca that is only
    // used as argument to that operation.
    if (!allocaOp || !allocaOp.getResult().hasOneUse())
      return false;

    // Check that uses of the pointers can be replaced by the block argument.
    for (OpOperand &use : declareOp.getOriginalBase().getUses()) {
      Operation *owner = use.getOwner();
      if (!isa<fir::StoreOp>(owner))
        return false;
    }
    for (OpOperand &use : declareOp.getBase().getUses()) {
      Operation *owner = use.getOwner();
      if (!isa<fir::LoadOp>(owner))
        return false;
    }

    return true;
  }

  /// Check whether a given fir.alloca known to only be used inside of the loop
  /// can be removed and replaced by the block argument representing the
  /// corresponding loop index.
  static bool isAllocaRemovable(fir::AllocaOp allocaOp) {
    // Check that uses of the pointer are all fir.load and fir.store.
    for (OpOperand &use : allocaOp.getResult().getUses()) {
      Operation *owner = use.getOwner();
      if (!isa<fir::LoadOp>(owner) && !isa<fir::StoreOp>(owner))
        return false;
    }

    return true;
  }

  /// Try to push an hlfir.declare operation defined outside of the loop inside,
  /// if all uses of that operation and the corresponding fir.alloca are
  /// contained inside of the loop.
  LogicalResult pushDeclareIntoLoop(hlfir::DeclareOp declareOp) {
    // Check that all uses are inside of the loop.
    if (!allUsesInLoop(declareOp->getResults()))
      return failure();

    // Push hlfir.declare into the beginning of the loop region.
    Block &b = loop.getRegion().getBlocks().front();
    declareOp->moveBefore(&b, b.begin());

    // Find associated fir.alloca and push into the beginning of the loop
    // region.
    fir::AllocaOp allocaOp =
        cast<fir::AllocaOp>(declareOp.getMemref().getDefiningOp());
    Value allocaVal = allocaOp.getResult();

    if (!allUsesInLoop(allocaVal))
      return failure();

    allocaOp->moveBefore(&b, b.begin());
    return success();
  }

  /// Try to push a fir.alloca operation defined outside of the loop inside,
  /// if all uses of that operation are contained inside of the loop.
  LogicalResult pushAllocaIntoLoop(fir::AllocaOp allocaOp) {
    Value store = allocaOp.getResult();

    // Check that all uses are inside of the loop.
    if (!allUsesInLoop(store))
      return failure();

    // Push fir.alloca into the beginning of the loop region.
    Block &b = loop.getRegion().getBlocks().front();
    allocaOp->moveBefore(&b, b.begin());
    return success();
  }

  void processLoopArg(BlockArgument arg, llvm::ArrayRef<Value> argStores,
                      SmallPtrSetImpl<Operation *> &opsToDelete) {
    llvm::SmallPtrSet<Operation *, 16> toDelete;
    for (Value store : argStores) {
      Operation *op = store.getDefiningOp();

      // Skip argument if storage not defined by an operation.
      if (!op)
        return;

      // Support HLFIR flow as well as regular FIR flow.
      if (auto declareOp = dyn_cast<hlfir::DeclareOp>(op)) {
        if (succeeded(pushDeclareIntoLoop(declareOp)) &&
            isDeclareRemovable(declareOp)) {
          // Mark hlfir.declare, fir.alloca and related uses for deletion.
          for (OpOperand &use : declareOp.getOriginalBase().getUses())
            toDelete.insert(use.getOwner());

          for (OpOperand &use : declareOp.getBase().getUses())
            toDelete.insert(use.getOwner());

          Operation *allocaOp = declareOp.getMemref().getDefiningOp();
          toDelete.insert(declareOp);
          toDelete.insert(allocaOp);
        }
      } else if (auto allocaOp = dyn_cast<fir::AllocaOp>(op)) {
        if (succeeded(pushAllocaIntoLoop(allocaOp)) &&
            isAllocaRemovable(allocaOp)) {
          // Do not make any further modifications if an address to the index
          // is necessary. Otherwise, the values can be used directly from the
          // loop region first block's arguments.

          // Mark fir.alloca and related uses for deletion.
          for (OpOperand &use : allocaOp.getResult().getUses())
            toDelete.insert(use.getOwner());

          // Delete now-unused fir.alloca.
          toDelete.insert(allocaOp);
        }
      } else {
        return;
      }
    }

    // Only consider marked operations if all load, store and allocation
    // operations associated with the given loop index can be removed.
    opsToDelete.insert(toDelete.begin(), toDelete.end());

    for (Operation *op : toDelete) {
      // Replace all fir.load operations with the index as returned by the
      // OpenMP loop operation.
      if (isa<fir::LoadOp>(op))
        op->replaceAllUsesWith(ValueRange(arg));
      // Drop all uses of fir.alloca and hlfir.declare because their defining
      // operations will be deleted as well.
      else if (isa<fir::AllocaOp>(op) || isa<hlfir::DeclareOp>(op))
        op->dropAllUses();
    }
  }

public:
  explicit LoopProcessorHelper(LoopOpTy loop) : loop(loop) {}

  void process() {
    llvm::SmallPtrSet<Operation *, 16> opsToDelete;
    llvm::SmallVector<llvm::SmallVector<Value>> storeAddresses;
    llvm::ArrayRef<BlockArgument> loopArgs = loop.getRegion().getArguments();

    // Collect arguments of the loop operation.
    for (BlockArgument arg : loopArgs) {
      // Find fir.store uses of these indices and gather all addresses where
      // they are stored.
      llvm::SmallVector<Value> &argStores = storeAddresses.emplace_back();
      for (OpOperand &argUse : arg.getUses())
        if (auto storeOp = dyn_cast<fir::StoreOp>(argUse.getOwner()))
          argStores.push_back(storeOp.getMemref());
    }

    // Process all loop indices and mark them for deletion independently of each
    // other.
    for (auto it : llvm::zip(loopArgs, storeAddresses))
      processLoopArg(std::get<0>(it), std::get<1>(it), opsToDelete);

    // Delete marked operations.
    for (Operation *op : opsToDelete)
      op->erase();
  }
};

namespace {
class OMPLoopIndexMemToRegPass
    : public fir::impl::OMPLoopIndexMemToRegBase<OMPLoopIndexMemToRegPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func->walk(
        [&](omp::WsLoopOp loop) { LoopProcessorHelper(loop).process(); });

    func.walk(
        [&](omp::SimdLoopOp loop) { LoopProcessorHelper(loop).process(); });

    func.walk(
        [&](omp::TaskLoopOp loop) { LoopProcessorHelper(loop).process(); });
  }
};
} // namespace

std::unique_ptr<Pass> fir::createOMPLoopIndexMemToRegPass() {
  return std::make_unique<OMPLoopIndexMemToRegPass>();
}
