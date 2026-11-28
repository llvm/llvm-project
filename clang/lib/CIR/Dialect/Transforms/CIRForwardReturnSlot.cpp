//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that returns the value stored to a function's
// return slot directly when that store dominates the only return.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRFORWARDRETURNSLOT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CIRForwardReturnSlotPass
    : public impl::CIRForwardReturnSlotBase<CIRForwardReturnSlotPass> {
  using CIRForwardReturnSlotBase::CIRForwardReturnSlotBase;

  void runOnOperation() override;
};

/// Returns `op` as a store to `slot` if it is one that may be forwarded.
static cir::StoreOp getStoreToSlot(Operation *op, Value slot) {
  auto store = dyn_cast_if_present<cir::StoreOp>(op);
  // Classic codegen also forwards the volatile stores that every memory access
  // in a function with an SEH __try becomes.
  assert(!cir::MissingFeatures::functionUsesSEHTry());
  if (!store || store.getAddr() != slot || store.isVolatileOrAtomic())
    return {};
  return store;
}

/// Returns true if `store` precedes `load` in its block, or if `store`'s block
/// reaches `load`'s block through a chain of single predecessors. This is the
/// quick dominance check of classic codegen.
static bool reachesThroughSinglePredecessors(cir::StoreOp store,
                                             cir::LoadOp load) {
  Block *storeBlock = store->getBlock();
  Block *block = load->getBlock();
  if (block == storeBlock)
    return store->isBeforeInBlock(load);
  llvm::SmallPtrSet<Block *, 4> seen;
  while (block != storeBlock) {
    if (!seen.insert(block).second || !(block = block->getSinglePredecessor()))
      return false;
  }
  return true;
}

/// Finds the store whose value `load` (the load of the return slot feeding the
/// function's only return) can be replaced with, following
/// findDominatingStoreToReturnValue in clang/lib/CodeGen/CGCall.cpp.
static cir::StoreOp findDominatingStoreToReturnSlot(cir::LoadOp load) {
  Value slot = load.getAddr();

  // If the slot has other uses than one store and the load, only the operation
  // right before the load is considered. The store is erased, so nothing may
  // read the slot after the load.
  if (!slot.hasNUses(2)) {
    Block *block = load->getBlock();
    if (llvm::any_of(slot.getUsers(), [&](Operation *user) {
          return user->getBlock() == block && load->isBeforeInBlock(user);
        }))
      return {};
    return getStoreToSlot(load->getPrevNode(), slot);
  }

  Operation *user = *llvm::find_if(slot.getUsers(), [&](Operation *user) {
    return user != load.getOperation();
  });
  cir::StoreOp store = getStoreToSlot(user, slot);
  if (!store || !reachesThroughSinglePredecessors(store, load))
    return {};
  return store;
}

/// Returns true if only operations without memory effects, such as branches,
/// run between `store` and `load`. Their blocks must be connected through
/// single predecessors.
static bool runsNoCodeBetween(cir::StoreOp store, cir::LoadOp load) {
  auto isInert = [](Operation &op) {
    return isa<cir::BrOp>(op) || isMemoryEffectFree(&op);
  };
  Block *block = load->getBlock();
  Block::iterator end = load->getIterator();
  while (block != store->getBlock()) {
    if (!llvm::all_of(llvm::make_range(block->begin(), end), isInert))
      return false;
    block = block->getSinglePredecessor();
    end = block->end();
  }
  return llvm::all_of(llvm::make_range(std::next(store->getIterator()), end),
                      isInert);
}

/// Classic codegen only stores scalars to the return value slot with a single
/// store; aggregates and complex values are copied or stored by parts.
static bool isScalarSlotType(Type type) {
  return !isa<cir::RecordType, cir::ArrayType, cir::ComplexType>(type);
}

/// Forwards the store to the return slot of `func`, and returns true if it
/// changed anything.
static bool forwardReturnSlot(cir::FuncOp func) {
  // Classic codegen returns through a single return block, so only a function
  // with one return of a value is a candidate.
  cir::ReturnOp ret;
  WalkResult walk = func.walk([&](cir::ReturnOp op) {
    if (op.getInput().empty())
      return WalkResult::advance();
    if (ret)
      return WalkResult::interrupt();
    ret = op;
    return WalkResult::advance();
  });
  if (walk.wasInterrupted() || !ret)
    return false;

  // Classic codegen forwards only a return that is not coerced. Call
  // convention lowering has already run, and it replaces the operand of a
  // coerced return with the coerced value, so a returned load of the slot is
  // never coerced.
  auto load = ret.getInput().front().getDefiningOp<cir::LoadOp>();
  if (!load || !load->hasOneUse() || load->getBlock() != ret->getBlock() ||
      load.isVolatileOrAtomic())
    return false;
  auto slot = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!slot || !slot.getReturnSlot() || !isScalarSlotType(slot.getAllocaType()))
    return false;

  cir::StoreOp store = findDominatingStoreToReturnSlot(load);
  if (!store)
    return false;

  // Classic codegen gives the return the store's location unless cleanup code
  // runs between the two. CIR does not record cleanups at this point, so the
  // location moves only when no code runs between them.
  if (runsNoCodeBetween(store, load))
    ret->setLoc(store.getLoc());
  ret.getInputMutable().assign(store.getValue());
  load.erase();
  store.erase();
  if (slot->use_empty())
    slot.erase();
  return true;
}

void CIRForwardReturnSlotPass::runOnOperation() {
  if (!forwardReturnSlot(getOperation()))
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRForwardReturnSlotPass() {
  return std::make_unique<CIRForwardReturnSlotPass>();
}
