//===- ACCHoistAllocatableRealloc.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoist the allocatable reallocation that SeparateAllocatableAssign leaves
// inside an OpenACC kernels construct out to the host, so only the assignment
// runs on the device. A device-side allocate/deallocate is undefined behavior
// under the OpenACC specification.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCHOISTALLOCATABLEREALLOC
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

using namespace mlir;

namespace {

/// True for an allocatable descriptor reference:
/// !fir.ref<!fir.box<!fir.heap<>>>.
static bool isAllocatableDescRef(Type t) {
  auto ref = dyn_cast<fir::ReferenceType>(t);
  if (!ref)
    return false;
  auto box = dyn_cast<fir::BaseBoxType>(ref.getEleTy());
  if (!box)
    return false;
  return isa<fir::HeapType>(box.getEleTy());
}

/// True if `v` is defined outside `region` (and therefore dominates the op that
/// owns the region).
static bool definedOutside(Value v, Region &region) {
  return !region.isAncestor(v.getParentRegion());
}

/// Return the ancestor of `op` that is a direct child of `region`'s entry
/// block, or null if `op` is not nested within that entry block.
static Operation *topLevelInEntry(Operation *op, Region &region) {
  Block *entry = &region.front();
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur->getBlock() == entry)
      return cur;
  }
  return nullptr;
}

/// Collect into `slice` the entry-block ops of `region` that `root` depends on
/// (its backward SSA slice, including values used by nested ops). Returns false
/// if a dependency is a region-local block argument, i.e. it cannot be hoisted.
static bool collectReallocSlice(Operation *root, Region &region,
                                llvm::SmallPtrSetImpl<Operation *> &slice) {
  llvm::SmallVector<Operation *> worklist{root};
  bool ok = true;
  auto use = [&](Value v) {
    if (definedOutside(v, region))
      return;
    if (Operation *def = v.getDefiningOp()) {
      if (Operation *top = topLevelInEntry(def, region))
        worklist.push_back(top);
      else
        ok = false;
    } else {
      ok = false; // region block argument
    }
  };
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!slice.insert(op).second)
      continue;
    for (Value v : op->getOperands())
      use(v);
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        use(v);
    });
  }
  return ok;
}

/// True if hoisting the realloc (whose backward slice is `slice`, ending at
/// `anchor`) past the kernel is safe. It is unsafe when a descriptor load that
/// is not strictly after the realloc feeds an op that stays in the kernel: that
/// op would read storage the realloc has already freed.
static bool isSafeToHoist(Value desc, Operation *anchor, Region &region,
                          const llvm::SmallPtrSetImpl<Operation *> &slice) {
  for (Operation *user : desc.getUsers()) {
    auto load = dyn_cast<fir::LoadOp>(user);
    if (!load)
      continue;
    Operation *loadTop = topLevelInEntry(load, region);
    if (!loadTop || anchor->isBeforeInBlock(loadTop))
      continue; // outside region, or the post-realloc reload
    for (Operation *consumer : load.getResult().getUsers()) {
      Operation *consTop = topLevelInEntry(consumer, region);
      if (consTop && !slice.contains(consTop))
        return false;
    }
  }
  return true;
}

/// True if `st` sits at the region's outer level, guarded only by the
/// `genReallocIfNeeded` `fir.if`s (never inside a loop). Hoisting an in-loop
/// realloc out of the construct would be wrong.
static bool reallocAtRegionTop(Operation *st, Block *entry) {
  for (Operation *cur = st->getParentOp(); cur; cur = cur->getParentOp()) {
    if (!isa<fir::IfOp>(cur))
      return false;
    if (cur->getBlock() == entry)
      return true;
  }
  return false;
}

/// True if `slice` contains the `.auto.alloc` allocation that identifies a
/// SeparateAllocatableAssign realloc (not a user allocate or other store).
static bool hasAutoAlloc(const llvm::SmallPtrSetImpl<Operation *> &slice) {
  bool found = false;
  for (Operation *op : slice)
    op->walk([&](fir::AllocMemOp a) {
      if (auto name = a.getUniqName(); name && *name == ".auto.alloc")
        found = true;
    });
  return found;
}

/// Hoist any in-region reallocation out of a single compute construct.
static void hoistFromComputeOp(Operation *computeOp) {
  Region &region = computeOp->getRegion(0);
  if (region.empty())
    return;

  // Descriptor stores writing an allocatable defined outside the region are the
  // reallocation's publish step.
  llvm::SmallVector<fir::StoreOp> reallocStores;
  computeOp->walk([&](fir::StoreOp st) {
    if (isAllocatableDescRef(st.getMemref().getType()) &&
        definedOutside(st.getMemref(), region))
      reallocStores.push_back(st);
  });
  if (reallocStores.empty())
    return;

  llvm::SmallPtrSet<Operation *, 16> slice;
  for (fir::StoreOp st : reallocStores) {
    Operation *anchor = topLevelInEntry(st, region);
    if (!anchor)
      continue;
    // Outer artificial-loop level only: never hoist a loop-body realloc.
    if (!reallocAtRegionTop(st, &region.front()))
      continue;
    llvm::SmallPtrSet<Operation *, 16> local;
    if (!collectReallocSlice(anchor, region, local))
      continue;
    // Must be the SeparateAllocatableAssign realloc, not some other store.
    if (!hasAutoAlloc(local))
      continue;
    // Unsafe to hoist if the assignment reads the old storage the realloc
    // frees: any consumer of a pre-realloc descriptor load that stays in the
    // kernel would read freed memory (e.g. a self-dependent `b = b + 1`).
    if (!isSafeToHoist(st.getMemref(), anchor, region, local))
      continue;
    slice.insert(local.begin(), local.end());
  }
  if (slice.empty())
    return;

  // Move the collected ops before the compute op, preserving their order.
  for (Operation &op : llvm::make_early_inc_range(region.front()))
    if (slice.contains(&op))
      op.moveBefore(computeOp);
}

class ACCHoistAllocatableRealloc
    : public fir::acc::impl::ACCHoistAllocatableReallocBase<
          ACCHoistAllocatableRealloc> {
public:
  void runOnOperation() override {
    // Only acc kernels: the compiler-introduced realloc is hoisted to the
    // host. For acc parallel/serial the (de)allocation is the user's
    // responsibility and is left untouched.
    llvm::SmallVector<mlir::acc::KernelsOp> kernels;
    getOperation().walk(
        [&](mlir::acc::KernelsOp op) { kernels.push_back(op); });
    for (mlir::acc::KernelsOp op : kernels)
      hoistFromComputeOp(op);
  }
};

} // namespace

std::unique_ptr<Pass> fir::acc::createACCHoistAllocatableReallocPass() {
  return std::make_unique<ACCHoistAllocatableRealloc>();
}
