//===- ACCEraseUnusedKernelAllocations.cpp - Drop dead kernel allocmem ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Erase fir.allocmem inside acc.compute_region when the storage is never read
// or written. fir.declare's debug effect and fir.freemem keep that chain alive
// through ordinary DCE, and lowering it would produce a checked device malloc.
// An unused private-recipe allocation is deleted the same way as an unused
// source array.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCERASEUNUSEDKERNELALLOCATIONS
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

namespace {

using namespace mlir;

// True when every use of \p value is fir.freemem, a view of that value
// (ViewLikeOpInterface, including fir.convert), or a fir.declare of that
// value, and the same is true of those results. No uses is included: the
// allocation is dead. fir.declare is not view-like; it only carries debug
// info for the memref.
bool isDeadAllocChain(Value value, SmallPtrSetImpl<Operation *> &bookkeeping) {
  for (Operation *user : value.getUsers()) {
    if (isa<fir::FreeMemOp>(user)) {
      bookkeeping.insert(user);
      continue;
    }
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (view.getViewSource() != value)
        return false;
      bookkeeping.insert(user);
      if (!isDeadAllocChain(view.getViewDest(), bookkeeping))
        return false;
      continue;
    }
    if (auto declare = dyn_cast<fir::DeclareOp>(user)) {
      if (declare.getMemref() != value)
        return false;
      bookkeeping.insert(declare);
      if (!isDeadAllocChain(declare.getResult(), bookkeeping))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

// Erase users before the values they consume.
void eraseDeadOps(ArrayRef<Operation *> ops) {
  SmallPtrSet<Operation *, 8> pending(ops.begin(), ops.end());
  while (!pending.empty()) {
    Operation *ready = nullptr;
    for (Operation *op : pending) {
      bool usedInSet = false;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (pending.contains(user)) {
            usedInSet = true;
            break;
          }
        }
        if (usedInSet)
          break;
      }
      if (!usedInSet) {
        ready = op;
        break;
      }
    }
    if (!ready)
      return;
    pending.erase(ready);
    ready->erase();
  }
}

class ACCEraseUnusedKernelAllocations
    : public fir::acc::impl::ACCEraseUnusedKernelAllocationsBase<
          ACCEraseUnusedKernelAllocations> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<fir::AllocMemOp> dead;
    func.walk([&](fir::AllocMemOp alloc) {
      if (!alloc->getParentOfType<acc::ComputeRegionOp>())
        return;
      SmallPtrSet<Operation *, 8> bookkeeping;
      if (!isDeadAllocChain(alloc.getResult(), bookkeeping))
        return;
      dead.push_back(alloc);
    });

    for (fir::AllocMemOp alloc : dead) {
      SmallPtrSet<Operation *, 8> bookkeeping;
      if (!isDeadAllocChain(alloc.getResult(), bookkeeping))
        continue;
      SmallVector<Operation *> toErase(bookkeeping.begin(), bookkeeping.end());
      toErase.push_back(alloc);
      eraseDeadOps(toErase);
    }
  }
};

} // namespace
