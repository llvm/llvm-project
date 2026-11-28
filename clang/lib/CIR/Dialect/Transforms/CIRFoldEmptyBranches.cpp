//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that removes blocks containing only an
// unconditional branch to a block that has no other predecessor, redirecting
// their predecessors to the branch target.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRFOLDEMPTYBRANCHES
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CIRFoldEmptyBranchesPass
    : public impl::CIRFoldEmptyBranchesBase<CIRFoldEmptyBranchesPass> {
  using CIRFoldEmptyBranchesBase::CIRFoldEmptyBranchesBase;

  void runOnOperation() override;
};

/// Erase `block` if it is a non-entry block with no arguments whose only
/// operation is a `cir.br` to another block, and `block` is the only
/// predecessor of that block. A `cir.br` with a loop id is a loop back edge
/// that carries the loop's metadata, so it is kept. Every predecessor
/// terminator must implement BranchOpInterface so the `cir.br` operands can be
/// forwarded.
static bool foldEmptyBranch(Block &block) {
  if (block.isEntryBlock() || block.getNumArguments() != 0 ||
      !llvm::hasSingleElement(block))
    return false;

  auto br = dyn_cast<cir::BrOp>(block.front());
  if (!br || br.getDest() == &block || br.getLoopIdAttr())
    return false;

  Block *dest = br.getDest();
  if (dest->getSinglePredecessor() != &block)
    return false;

  if (!llvm::all_of(block.getPredecessors(), [](Block *pred) {
        return isa<BranchOpInterface>(pred->getTerminator());
      }))
    return false;

  SmallVector<Value> operands(br.getDestOperands());
  for (BlockOperand &use : llvm::make_early_inc_range(block.getUses())) {
    auto branch = cast<BranchOpInterface>(use.getOwner());
    unsigned index = use.getOperandNumber();
    branch.getSuccessorOperands(index).append(operands);
    branch->setSuccessor(dest, index);
  }
  block.erase();
  return true;
}

void CIRFoldEmptyBranchesPass::runOnOperation() {
  getOperation()->walk([](Region *region) {
    bool changed;
    do {
      changed = false;
      for (Block &block : llvm::make_early_inc_range(*region))
        changed |= foldEmptyBranch(block);
    } while (changed);
  });
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRFoldEmptyBranchesPass() {
  return std::make_unique<CIRFoldEmptyBranchesPass>();
}
