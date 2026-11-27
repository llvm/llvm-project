//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that replaces symbolic `cir.goto` operations
// with control-flow edges, so that the CFG and the region-branch interfaces
// describe every way into a block.
//
// A goto whose label is in the same region becomes a `cir.br`.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_GOTOELIMINATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct GotoEliminationPass
    : public impl::GotoEliminationBase<GotoEliminationPass> {
  using GotoEliminationBase::GotoEliminationBase;

  void runOnOperation() override;
};

} // namespace

/// Erases the labels of `func` that no goto names and whose address is not
/// taken. Returns the number of labels erased.
static unsigned eraseUnusedLabels(cir::FuncOp func, RewriterBase &rewriter) {
  llvm::DenseSet<StringAttr> named;
  SmallVector<cir::LabelOp> labels;
  func.getBody().walk([&](Operation *op) {
    if (auto label = dyn_cast<cir::LabelOp>(op))
      labels.push_back(label);
    else if (auto gotoOp = dyn_cast<cir::GotoOp>(op))
      named.insert(gotoOp.getLabelAttr());
  });
  unsigned numErased = 0;
  for (cir::LabelOp label : labels) {
    if (label.getAddressTaken() || named.contains(label.getLabelAttr()))
      continue;
    rewriter.eraseOp(label);
    ++numErased;
  }
  return numErased;
}

void GotoEliminationPass::runOnOperation() {
  cir::FuncOp func = getOperation();
  IRRewriter rewriter(func.getContext());

  llvm::DenseMap<StringAttr, cir::LabelOp> labels;
  SmallVector<cir::GotoOp> gotos;
  func.getBody().walk([&](Operation *op) {
    if (auto label = dyn_cast<cir::LabelOp>(op))
      labels.try_emplace(label.getLabelAttr(), label);
    else if (auto gotoOp = dyn_cast<cir::GotoOp>(op))
      gotos.push_back(gotoOp);
  });
  for (cir::GotoOp gotoOp : gotos) {
    Block *dest = labels.lookup(gotoOp.getLabelAttr())->getBlock();
    // A goto this cannot route yet is left for GotoSolver.
    if (dest->getParent() != gotoOp->getParentRegion())
      continue;
    rewriter.setInsertionPoint(gotoOp);
    rewriter.replaceOpWithNewOp<cir::BrOp>(gotoOp, dest);
    ++numSameRegionGotos;
  }

  if (!labels.empty())
    numErasedLabels += eraseUnusedLabels(func, rewriter);
}

std::unique_ptr<Pass> mlir::createGotoEliminationPass() {
  return std::make_unique<GotoEliminationPass>();
}
