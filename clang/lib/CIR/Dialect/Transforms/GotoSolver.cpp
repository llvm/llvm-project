//====- GotoSolver.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_GOTOSOLVER
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct GotoSolverPass : public impl::GotoSolverBase<GotoSolverPass> {
  GotoSolverPass() = default;
  void runOnOperation() override;
};

static void process(cir::FuncOp func) {
  mlir::OpBuilder rewriter(func.getContext());
  llvm::StringMap<Block *> labels;
  llvm::SmallVector<cir::GotoOp, 4> gotos;
  llvm::SmallSet<StringRef, 4> blockAddrLabel;

  func.getBody().walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.try_emplace(lab.getLabel(), lab->getBlock());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.push_back(goTo);
    } else if (auto blockAddr = dyn_cast<cir::BlockAddressOp>(op)) {
      blockAddrLabel.insert(blockAddr.getBlockAddrInfo().getLabel());
    }
  });

  for (auto &lab : labels) {
    StringRef labelName = lab.getKey();
    Block *block = lab.getValue();
    if (!blockAddrLabel.contains(labelName)) {
      // erase the LabelOp inside the block if safe
      if (auto lab = dyn_cast<cir::LabelOp>(&block->front())) {
        lab.erase();
      }
    }
  }

  for (auto goTo : gotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(goTo);
    Block *dest = labels[goTo.getLabel()];
    cir::BrOp::create(rewriter, goTo.getLoc(), dest);
    goTo.erase();
  }
}

void GotoSolverPass::runOnOperation() {
  llvm::TimeTraceScope scope("Goto Solver");
  getOperation()->walk(&process);
}

} // namespace

std::unique_ptr<Pass> mlir::createGotoSolverPass() {
  return std::make_unique<GotoSolverPass>();
}
