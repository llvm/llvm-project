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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
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

static void process(cir::FuncOp func,
                    llvm::ArrayRef<StringRef> globalBlockAddrLabels) {
  mlir::OpBuilder rewriter(func.getContext());
  llvm::StringMap<Block *> labels;
  llvm::SmallVector<cir::GotoOp, 4> gotos;
  llvm::SmallVector<cir::IndirectGotoOp> indirectGotos;
  // Labels whose address is taken by a cir.block_address op in this function,
  // in IR order.
  llvm::SmallVector<StringRef> opBlockAddrLabels;

  func.getBody().walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.try_emplace(lab.getLabel(), lab->getBlock());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.push_back(goTo);
    } else if (auto indirect = dyn_cast<cir::IndirectGotoOp>(op)) {
      indirectGotos.push_back(indirect);
    } else if (auto blockAddr = dyn_cast<cir::BlockAddressOp>(op)) {
      opBlockAddrLabels.push_back(blockAddr.getBlockAddrInfo().getLabel());
    }
  });

  // Address-taken labels in a deterministic order: those referenced from
  // global initializers first (in initializer order), then those taken by a
  // cir.block_address op (in IR order).  A label may be named more than once (a
  // dispatch table can list it twice); a block only needs to be a successor
  // once, so keep the first occurrence.
  llvm::SmallVector<StringRef> addrTakenLabels;
  llvm::StringSet<> addrTaken;
  auto noteAddrTaken = [&](StringRef name) {
    if (addrTaken.insert(name).second)
      addrTakenLabels.push_back(name);
  };
  for (StringRef name : globalBlockAddrLabels)
    noteAddrTaken(name);
  for (StringRef name : opBlockAddrLabels)
    noteAddrTaken(name);

  // Drop LabelOps whose address is never taken; the rest may be indirect-branch
  // successors and must survive.
  for (auto &lab : labels) {
    if (!addrTaken.contains(lab.getKey())) {
      if (auto labelOp = dyn_cast<cir::LabelOp>(&lab.getValue()->front()))
        labelOp.erase();
    }
  }

  // Resolve regular symbolic gotos to direct branches.
  for (auto goTo : gotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(goTo);
    Block *dest = labels[goTo.getLabel()];
    cir::BrOp::create(rewriter, goTo.getLoc(), dest);
    goTo.erase();
  }

  // A label whose address is merely taken still emits its address constant; an
  // indirect branch is only needed when the function actually branches with a
  // `goto *expr`.
  if (indirectGotos.empty())
    return;

  // Resolve indirect gotos.  FlattenCFG has already merged the nested scopes
  // into one region, so the shared indirect-branch block and its successors all
  // live in func's body now -- the cross-region branch that broke a nested
  // `goto *` during CIRGen cannot arise here.
  // The shared block represents every `goto *expr` that funnels into it, so
  // fuse their locations when there is more than one.
  llvm::SmallVector<mlir::Location> gotoLocs;
  for (cir::IndirectGotoOp indirect : indirectGotos)
    gotoLocs.push_back(indirect.getLoc());
  mlir::Location loc = gotoLocs.size() == 1
                           ? gotoLocs.front()
                           : mlir::FusedLoc::get(func.getContext(), gotoLocs);
  mlir::Type addrType = indirectGotos.front().getAddr().getType();
  Block *indirectGotoBlock = rewriter.createBlock(
      &func.getBody(), func.getBody().end(), {addrType}, {loc});

  llvm::SmallVector<Block *> successors;
  llvm::SmallVector<mlir::ValueRange> succOperands;
  for (StringRef name : addrTakenLabels) {
    Block *dest = labels[name];
    assert(dest && "address-taken label has no cir.label in this function");
    successors.push_back(dest);
    succOperands.push_back(dest->getArguments());
  }
  cir::IndirectBrOp::create(rewriter, loc, indirectGotoBlock->getArgument(0),
                            /*poison=*/false, succOperands, successors);

  for (auto indirect : indirectGotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(indirect);
    cir::BrOp::create(rewriter, indirect.getLoc(), indirectGotoBlock,
                      indirect.getAddr());
    indirect.erase();
  }
}

void GotoSolverPass::runOnOperation() {
  llvm::TimeTraceScope scope("Goto Solver");

  // Block addresses can also appear in attributes outside of any function body,
  // such as global variable initializers.  Collect, per target function and in
  // initializer order, the labels referenced this way so their LabelOps survive
  // and join the indirect branch's successors.  A SetVector keeps the first
  // occurrence in order: a label named more than once across initializers needs
  // to be a successor only once.
  llvm::StringMap<llvm::SmallSetVector<StringRef, 4>> globalBlockAddrLabels;
  getOperation()->walk([&](mlir::Operation *op) {
    for (const mlir::NamedAttribute &namedAttr : op->getAttrs()) {
      namedAttr.getValue().walk([&](cir::BlockAddrInfoAttr info) {
        globalBlockAddrLabels[info.getFunc().getValue()].insert(
            info.getLabel());
      });
    }
  });

  static const llvm::SmallVector<StringRef> empty;
  getOperation()->walk([&](cir::FuncOp func) {
    auto it = globalBlockAddrLabels.find(func.getSymName());
    process(func, it == globalBlockAddrLabels.end()
                      ? llvm::ArrayRef<StringRef>(empty)
                      : it->second.getArrayRef());
  });
}

} // namespace

std::unique_ptr<Pass> mlir::createGotoSolverPass() {
  return std::make_unique<GotoSolverPass>();
}
