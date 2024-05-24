//===- SCFPrepare.cpp - pareparation work for SCF lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

static Value findIVAddr(Block *step) {
  Value IVAddr = nullptr;
  for (Operation &op : *step) {
    if (auto loadOp = dyn_cast<LoadOp>(op))
      IVAddr = loadOp.getAddr();
    else if (auto storeOp = dyn_cast<StoreOp>(op))
      if (IVAddr != storeOp.getAddr())
        return nullptr;
  }
  return IVAddr;
}

static CmpOp findLoopCmpAndIV(Block *cond, Value IVAddr, Value &IV) {
  Operation *IVLoadOp = nullptr;
  for (Operation &op : *cond) {
    if (auto loadOp = dyn_cast<LoadOp>(op))
      if (loadOp.getAddr() == IVAddr) {
        IVLoadOp = &op;
        break;
      }
  }
  if (!IVLoadOp)
    return nullptr;
  if (!IVLoadOp->hasOneUse())
    return nullptr;
  IV = IVLoadOp->getResult(0);
  return dyn_cast<CmpOp>(*IVLoadOp->user_begin());
}

// Canonicalize IV to LHS of loop comparison
// For example, transfer cir.cmp(gt, %bound, %IV) to cir.cmp(lt, %IV, %bound).
// So we could use RHS as boundary and use lt to determine it's an upper bound.
struct canonicalizeIVtoCmpLHS : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  CmpOpKind swapCmpKind(CmpOpKind kind) const {
    switch (kind) {
    case CmpOpKind::gt:
      return CmpOpKind::lt;
    case CmpOpKind::ge:
      return CmpOpKind::le;
    case CmpOpKind::lt:
      return CmpOpKind::gt;
    case CmpOpKind::le:
      return CmpOpKind::ge;
    default:
      break;
    }
    return kind;
  }

  void replaceWithNewCmpOp(CmpOp oldCmp, CmpOpKind newKind, Value lhs,
                           Value rhs, PatternRewriter &rewriter) const {
    rewriter.setInsertionPointAfter(oldCmp.getOperation());
    auto newCmp = rewriter.create<mlir::cir::CmpOp>(
        oldCmp.getLoc(), oldCmp.getType(), newKind, lhs, rhs);
    oldCmp->replaceAllUsesWith(newCmp);
    oldCmp->erase();
  }

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const final {
    auto *cond = &op.getCond().front();
    auto *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);
    if (!step)
      return failure();
    Value IVAddr = findIVAddr(step);
    if (!IVAddr)
      return failure();
    Value IV = nullptr;
    auto loopCmp = findLoopCmpAndIV(cond, IVAddr, IV);
    if (!loopCmp || !IV)
      return failure();

    CmpOpKind cmpKind = loopCmp.getKind();
    Value cmpRhs = loopCmp.getRhs();
    // Canonicalize IV to LHS of loop Cmp.
    if (loopCmp.getLhs() != IV) {
      cmpKind = swapCmpKind(cmpKind);
      cmpRhs = loopCmp.getLhs();
      replaceWithNewCmpOp(loopCmp, cmpKind, IV, cmpRhs, rewriter);
      return success();
    }

    return failure();
  }
};

// Hoist loop invariant operations in condition block out of loop
// The condition block may be generated as following which contains the
// operations produced upper bound.
// SCF for loop required loop boundary as input operands. So we need to
// hoist the boundary operations out of loop.
//
//   cir.for : cond {
//     %4 = cir.load %2 : !cir.ptr<!s32i>, !s32i
//     %5 = cir.const #cir.int<100> : !s32i       <- upper bound
//     %6 = cir.cmp(lt, %4, %5) : !s32i, !s32i
//     %7 = cir.cast(int_to_bool, %6 : !s32i), !cir.bool
//     cir.condition(%7
//  } body {
struct hoistLoopInvariantInCondBlock : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  bool isLoopInvariantLoad(Operation *op, ForOp forOp) const {
    auto load = dyn_cast<LoadOp>(op);
    if (!load)
      return false;

    auto loadAddr = load.getAddr();
    auto result =
        forOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
          if (auto store = dyn_cast<StoreOp>(op)) {
            if (store.getAddr() == loadAddr)
              return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });

    if (result.wasInterrupted())
      return false;

    return true;
  }

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const final {
    auto *cond = &forOp.getCond().front();
    auto *step =
        (forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr);
    if (!step)
      return failure();
    Value IVAddr = findIVAddr(step);
    if (!IVAddr)
      return failure();
    Value IV = nullptr;
    auto loopCmp = findLoopCmpAndIV(cond, IVAddr, IV);
    if (!loopCmp || !IV)
      return failure();

    Value cmpRhs = loopCmp.getRhs();
    auto defOp = cmpRhs.getDefiningOp();
    SmallVector<Operation *> ops;
    // Go through the cast if exist.
    if (defOp && isa<mlir::cir::CastOp>(defOp)) {
      ops.push_back(defOp);
      defOp = defOp->getOperand(0).getDefiningOp();
    }
    if (defOp &&
        (isa<ConstantOp>(defOp) || isLoopInvariantLoad(defOp, forOp))) {
      ops.push_back(defOp);
      for (auto op : reverse(ops))
        op->moveBefore(forOp);
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// SCFPreparePass
//===----------------------------------------------------------------------===//

struct SCFPreparePass : public SCFPrepareBase<SCFPreparePass> {
  using SCFPrepareBase::SCFPrepareBase;
  void runOnOperation() override;
};

void populateSCFPreparePatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    canonicalizeIVtoCmpLHS,
    hoistLoopInvariantInCondBlock
  >(patterns.getContext());
  // clang-format on
}

void SCFPreparePass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateSCFPreparePatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    // CastOp here is to perform a manual `fold` in
    // applyOpPatternsAndFold
    if (isa<ForOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createSCFPreparePass() {
  return std::make_unique<SCFPreparePass>();
}
