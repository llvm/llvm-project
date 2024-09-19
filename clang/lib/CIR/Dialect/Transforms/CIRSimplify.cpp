//===- CIRSimplify.cpp - performs CIR simplification ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

/// Simplify suitable ternary operations into select operations.
///
/// For now we only simplify those ternary operations whose true and false
/// branches directly yield a value or a constant. That is, both of the true and
/// the false branch must either contain a cir.yield operation as the only
/// operation in the branch, or contain a cir.const operation followed by a
/// cir.yield operation that yields the constant value.
///
/// For example, we will simplify the following ternary operation:
///
///   %0 = cir.ternary (%condition, true {
///     %1 = cir.const ...
///     cir.yield %1
///   } false {
///     cir.yield %2
///   })
///
/// into the following sequence of operations:
///
///   %1 = cir.const ...
///   %0 = cir.select if %condition then %1 else %2
struct SimplifyTernary final : public OpRewritePattern<TernaryOp> {
  using OpRewritePattern<TernaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TernaryOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return mlir::failure();

    if (!isSimpleTernaryBranch(op.getTrueRegion()) ||
        !isSimpleTernaryBranch(op.getFalseRegion()))
      return mlir::failure();

    mlir::cir::YieldOp trueBranchYieldOp = mlir::cast<mlir::cir::YieldOp>(
        op.getTrueRegion().front().getTerminator());
    mlir::cir::YieldOp falseBranchYieldOp = mlir::cast<mlir::cir::YieldOp>(
        op.getFalseRegion().front().getTerminator());
    auto trueValue = trueBranchYieldOp.getArgs()[0];
    auto falseValue = falseBranchYieldOp.getArgs()[0];

    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);
    rewriter.eraseOp(trueBranchYieldOp);
    rewriter.eraseOp(falseBranchYieldOp);
    rewriter.replaceOpWithNewOp<mlir::cir::SelectOp>(op, op.getCond(),
                                                     trueValue, falseValue);

    return mlir::success();
  }

private:
  bool isSimpleTernaryBranch(mlir::Region &region) const {
    if (!region.hasOneBlock())
      return false;

    mlir::Block &onlyBlock = region.front();
    auto &ops = onlyBlock.getOperations();

    // The region/block could only contain at most 2 operations.
    if (ops.size() > 2)
      return false;

    if (ops.size() == 1) {
      // The region/block only contain a cir.yield operation.
      return true;
    }

    // Check whether the region/block contains a cir.const followed by a
    // cir.yield that yields the value.
    auto yieldOp = mlir::cast<mlir::cir::YieldOp>(onlyBlock.getTerminator());
    auto yieldValueDefOp = mlir::dyn_cast_if_present<mlir::cir::ConstantOp>(
        yieldOp.getArgs()[0].getDefiningOp());
    return yieldValueDefOp && yieldValueDefOp->getBlock() == &onlyBlock;
  }
};

struct SimplifySelect : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const final {
    mlir::Operation *trueValueOp = op.getTrueValue().getDefiningOp();
    mlir::Operation *falseValueOp = op.getFalseValue().getDefiningOp();
    auto trueValueConstOp =
        mlir::dyn_cast_if_present<mlir::cir::ConstantOp>(trueValueOp);
    auto falseValueConstOp =
        mlir::dyn_cast_if_present<mlir::cir::ConstantOp>(falseValueOp);
    if (!trueValueConstOp || !falseValueConstOp)
      return mlir::failure();

    auto trueValue =
        mlir::dyn_cast<mlir::cir::BoolAttr>(trueValueConstOp.getValue());
    auto falseValue =
        mlir::dyn_cast<mlir::cir::BoolAttr>(falseValueConstOp.getValue());
    if (!trueValue || !falseValue)
      return mlir::failure();

    // cir.select if %0 then #true else #false -> %0
    if (trueValue.getValue() && !falseValue.getValue()) {
      rewriter.replaceAllUsesWith(op, op.getCondition());
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // cir.select if %0 then #false else #true -> cir.unary not %0
    if (!trueValue.getValue() && falseValue.getValue()) {
      rewriter.replaceOpWithNewOp<mlir::cir::UnaryOp>(
          op, mlir::cir::UnaryOpKind::Not, op.getCondition());
      return mlir::success();
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// CIRSimplifyPass
//===----------------------------------------------------------------------===//

struct CIRSimplifyPass : public CIRSimplifyBase<CIRSimplifyPass> {
  using CIRSimplifyBase::CIRSimplifyBase;

  void runOnOperation() override;
};

void populateMergeCleanupPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    SimplifyTernary,
    SimplifySelect
  >(patterns.getContext());
  // clang-format on
}

void CIRSimplifyPass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateMergeCleanupPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<TernaryOp, SelectOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRSimplifyPass() {
  return std::make_unique<CIRSimplifyPass>();
}
