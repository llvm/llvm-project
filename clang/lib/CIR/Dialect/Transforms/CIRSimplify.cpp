//===----------------------------------------------------------------------===//
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
///   %0 = ...
///   %1 = cir.ternary (%condition, true {
///     %2 = cir.const ...
///     cir.yield %2
///   } false {
///     cir.yield %0
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

    cir::YieldOp trueBranchYieldOp =
        mlir::cast<cir::YieldOp>(op.getTrueRegion().front().getTerminator());
    cir::YieldOp falseBranchYieldOp =
        mlir::cast<cir::YieldOp>(op.getFalseRegion().front().getTerminator());
    mlir::Value trueValue = trueBranchYieldOp.getArgs()[0];
    mlir::Value falseValue = falseBranchYieldOp.getArgs()[0];

    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);
    rewriter.eraseOp(trueBranchYieldOp);
    rewriter.eraseOp(falseBranchYieldOp);
    rewriter.replaceOpWithNewOp<cir::SelectOp>(op, op.getCond(), trueValue,
                                               falseValue);

    return mlir::success();
  }

private:
  bool isSimpleTernaryBranch(mlir::Region &region) const {
    if (!region.hasOneBlock())
      return false;

    mlir::Block &onlyBlock = region.front();
    mlir::Block::OpListType &ops = onlyBlock.getOperations();

    // The region/block could only contain at most 2 operations.
    if (ops.size() > 2)
      return false;

    if (ops.size() == 1) {
      // The region/block only contain a cir.yield operation.
      return true;
    }

    // Check whether the region/block contains a cir.const followed by a
    // cir.yield that yields the value.
    auto yieldOp = mlir::cast<cir::YieldOp>(onlyBlock.getTerminator());
    auto yieldValueDefOp = mlir::dyn_cast_if_present<cir::ConstantOp>(
        yieldOp.getArgs()[0].getDefiningOp());
    return yieldValueDefOp && yieldValueDefOp->getBlock() == &onlyBlock;
  }
};

/// Simplify select operations with boolean constants into simpler forms.
///
/// This pattern simplifies select operations where both true and false values
/// are boolean constants. Two specific cases are handled:
///
/// 1. When selecting between true and false based on a condition,
///    the operation simplifies to just the condition itself:
///
///    %0 = cir.select if %condition then true else false
///    ->
///    (replaced with %condition directly)
///
/// 2. When selecting between false and true based on a condition,
///    the operation simplifies to the logical negation of the condition:
///
///    %0 = cir.select if %condition then false else true
///    ->
///    %0 = cir.unary not %condition
struct SimplifySelect : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const final {
    mlir::Operation *trueValueOp = op.getTrueValue().getDefiningOp();
    mlir::Operation *falseValueOp = op.getFalseValue().getDefiningOp();
    auto trueValueConstOp =
        mlir::dyn_cast_if_present<cir::ConstantOp>(trueValueOp);
    auto falseValueConstOp =
        mlir::dyn_cast_if_present<cir::ConstantOp>(falseValueOp);
    if (!trueValueConstOp || !falseValueConstOp)
      return mlir::failure();

    auto trueValue = mlir::dyn_cast<cir::BoolAttr>(trueValueConstOp.getValue());
    auto falseValue =
        mlir::dyn_cast<cir::BoolAttr>(falseValueConstOp.getValue());
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
      rewriter.replaceOpWithNewOp<cir::UnaryOp>(op, cir::UnaryOpKind::Not,
                                                op.getCondition());
      return mlir::success();
    }

    return mlir::failure();
  }
};

/// Simplify `cir.switch` operations by folding cascading cases
/// into a single `cir.case` with the `anyof` kind.
///
/// This pattern identifies cascading cases within a `cir.switch` operation.
/// Cascading cases are defined as consecutive `cir.case` operations of kind
/// `equal`, each containing a single `cir.yield` operation in their body.
///
/// The pattern merges these cascading cases into a single `cir.case` operation
/// with kind `anyof`, aggregating all the case values.
///
/// The merging process continues until a `cir.case` with a different body
/// (e.g., containing `cir.break` or compound stmt) is encountered, which
/// breaks the chain.
///
/// Example:
///
/// Before:
///   cir.case equal, [#cir.int<0> : !s32i] {
///     cir.yield
///   }
///   cir.case equal, [#cir.int<1> : !s32i] {
///     cir.yield
///   }
///   cir.case equal, [#cir.int<2> : !s32i] {
///     cir.break
///   }
///
/// After applying SimplifySwitch:
///   cir.case anyof, [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> :
///   !s32i] {
///     cir.break
///   }
struct SimplifySwitch : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SwitchOp op,
                                PatternRewriter &rewriter) const override {

    LogicalResult changed = mlir::failure();
    SmallVector<CaseOp, 8> cases;
    SmallVector<CaseOp, 4> cascadingCases;
    SmallVector<mlir::Attribute, 4> cascadingCaseValues;

    op.collectCases(cases);
    if (cases.empty())
      return mlir::failure();

    auto flushMergedOps = [&]() {
      for (CaseOp &c : cascadingCases)
        rewriter.eraseOp(c);
      cascadingCases.clear();
      cascadingCaseValues.clear();
    };

    auto mergeCascadingInto = [&](CaseOp &target) {
      rewriter.modifyOpInPlace(target, [&]() {
        target.setValueAttr(rewriter.getArrayAttr(cascadingCaseValues));
        target.setKind(CaseOpKind::Anyof);
      });
      changed = mlir::success();
    };

    for (CaseOp c : cases) {
      cir::CaseOpKind kind = c.getKind();
      if (kind == cir::CaseOpKind::Equal &&
          isa<YieldOp>(c.getCaseRegion().front().front())) {
        // If the case contains only a YieldOp, collect it for cascading merge
        cascadingCases.push_back(c);
        cascadingCaseValues.push_back(c.getValue()[0]);
      } else if (kind == cir::CaseOpKind::Equal && !cascadingCases.empty()) {
        // merge previously collected cascading cases
        cascadingCaseValues.push_back(c.getValue()[0]);
        mergeCascadingInto(c);
        flushMergedOps();
      } else if (kind != cir::CaseOpKind::Equal && cascadingCases.size() > 1) {
        // If a Default, Anyof or Range case is found and there are previous
        // cascading cases, merge all of them into the last cascading case.
        // We don't currently fold case range statements with other case
        // statements.
        assert(!cir::MissingFeatures::foldRangeCase());
        CaseOp lastCascadingCase = cascadingCases.back();
        mergeCascadingInto(lastCascadingCase);
        cascadingCases.pop_back();
        flushMergedOps();
      } else {
        cascadingCases.clear();
        cascadingCaseValues.clear();
      }
    }

    // Edge case: all cases are simple cascading cases
    if (cascadingCases.size() == cases.size()) {
      CaseOp lastCascadingCase = cascadingCases.back();
      mergeCascadingInto(lastCascadingCase);
      cascadingCases.pop_back();
      flushMergedOps();
    }

    return changed;
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
    SimplifySelect,
    SimplifySwitch
  >(patterns.getContext());
  // clang-format on
}

void CIRSimplifyPass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateMergeCleanupPatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<TernaryOp, SelectOp, SwitchOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRSimplifyPass() {
  return std::make_unique<CIRSimplifyPass>();
}
