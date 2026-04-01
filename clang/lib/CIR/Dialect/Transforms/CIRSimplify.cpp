//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Block.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/Region.h"
#include "aiir/Support/LogicalResult.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace aiir;
using namespace cir;

namespace aiir {
#define GEN_PASS_DEF_CIRSIMPLIFY
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace aiir

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
      return aiir::failure();

    if (!isSimpleTernaryBranch(op.getTrueRegion()) ||
        !isSimpleTernaryBranch(op.getFalseRegion()))
      return aiir::failure();

    cir::YieldOp trueBranchYieldOp =
        aiir::cast<cir::YieldOp>(op.getTrueRegion().front().getTerminator());
    cir::YieldOp falseBranchYieldOp =
        aiir::cast<cir::YieldOp>(op.getFalseRegion().front().getTerminator());
    aiir::Value trueValue = trueBranchYieldOp.getArgs()[0];
    aiir::Value falseValue = falseBranchYieldOp.getArgs()[0];

    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);
    rewriter.eraseOp(trueBranchYieldOp);
    rewriter.eraseOp(falseBranchYieldOp);
    rewriter.replaceOpWithNewOp<cir::SelectOp>(op, op.getCond(), trueValue,
                                               falseValue);

    return aiir::success();
  }

private:
  bool isSimpleTernaryBranch(aiir::Region &region) const {
    if (!region.hasOneBlock())
      return false;

    aiir::Block &onlyBlock = region.front();
    aiir::Block::OpListType &ops = onlyBlock.getOperations();

    // The region/block could only contain at most 2 operations.
    if (ops.size() > 2)
      return false;

    if (ops.size() == 1) {
      // The region/block only contain a cir.yield operation.
      return true;
    }

    // Check whether the region/block contains a cir.const followed by a
    // cir.yield that yields the value.
    auto yieldOp = aiir::cast<cir::YieldOp>(onlyBlock.getTerminator());
    auto yieldValueDefOp =
        yieldOp.getArgs()[0].getDefiningOp<cir::ConstantOp>();
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
///    %0 = cir.not %condition
struct SimplifySelect : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const final {
    auto trueValueOp = op.getTrueValue().getDefiningOp<cir::ConstantOp>();
    auto falseValueOp = op.getFalseValue().getDefiningOp<cir::ConstantOp>();
    if (!trueValueOp || !falseValueOp)
      return aiir::failure();

    auto trueValue = trueValueOp.getValueAttr<cir::BoolAttr>();
    auto falseValue = falseValueOp.getValueAttr<cir::BoolAttr>();
    if (!trueValue || !falseValue)
      return aiir::failure();

    // cir.select if %0 then #true else #false -> %0
    if (trueValue.getValue() && !falseValue.getValue()) {
      rewriter.replaceAllUsesWith(op, op.getCondition());
      rewriter.eraseOp(op);
      return aiir::success();
    }

    // cir.select if %0 then #false else #true -> cir.not %0
    if (!trueValue.getValue() && falseValue.getValue()) {
      rewriter.replaceOpWithNewOp<cir::NotOp>(op, op.getCondition());
      return aiir::success();
    }

    return aiir::failure();
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

    LogicalResult changed = aiir::failure();
    SmallVector<CaseOp, 8> cases;
    SmallVector<CaseOp, 4> cascadingCases;
    SmallVector<aiir::Attribute, 4> cascadingCaseValues;

    op.collectCases(cases);
    if (cases.empty())
      return aiir::failure();

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
      changed = aiir::success();
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

struct SimplifyVecSplat : public OpRewritePattern<VecSplatOp> {
  using OpRewritePattern<VecSplatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(VecSplatOp op,
                                PatternRewriter &rewriter) const override {
    aiir::Value splatValue = op.getValue();
    auto constant = splatValue.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return aiir::failure();

    auto value = constant.getValue();
    if (!aiir::isa_and_nonnull<cir::IntAttr>(value) &&
        !aiir::isa_and_nonnull<cir::FPAttr>(value))
      return aiir::failure();

    cir::VectorType resultType = op.getResult().getType();
    SmallVector<aiir::Attribute, 16> elements(resultType.getSize(), value);
    auto constVecAttr = cir::ConstVectorAttr::get(
        resultType, aiir::ArrayAttr::get(getContext(), elements));

    rewriter.replaceOpWithNewOp<cir::ConstantOp>(op, constVecAttr);
    return aiir::success();
  }
};

//===----------------------------------------------------------------------===//
// CIRSimplifyPass
//===----------------------------------------------------------------------===//

struct CIRSimplifyPass : public impl::CIRSimplifyBase<CIRSimplifyPass> {
  using CIRSimplifyBase::CIRSimplifyBase;

  void runOnOperation() override;
};

void populateMergeCleanupPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    SimplifyTernary,
    SimplifySelect,
    SimplifySwitch,
    SimplifyVecSplat
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
    if (isa<TernaryOp, SelectOp, SwitchOp, VecSplatOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> aiir::createCIRSimplifyPass() {
  return std::make_unique<CIRSimplifyPass>();
}
