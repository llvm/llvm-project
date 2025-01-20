//===- InlineHLFIRAssign.cpp - Inline hlfir.assign ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Transform hlfir.assign array operations into loop nests performing element
// per element assignments. The inlining is done for trivial data types always,
// though, we may add performance/code-size heuristics in future.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hlfir {
#define GEN_PASS_DEF_INLINEHLFIRASSIGN
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "inline-hlfir-assign"

namespace {
/// Expand hlfir.assign of array RHS to array LHS into a loop nest
/// of element-by-element assignments:
///   hlfir.assign %4 to %5 : !fir.ref<!fir.array<3x3xf32>>,
///                           !fir.ref<!fir.array<3x3xf32>>
/// into:
///   fir.do_loop %arg1 = %c1 to %c3 step %c1 unordered {
///     fir.do_loop %arg2 = %c1 to %c3 step %c1 unordered {
///       %6 = hlfir.designate %4 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       %7 = fir.load %6 : !fir.ref<f32>
///       %8 = hlfir.designate %5 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       hlfir.assign %7 to %8 : f32, !fir.ref<f32>
///     }
///   }
///
/// The transformation is correct only when LHS and RHS do not alias.
/// When RHS is an array expression, then there is no aliasing.
/// This transformation does not support runtime checking for
/// non-conforming LHS/RHS arrays' shapes currently.
class InlineHLFIRAssignConversion
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override {
    if (assign.isAllocatableAssignment())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp may imply allocation");

    hlfir::Entity rhs{assign.getRhs()};

    if (!rhs.isArray())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp's RHS is not an array");

    mlir::Type rhsEleTy = rhs.getFortranElementType();
    if (!fir::isa_trivial(rhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's RHS data type is not trivial");

    hlfir::Entity lhs{assign.getLhs()};
    if (!lhs.isArray())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp's LHS is not an array");

    mlir::Type lhsEleTy = lhs.getFortranElementType();
    if (!fir::isa_trivial(lhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's LHS data type is not trivial");

    if (lhsEleTy != rhsEleTy)
      return rewriter.notifyMatchFailure(assign,
                                         "RHS/LHS element types mismatch");

    if (!mlir::isa<hlfir::ExprType>(rhs.getType())) {
      // If RHS is not an hlfir.expr, then we should prove that
      // LHS and RHS do not alias.
      // TODO: if they may alias, we can insert hlfir.as_expr for RHS,
      // and proceed with the inlining.
      fir::AliasAnalysis aliasAnalysis;
      mlir::AliasResult aliasRes = aliasAnalysis.alias(lhs, rhs);
      // TODO: use areIdenticalOrDisjointSlices() from
      // OptimizedBufferization.cpp to check if we can still do the expansion.
      if (!aliasRes.isNo()) {
        LLVM_DEBUG(llvm::dbgs() << "InlineHLFIRAssign:\n"
                                << "\tLHS: " << lhs << "\n"
                                << "\tRHS: " << rhs << "\n"
                                << "\tALIAS: " << aliasRes << "\n");
        return rewriter.notifyMatchFailure(assign, "RHS/LHS may alias");
      }
    }

    mlir::Location loc = assign->getLoc();
    fir::FirOpBuilder builder(rewriter, assign.getOperation());
    builder.setInsertionPoint(assign);
    rhs = hlfir::derefPointersAndAllocatables(loc, builder, rhs);
    lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
    mlir::Value shape = hlfir::genShape(loc, builder, lhs);
    llvm::SmallVector<mlir::Value> extents =
        hlfir::getIndexExtents(loc, builder, shape);
    hlfir::LoopNest loopNest =
        hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true,
                           flangomp::shouldUseWorkshareLowering(assign));
    builder.setInsertionPointToStart(loopNest.body);
    auto rhsArrayElement =
        hlfir::getElementAt(loc, builder, rhs, loopNest.oneBasedIndices);
    rhsArrayElement = hlfir::loadTrivialScalar(loc, builder, rhsArrayElement);
    auto lhsArrayElement =
        hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
    builder.create<hlfir::AssignOp>(loc, rhsArrayElement, lhsArrayElement);
    rewriter.eraseOp(assign);
    return mlir::success();
  }
};

class InlineHLFIRAssignPass
    : public hlfir::impl::InlineHLFIRAssignBase<InlineHLFIRAssignPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;

    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineHLFIRAssignConversion>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in hlfir.assign inlining");
      signalPassFailure();
    }
  }
};
} // namespace
