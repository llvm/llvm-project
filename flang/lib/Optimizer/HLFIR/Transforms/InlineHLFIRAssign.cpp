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
#include "flang/Optimizer/Dialect/FIRType.h"
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

class InlineCopyInConversion : public mlir::OpRewritePattern<hlfir::CopyInOp> {
public:
  using mlir::OpRewritePattern<hlfir::CopyInOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CopyInOp copyIn,
                  mlir::PatternRewriter &rewriter) const override;
};

llvm::LogicalResult
InlineCopyInConversion::matchAndRewrite(hlfir::CopyInOp copyIn,
                                        mlir::PatternRewriter &rewriter) const {
  fir::FirOpBuilder builder(rewriter, copyIn.getOperation());
  mlir::Location loc = copyIn.getLoc();
  hlfir::Entity inputVariable{copyIn.getVar()};
  if (!fir::isa_trivial(inputVariable.getFortranElementType()))
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp's data type is not trivial");

  if (fir::isPointerType(inputVariable.getType()))
    return rewriter.notifyMatchFailure(
        copyIn, "CopyInOp's input variable is a pointer");

  // There should be exactly one user of WasCopied - the corresponding
  // CopyOutOp.
  if (copyIn.getWasCopied().getUses().empty())
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp's WasCopied has no uses");
  // The copy out should always be present, either to actually copy or just
  // deallocate memory.
  auto *copyOut =
      copyIn.getWasCopied().getUsers().begin().getCurrent().getUser();

  if (!mlir::isa<hlfir::CopyOutOp>(copyOut))
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp has no direct CopyOut");

  // Only inline the copy_in when copy_out does not need to be done, i.e. in
  // case of intent(in).
  if (::llvm::cast<hlfir::CopyOutOp>(copyOut).getVar())
    return rewriter.notifyMatchFailure(copyIn, "CopyIn needs a copy-out");

  inputVariable =
      hlfir::derefPointersAndAllocatables(loc, builder, inputVariable);
  mlir::Type resultAddrType = copyIn.getCopiedIn().getType();
  mlir::Value isContiguous =
      builder.create<fir::IsContiguousBoxOp>(loc, inputVariable);
  auto results =
      builder
          .genIfOp(loc, {resultAddrType, builder.getI1Type()}, isContiguous,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            mlir::Value falseVal = builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(false));
            builder.create<fir::ResultOp>(
                loc, mlir::ValueRange{inputVariable, falseVal});
          })
          .genElse([&] {
            auto [temp, cleanup] =
                hlfir::createTempFromMold(loc, builder, inputVariable);
            mlir::Value shape = hlfir::genShape(loc, builder, inputVariable);
            llvm::SmallVector<mlir::Value> extents =
                hlfir::getIndexExtents(loc, builder, shape);
            hlfir::LoopNest loopNest = hlfir::genLoopNest(
                loc, builder, extents, /*isUnordered=*/true,
                flangomp::shouldUseWorkshareLowering(copyIn));
            builder.setInsertionPointToStart(loopNest.body);
            auto elem = hlfir::getElementAt(loc, builder, inputVariable,
                                            loopNest.oneBasedIndices);
            elem = hlfir::loadTrivialScalar(loc, builder, elem);
            auto tempElem = hlfir::getElementAt(loc, builder, temp,
                                                loopNest.oneBasedIndices);
            builder.create<hlfir::AssignOp>(loc, elem, tempElem);
            builder.setInsertionPointAfter(loopNest.outerOp);

            mlir::Value result;
            // Make sure the result is always a boxed array by boxing it
            // ourselves if need be.
            if (mlir::isa<fir::BaseBoxType>(temp.getType())) {
              result = temp;
            } else {
              auto refTy =
                  fir::ReferenceType::get(temp.getElementOrSequenceType());
              auto refVal = builder.createConvert(loc, refTy, temp);
              result =
                  builder.create<fir::EmboxOp>(loc, resultAddrType, refVal);
            }

            builder.create<fir::ResultOp>(loc,
                                          mlir::ValueRange{result, cleanup});
          })
          .getResults();

  auto addr = results[0];
  auto needsCleanup = results[1];

  builder.setInsertionPoint(copyOut);
  builder.genIfOp(loc, {}, needsCleanup, false).genThen([&] {
    auto boxAddr = builder.create<fir::BoxAddrOp>(loc, addr);
    auto heapType = fir::HeapType::get(fir::BoxValue(addr).getBaseTy());
    auto heapVal = builder.createConvert(loc, heapType, boxAddr.getResult());
    builder.create<fir::FreeMemOp>(loc, heapVal);
  });
  rewriter.eraseOp(copyOut);

  auto tempBox = copyIn.getTempBox();

  rewriter.replaceOp(copyIn, {addr, builder.genNot(loc, isContiguous)});

  // The TempBox is only needed for flang-rt calls which we're no longer
  // generating.
  rewriter.eraseOp(tempBox.getDefiningOp());
  return mlir::success();
}

class InlineHLFIRAssignPass
    : public hlfir::impl::InlineHLFIRAssignBase<InlineHLFIRAssignPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineHLFIRAssignConversion>(context);
    patterns.insert<InlineCopyInConversion>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in hlfir.assign inlining");
      signalPassFailure();
    }
  }
};
} // namespace
