//===- InlineHLFIRCopyIn.cpp - Inline hlfir.copy_in ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Transform hlfir.copy_in array operations into loop nests performing element
// per element assignments. For simplicity, the inlining is done for trivial
// data types when the copy_in does not require a corresponding copy_out and
// when the input array is not behind a pointer. This may change in the future.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hlfir {
#define GEN_PASS_DEF_INLINEHLFIRCOPYIN
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "inline-hlfir-copy-in"

static llvm::cl::opt<bool> noInlineHLFIRCopyIn(
    "no-inline-hlfir-copy-in",
    llvm::cl::desc("Do not inline hlfir.copy_in operations"),
    llvm::cl::init(false));

namespace {
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
  auto copyOut = mlir::dyn_cast<hlfir::CopyOutOp>(
      copyIn.getWasCopied().getUsers().begin().getCurrent().getUser());

  if (!copyOut)
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp has no direct CopyOut");

  // Only inline the copy_in when copy_out does not need to be done, i.e. in
  // case of intent(in).
  if (copyOut.getVar())
    return rewriter.notifyMatchFailure(copyIn, "CopyIn needs a copy-out");

  inputVariable =
      hlfir::derefPointersAndAllocatables(loc, builder, inputVariable);
  mlir::Type resultAddrType = copyIn.getCopiedIn().getType();
  mlir::Value isContiguous =
      builder.create<fir::IsContiguousBoxOp>(loc, inputVariable);
  mlir::Operation::result_range results =
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
            hlfir::Entity elem = hlfir::getElementAt(
                loc, builder, inputVariable, loopNest.oneBasedIndices);
            elem = hlfir::loadTrivialScalar(loc, builder, elem);
            hlfir::Entity tempElem = hlfir::getElementAt(
                loc, builder, temp, loopNest.oneBasedIndices);
            builder.create<hlfir::AssignOp>(loc, elem, tempElem);
            builder.setInsertionPointAfter(loopNest.outerOp);

            mlir::Value result;
            // Make sure the result is always a boxed array by boxing it
            // ourselves if need be.
            if (mlir::isa<fir::BaseBoxType>(temp.getType())) {
              result = temp;
            } else {
              fir::ReferenceType refTy =
                  fir::ReferenceType::get(temp.getElementOrSequenceType());
              mlir::Value refVal = builder.createConvert(loc, refTy, temp);
              result =
                  builder.create<fir::EmboxOp>(loc, resultAddrType, refVal);
            }

            builder.create<fir::ResultOp>(loc,
                                          mlir::ValueRange{result, cleanup});
          })
          .getResults();

  mlir::OpResult addr = results[0];
  mlir::OpResult needsCleanup = results[1];

  builder.setInsertionPoint(copyOut);
  builder.genIfOp(loc, {}, needsCleanup, /*withElseRegion=*/false).genThen([&] {
    auto boxAddr = builder.create<fir::BoxAddrOp>(loc, addr);
    fir::HeapType heapType =
        fir::HeapType::get(fir::BoxValue(addr).getBaseTy());
    mlir::Value heapVal =
        builder.createConvert(loc, heapType, boxAddr.getResult());
    builder.create<fir::FreeMemOp>(loc, heapVal);
  });
  rewriter.eraseOp(copyOut);

  mlir::Value tempBox = copyIn.getTempBox();

  rewriter.replaceOp(copyIn, {addr, builder.genNot(loc, isContiguous)});

  // The TempBox is only needed for flang-rt calls which we're no longer
  // generating. It should have no uses left at this stage.
  if (!tempBox.getUses().empty())
    return mlir::failure();
  rewriter.eraseOp(tempBox.getDefiningOp());

  return mlir::success();
}

class InlineHLFIRCopyInPass
    : public hlfir::impl::InlineHLFIRCopyInBase<InlineHLFIRCopyInPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    if (!noInlineHLFIRCopyIn) {
      patterns.insert<InlineCopyInConversion>(context);
    }

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in hlfir.copy_in inlining");
      signalPassFailure();
    }
  }
};
} // namespace
