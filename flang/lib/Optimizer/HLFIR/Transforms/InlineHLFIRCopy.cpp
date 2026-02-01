//===- InlineHLFIRCopy.cpp - Inline hlfir.copy_in/copy_out ops ------------===//
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
//
// When the copy_in is inlined, the corresponding copy_out (deallocation-only
// case, i.e., when var is null) is also inlined with direct fir.freemem.
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
#define GEN_PASS_DEF_INLINEHLFIRCOPY
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "inline-hlfir-copy"

static llvm::cl::opt<bool> noInlineHLFIRCopy(
    "no-inline-hlfir-copy",
    llvm::cl::desc("Do not inline hlfir.copy_in/copy_out operations"),
    llvm::cl::init(false));

namespace {
class InlineCopyInConversion : public mlir::OpRewritePattern<hlfir::CopyInOp> {
public:
  using mlir::OpRewritePattern<hlfir::CopyInOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CopyInOp copyIn,
                  mlir::PatternRewriter &rewriter) const override;
};

// Helper function to inline a copy_out deallocation (no copy-back case).
// Generates: if (wasCopied) { freemem(box_addr(load(tempBox))) }
static void inlineCopyOutDeallocation(fir::FirOpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::Value tempBox,
                                      mlir::Value wasCopied,
                                      mlir::Type sequenceType) {
  builder.genIfOp(loc, {}, wasCopied, /*withElseRegion=*/false)
      .genThen([&]() {
        mlir::Value box = fir::LoadOp::create(builder, loc, tempBox);
        mlir::Value addr = fir::BoxAddrOp::create(builder, loc, box);
        auto heapType = fir::HeapType::get(sequenceType);
        mlir::Value heapAddr =
            fir::ConvertOp::create(builder, loc, heapType, addr);
        fir::FreeMemOp::create(builder, loc, heapAddr);
      });
}

// Note: We don't have a separate InlineCopyOutConversion pattern.
// Copy_out inlining is handled by InlineCopyInConversion when it inlines
// the paired copy_in. For copy_outs that aren't paired with an eligible
// copy_in (e.g., optional args, assumed-rank, non-trivial types), the
// copy_out is left as-is and will be lowered to a runtime call.
// This is the conservative approach for the upstream pass.

llvm::LogicalResult
InlineCopyInConversion::matchAndRewrite(hlfir::CopyInOp copyIn,
                                        mlir::PatternRewriter &rewriter) const {
  fir::FirOpBuilder builder(rewriter, copyIn.getOperation());
  mlir::Location loc = copyIn.getLoc();
  hlfir::Entity inputVariable{copyIn.getVar()};
  mlir::Type resultAddrType = copyIn.getCopiedIn().getType();
  if (!fir::isa_trivial(inputVariable.getFortranElementType()))
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp's data type is not trivial");

  // There should be exactly one user of WasCopied - the corresponding
  // CopyOutOp.
  if (!copyIn.getWasCopied().hasOneUse())
    return rewriter.notifyMatchFailure(
        copyIn, "CopyInOp's WasCopied has no single user");
  // The copy out should always be present, either to actually copy or just
  // deallocate memory.
  auto copyOut = mlir::dyn_cast<hlfir::CopyOutOp>(
      copyIn.getWasCopied().user_begin().getCurrent().getUser());

  if (!copyOut)
    return rewriter.notifyMatchFailure(copyIn,
                                       "CopyInOp has no direct CopyOut");

  if (mlir::cast<fir::BaseBoxType>(resultAddrType).isAssumedRank())
    return rewriter.notifyMatchFailure(copyIn,
                                       "The result array is assumed-rank");

  // Only inline the copy_in when copy_out does not need to be done, i.e. in
  // case of intent(in).
  if (copyOut.getVar())
    return rewriter.notifyMatchFailure(copyIn, "CopyIn needs a copy-out");

  inputVariable =
      hlfir::derefPointersAndAllocatables(loc, builder, inputVariable);
  mlir::Type sequenceType =
      hlfir::getFortranElementOrSequenceType(inputVariable.getType());
  fir::BoxType resultBoxType = fir::BoxType::get(sequenceType);
  mlir::Value isContiguous =
      fir::IsContiguousBoxOp::create(builder, loc, inputVariable);
  mlir::Operation::result_range results =
      builder
          .genIfOp(loc, {resultBoxType, builder.getI1Type()}, isContiguous,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            mlir::Value result = inputVariable;
            if (fir::isPointerType(inputVariable.getType())) {
              result = fir::ReboxOp::create(builder, loc, resultBoxType,
                                            inputVariable, mlir::Value{},
                                            mlir::Value{});
            }
            fir::ResultOp::create(
                builder, loc,
                mlir::ValueRange{result, builder.createBool(loc, false)});
          })
          .genElse([&] {
            mlir::Value shape = hlfir::genShape(loc, builder, inputVariable);
            llvm::SmallVector<mlir::Value> extents =
                hlfir::getIndexExtents(loc, builder, shape);
            llvm::StringRef tmpName{".tmp.copy_in"};
            llvm::SmallVector<mlir::Value> lenParams;
            mlir::Value alloc = builder.createHeapTemporary(
                loc, sequenceType, tmpName, extents, lenParams);

            auto declareOp = hlfir::DeclareOp::create(builder, loc, alloc,
                                                      tmpName, shape, lenParams,
                                                      /*dummy_scope=*/nullptr,
                                                      /*storage=*/nullptr,
                                                      /*storage_offset=*/0);
            hlfir::Entity temp{declareOp.getBase()};
            hlfir::LoopNest loopNest =
                hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true,
                                   flangomp::shouldUseWorkshareLowering(copyIn),
                                   /*couldVectorize=*/false);
            builder.setInsertionPointToStart(loopNest.body);
            hlfir::Entity elem = hlfir::getElementAt(
                loc, builder, inputVariable, loopNest.oneBasedIndices);
            elem = hlfir::loadTrivialScalar(loc, builder, elem);
            hlfir::Entity tempElem = hlfir::getElementAt(
                loc, builder, temp, loopNest.oneBasedIndices);
            hlfir::AssignOp::create(builder, loc, elem, tempElem);
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
              result = fir::EmboxOp::create(builder, loc, resultBoxType, refVal,
                                            shape);
            }

            fir::ResultOp::create(
                builder, loc,
                mlir::ValueRange{result, builder.createBool(loc, true)});
          })
          .getResults();

  mlir::OpResult resultBox = results[0];
  mlir::OpResult needsCleanup = results[1];

  // Inline the corresponding copyOut to free the temporary if it is required.
  // Generate: if (needsCleanup) { freemem(box_addr(resultBox)) }
  // We need to store the resultBox first since it's a box value, then generate
  // the deallocation code at the copyOut location.
  auto alloca = fir::AllocaOp::create(builder, loc, resultBox.getType());
  fir::StoreOp::create(builder, loc, resultBox, alloca);

  // Move to the copyOut location to generate the deallocation
  rewriter.setInsertionPoint(copyOut);
  fir::FirOpBuilder copyOutBuilder(rewriter, copyOut.getOperation());
  inlineCopyOutDeallocation(copyOutBuilder, copyOut.getLoc(), alloca,
                            needsCleanup, sequenceType);

  // Erase the copyOut since we've inlined it
  rewriter.eraseOp(copyOut);

  rewriter.replaceOp(copyIn, {resultBox, builder.genNot(loc, isContiguous)});
  return mlir::success();
}

class InlineHLFIRCopyPass
    : public hlfir::impl::InlineHLFIRCopyBase<InlineHLFIRCopyPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    if (!noInlineHLFIRCopy) {
      patterns.insert<InlineCopyInConversion>(context);
    }

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in hlfir.copy_in/copy_out inlining");
      signalPassFailure();
    }
  }
};
} // namespace
