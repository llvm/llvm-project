//===- InlineHLFIRCopy.cpp - Inline hlfir.copy_in/copy_out ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Transform hlfir.copy_in and hlfir.copy_out array operations into loop nests
// performing element per element assignments. For simplicity, the inlining is
// done for trivial data types when the input array is not behind a pointer.
// This may change in the future.
//
// When the copy_in is inlined, the corresponding copy_out is also inlined.
// Currently only intent(in) (deallocation-only) copy_out ops are inlined;
// the copy_in/copy_out pair is left as-is when copy-back is required
// (intent(inout/out)). Copy-back inlining may be added in the future.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Support/AllocationPolicy.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
/// Everything needed to compute the constant byte size of a buffer, gathered
/// once by the pass since it is module-level information.
struct SizeContext {
  std::optional<mlir::DataLayout> dataLayout;
  std::optional<fir::KindMapping> kindMap;
};

/// Gather the module level information needed to compute buffer sizes. Without
/// a data layout no size can be computed, and all the buffers are then left on
/// the heap.
static SizeContext getSizeContext(mlir::Operation *op) {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return SizeContext{std::nullopt, std::nullopt};
  return SizeContext{fir::support::getOrSetMLIRDataLayout(
                         module, /*allowDefaultLayout=*/false),
                     fir::getKindMapping(module)};
}

class InlineCopyInConversion : public mlir::OpRewritePattern<hlfir::CopyInOp> {
public:
  InlineCopyInConversion(mlir::MLIRContext *context,
                         const fir::AllocationPolicy &policy,
                         const SizeContext &sizeContext)
      : mlir::OpRewritePattern<hlfir::CopyInOp>(context), policy(policy),
        sizeContext(sizeContext) {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::CopyInOp copyIn,
                  mlir::PatternRewriter &rewriter) const override;

private:
  /// Return true if the copy-in buffer of type \p sequenceType should be
  /// allocated on the stack rather than on the heap.
  bool shouldUseStack(mlir::Location loc, mlir::Type sequenceType) const;

  fir::AllocationPolicy policy;
  const SizeContext &sizeContext;
};

bool InlineCopyInConversion::shouldUseStack(mlir::Location loc,
                                            mlir::Type sequenceType) const {
  // Only buffers with a compile-time constant size are considered. A buffer
  // with a runtime size would need stack save/restore to avoid growing the
  // stack when the copy-in is inside a loop. There is also little to gain: for
  // a big buffer the element-per-element copy costs much more than the
  // allocation itself.
  if (fir::hasDynamicSize(sequenceType))
    return false;
  if (!sizeContext.dataLayout || !sizeContext.kindMap)
    return false;
  auto sizeAndAlignment = fir::getTypeSizeAndAlignment(
      loc, sequenceType, *sizeContext.dataLayout, *sizeContext.kindMap);
  if (!sizeAndAlignment)
    return false;

  fir::PendingAllocationInfo info;
  info.isTemporary = true;
  info.isDynamic = false;
  info.byteSize = static_cast<std::int64_t>(sizeAndAlignment->first);
  // The per-function stack budget is not tracked here: the
  // allocation-placement pass sees the fir.alloca generated below and can
  // still move it back to the heap if the budget turns out to be exceeded.
  return fir::shouldAllocateOnStack(info, policy, /*stackBytesUsed=*/0);
}

// Inline a copy_out operation (deallocation only — no copy-back).
// Generates: if (wasCopied) { freemem(temp) }
static void inlineCopyOut(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value tempBox, mlir::Value wasCopied,
                          mlir::Type sequenceType) {
  builder.genIfOp(loc, {}, wasCopied, /*withElseRegion=*/false).genThen([&]() {
    mlir::Value addr = fir::BoxAddrOp::create(builder, loc, tempBox);
    auto heapType = fir::HeapType::get(sequenceType);
    mlir::Value heapAddr = fir::ConvertOp::create(builder, loc, heapType, addr);
    fir::FreeMemOp::create(builder, loc, heapAddr);
  });
}

// Note: We don't have a separate InlineCopyOutConversion pattern.
// Copy_out inlining is handled by InlineCopyInConversion when it inlines
// the paired copy_in. For copy_outs that aren't paired with an eligible
// copy_in (e.g., optional args, assumed-rank, non-trivial types), the
// copy_out is left as-is and will be lowered to a runtime call.

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

  // Compute shape for use in the copy-in loop and temporary declaration.
  mlir::Value shape = hlfir::genShape(loc, builder, inputVariable);
  llvm::SmallVector<mlir::Value> extents =
      hlfir::getIndexExtents(loc, builder, shape);

  // Decide where the buffer will live before creating it, so that the matching
  // kind of allocation and deallocation is generated.
  const bool useStack = shouldUseStack(loc, sequenceType);

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
            llvm::StringRef tmpName{".tmp.copy_in"};
            llvm::SmallVector<mlir::Value> lenParams;
            mlir::Value alloc =
                useStack ? builder.createTemporary(loc, sequenceType, tmpName,
                                                   extents, lenParams)
                         : builder.createHeapTemporary(
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
  mlir::OpResult wasCopied = results[1];

  // Inline the corresponding copyOut. A stack buffer needs no deallocation, so
  // there is nothing to generate for it.
  if (!useStack) {
    rewriter.setInsertionPoint(copyOut);
    fir::FirOpBuilder copyOutBuilder(rewriter, copyOut.getOperation());
    inlineCopyOut(copyOutBuilder, copyOut.getLoc(), resultBox, wasCopied,
                  sequenceType);
  }

  // Erase the copyOut since we've inlined it
  rewriter.eraseOp(copyOut);

  rewriter.replaceOp(copyIn, {resultBox, builder.genNot(loc, isContiguous)});
  return mlir::success();
}

class InlineHLFIRCopyPass
    : public hlfir::impl::InlineHLFIRCopyBase<InlineHLFIRCopyPass> {
public:
  using InlineHLFIRCopyBase<InlineHLFIRCopyPass>::InlineHLFIRCopyBase;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    // Gather allocation policy for created temporary buffers (heap vs stack).
    // This is done here because StackArrays cannot promote the heap allocations
    // created here to stack allocations because of the branches.
    // StackArrays is not honored yet: copy-in buffers are often unused at
    // runtime, so it is unclear whether putting large buffers on the stack is
    // beneficial. Runtime-sized buffers additionally need stack save/restore
    // to avoid growing the stack when the copy-in sits in a loop.
    fir::AllocationPolicy policy = fir::getAllocationPolicy(getOperation());
    policy.stackArrays = false;
    fir::overrideIfExplicitlySet(policy.smallArrayThresholdBytes,
                                 smallArrayThresholdBytes);
    const SizeContext sizeContext = getSizeContext(getOperation());

    mlir::RewritePatternSet patterns(context);
    if (!noInlineHLFIRCopy) {
      patterns.insert<InlineCopyInConversion>(context, policy, sizeContext);
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
