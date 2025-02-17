//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {
namespace looputils {
using LoopNest = llvm::SetVector<fir::DoLoopOp>;

/// Loop \p innerLoop is considered perfectly-nested inside \p outerLoop iff
/// there are no operations in \p outerloop's body other than:
///
/// 1. the operations needed to assing/update \p outerLoop's induction variable.
/// 2. \p innerLoop itself.
///
/// \p return true if \p innerLoop is perfectly nested inside \p outerLoop
/// according to the above definition.
bool isPerfectlyNested(fir::DoLoopOp outerLoop, fir::DoLoopOp innerLoop) {
  mlir::ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.inclusive = true;
  // We don't care about the outer-loop's induction variable's uses within the
  // inner-loop, so we filter out these uses.
  //
  // This filter tells `getForwardSlice` (below) to only collect operations
  // which produce results defined above (i.e. outside) the inner-loop's body.
  //
  // Since `outerLoop.getInductionVar()` is a block argument (to the
  // outer-loop's body), the filter effectively collects uses of
  // `outerLoop.getInductionVar()` inside the outer-loop but outside the
  // inner-loop.
  forwardSliceOptions.filter = [&](mlir::Operation *op) {
    return mlir::areValuesDefinedAbove(op->getResults(), innerLoop.getRegion());
  };

  llvm::SetVector<mlir::Operation *> indVarSlice;
  mlir::getForwardSlice(outerLoop.getInductionVar(), &indVarSlice,
                        forwardSliceOptions);
  llvm::DenseSet<mlir::Operation *> indVarSet(indVarSlice.begin(),
                                              indVarSlice.end());

  llvm::DenseSet<mlir::Operation *> outerLoopBodySet;
  // The following walk collects ops inside `outerLoop` that are **not**:
  // * the outer-loop itself,
  // * or the inner-loop,
  // * or the `fir.result` op (the outer-loop's terminator).
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == outerLoop)
      return mlir::WalkResult::advance();

    if (op == innerLoop)
      return mlir::WalkResult::skip();

    if (mlir::isa<fir::ResultOp>(op))
      return mlir::WalkResult::advance();

    outerLoopBodySet.insert(op);
    return mlir::WalkResult::advance();
  });

  // If `outerLoopBodySet` ends up having the same ops as `indVarSet`, then
  // `outerLoop` only contains ops that setup its induction variable +
  // `innerLoop` + the `fir.result` terminator. In other words, `innerLoop` is
  // perfectly nested inside `outerLoop`.
  bool result = (outerLoopBodySet == indVarSet);
  mlir::Location loc = outerLoop.getLoc();
  LLVM_DEBUG(DBGS() << "Loop pair starting at location " << loc << " is"
                    << (result ? "" : " not") << " perfectly nested\n");

  return result;
}

/// Starting with `outerLoop` collect a perfectly nested loop nest, if any. This
/// function collects as much as possible loops in the nest; it case it fails to
/// recognize a certain nested loop as part of the nest it just returns the
/// parent loops it discovered before.
mlir::LogicalResult collectLoopNest(fir::DoLoopOp currentLoop,
                                    LoopNest &loopNest) {
  assert(currentLoop.getUnordered());

  while (true) {
    loopNest.insert(currentLoop);
    auto directlyNestedLoops = currentLoop.getRegion().getOps<fir::DoLoopOp>();
    llvm::SmallVector<fir::DoLoopOp> unorderedLoops;

    for (auto nestedLoop : directlyNestedLoops)
      if (nestedLoop.getUnordered())
        unorderedLoops.push_back(nestedLoop);

    if (unorderedLoops.empty())
      break;

    if (unorderedLoops.size() > 1)
      return mlir::failure();

    fir::DoLoopOp nestedUnorderedLoop = unorderedLoops.front();

    if (!isPerfectlyNested(currentLoop, nestedUnorderedLoop))
      return mlir::failure();

    currentLoop = nestedUnorderedLoop;
  }

  return mlir::success();
}
} // namespace looputils

class DoConcurrentConversion : public mlir::OpConversionPattern<fir::DoLoopOp> {
public:
  using mlir::OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  DoConcurrentConversion(mlir::MLIRContext *context, bool mapToDevice)
      : OpConversionPattern(context), mapToDevice(mapToDevice) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    looputils::LoopNest loopNest;
    bool hasRemainingNestedLoops =
        failed(looputils::collectLoopNest(doLoop, loopNest));
    if (hasRemainingNestedLoops)
      mlir::emitWarning(doLoop.getLoc(),
                        "Some `do concurent` loops are not perfectly-nested. "
                        "These will be serialzied.");

    // TODO This will be filled in with the next PRs that upstreams the rest of
    // the ROCm implementaion.
    return mlir::success();
  }

  bool mapToDevice;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration())
      return;

    mlir::MLIRContext *context = &getContext();

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoLoopOp>([&](fir::DoLoopOp op) {
      // The goal is to handle constructs that eventually get lowered to
      // `fir.do_loop` with the `unordered` attribute (e.g. array expressions).
      // Currently, this is only enabled for the `do concurrent` construct since
      // the pass runs early in the pipeline.
      return !op.getUnordered();
    });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting do-concurrent op");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? flangomp::DoConcurrentMappingKind::DCMK_Device
                              : flangomp::DoConcurrentMappingKind::DCMK_Host;

  return std::make_unique<DoConcurrentConversionPass>(options);
}
