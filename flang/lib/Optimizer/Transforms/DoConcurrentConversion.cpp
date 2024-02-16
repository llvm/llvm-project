//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace fir {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fopenmp-do-concurrent-conversion"

namespace {
class DoConcurrentConversion : public mlir::OpConversionPattern<fir::DoLoopOp> {
public:
  using mlir::OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm();

    mlir::omp::ParallelOp parallelOp =
        rewriter.create<mlir::omp::ParallelOp>(doLoop.getLoc());

    mlir::Block *block = rewriter.createBlock(&parallelOp.getRegion());

    rewriter.setInsertionPointToEnd(block);
    rewriter.create<mlir::omp::TerminatorOp>(doLoop.getLoc());

    rewriter.setInsertionPointToStart(block);

    // ==== TODO (1) Start ====
    //
    // The goal of the few lines below is to collect and clone
    // the list of operations that define the loop's lower and upper bounds as
    // well as the step. Should we, instead of doing this here, split it into 2
    // stages?
    //
    //   1. **Stage 1**: add an analysis that extracts all the relevant
    //                   operations defining the lower-bound, upper-bound, and
    //                   step.
    //   2. **Stage 2**: clone the collected operations in the parallel region.
    //
    // So far, the pass has been tested with very simple loops (where the bounds
    // and step are constants) so the goal of **Stage 1** is to have a
    // well-defined component that has the sole responsibility of collecting all
    // the relevant ops relevant to the loop header. This was we can test this
    // in isolation for more complex loops and better organize the code. **Stage
    // 2** would then be responsible for the actual cloning of the collected
    // loop header preparation/allocation operations.

    // Clone the LB, UB, step defining ops inside the parallel region.
    llvm::SmallVector<mlir::Value> lowerBound, upperBound, step;
    lowerBound.push_back(
        rewriter.clone(*doLoop.getLowerBound().getDefiningOp())->getResult(0));
    upperBound.push_back(
        rewriter.clone(*doLoop.getUpperBound().getDefiningOp())->getResult(0));
    step.push_back(
        rewriter.clone(*doLoop.getStep().getDefiningOp())->getResult(0));
    // ==== TODO (1) End ====

    auto wsLoopOp = rewriter.create<mlir::omp::WsLoopOp>(
        doLoop.getLoc(), lowerBound, upperBound, step);
    wsLoopOp.setInclusive(true);

    auto outlineableOp =
        mlir::dyn_cast<mlir::omp::OutlineableOpenMPOpInterface>(*parallelOp);
    rewriter.setInsertionPointToStart(outlineableOp.getAllocaBlock());

    // ==== TODO (2) Start ====
    //
    // The goal of the following simple work-list algorithm and
    // the following `for` loop is to collect all the operations related to the
    // allocation of the induction variable for the `do concurrent` loop. The
    // operations collected by this algorithm are very similar to what is
    // usually emitted for privatized variables, e.g. for omp.parallel loops.
    // Therefore, I think we can:
    //
    //   1. **Stage 1**: Add an analysis that colects all these operations. The
    //                   goal is similar to **Stage 1** of TODO (1): isolate the
    //                   algorithm is an individually-testable component so that
    //                   we properly implement and test it for more complicated
    //                   `do concurrent` loops.
    //   1. **Stage 2**: Using the collected operations, create and populate an
    //                   `omp.private {type=private}` op to server as the
    //                   delayed privatizer for the new work-sharing loop.

    // For the induction variable, we need to privative its allocation and
    // binding inside the parallel region.
    llvm::SmallSetVector<mlir::Operation *, 2> workList;
    // Therefore, we first discover the induction variable by discovering
    // `fir.store`s where the source is the loop's block argument.
    workList.insert(doLoop.getInductionVar().getUsers().begin(),
                    doLoop.getInductionVar().getUsers().end());
    llvm::SmallSetVector<fir::StoreOp, 2> inductionVarTargetStores;

    // Walk the def-chain of the loop's block argument until we hit `fir.store`.
    while (!workList.empty()) {
      mlir::Operation *item = workList.front();

      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(item)) {
        inductionVarTargetStores.insert(storeOp);
      } else {
        workList.insert(item->getUsers().begin(), item->getUsers().end());
      }

      workList.remove(item);
    }

    // For each collected `fir.sotre`, find the target memref's alloca's and
    // declare ops.
    llvm::SmallSetVector<mlir::Operation *, 4> declareAndAllocasToClone;
    for (auto storeOp : inductionVarTargetStores) {
      mlir::Operation *storeTarget = storeOp.getMemref().getDefiningOp();

      for (auto operand : storeTarget->getOperands()) {
        declareAndAllocasToClone.insert(operand.getDefiningOp());
      }
      declareAndAllocasToClone.insert(storeTarget);
    }
    // ==== TODO (2) End ====
    //
    // TODO (1 & 2): Isolating analyses proposed in both TODOs, I think we can
    // more easily generalize the pass to work for targets other than OpenMP,
    // e.g. OpenACC, I think can, can reuse the results of the analyses and only
    // change the code-gen/rewriting.

    mlir::IRMapping mapper;

    // Collect the memref defining ops in the parallel region.
    for (mlir::Operation *opToClone : declareAndAllocasToClone) {
      rewriter.clone(*opToClone, mapper);
    }

    // Clone the loop's body inside the worksharing construct using the mapped
    // memref values.
    rewriter.cloneRegionBefore(doLoop.getRegion(), wsLoopOp.getRegion(),
                               wsLoopOp.getRegion().begin(), mapper);

    mlir::Operation *terminator = wsLoopOp.getRegion().back().getTerminator();
    rewriter.setInsertionPointToEnd(&wsLoopOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);

    rewriter.eraseOp(doLoop);

    return mlir::success();
  }
};

class DoConcurrentConversionPass
    : public fir::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration()) {
      return;
    }

    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<fir::FIROpsDialect, hlfir::hlfirDialect,
                           mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::omp::OpenMPDialect>();

    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [](fir::DoLoopOp op) { return !op.getUnordered(); });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting do-concurrent op");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createDoConcurrentConversionPass() {
  return std::make_unique<DoConcurrentConversionPass>();
}
