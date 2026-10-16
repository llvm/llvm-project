//===-- SimdOnly.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace flangomp {
#define GEN_PASS_DEF_SIMDONLYPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {

#define DEBUG_TYPE "omp-simd-only-pass"

/// Rewrite and remove OpenMP operations left after the parse tree rewriting for
/// -fopenmp-simd is done. If possible, OpenMP constructs should be rewritten at
/// the parse tree stage. This pass is supposed to only handle complexities
/// around untangling composite simd constructs, and perform the necessary
/// cleanup.
class SimdOnlyConversionPattern : public mlir::RewritePattern {
public:
  SimdOnlyConversionPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(MatchAnyOpTypeTag{}, 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() !=
        mlir::omp::OpenMPDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(op, "Not an OpenMP op");

    if (auto simdOp = mlir::dyn_cast<mlir::omp::SimdOp>(op)) {
      // Remove the composite attr given that the op will no longer be composite
      if (simdOp.isComposite()) {
        simdOp.setComposite(false);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Op is a plain SimdOp");
    }

    if (op->getParentOfType<mlir::omp::SimdOp>() &&
        (mlir::isa<mlir::omp::YieldOp>(op) ||
         mlir::isa<mlir::omp::ScanOp>(op) ||
         mlir::isa<mlir::omp::LoopNestOp>(op) ||
         mlir::isa<mlir::omp::TerminatorOp>(op)))
      return rewriter.notifyMatchFailure(op, "Op is part of a simd construct");

    if (!mlir::isa<mlir::func::FuncOp>(op->getParentOp()) &&
        (mlir::isa<mlir::omp::TerminatorOp>(op) ||
         mlir::isa<mlir::omp::YieldOp>(op)))
      return rewriter.notifyMatchFailure(op,
                                         "Non top-level yield or terminator");

    LLVM_DEBUG(llvm::dbgs() << "SimdOnlyPass matched OpenMP op:\n");
    LLVM_DEBUG(op->dump());

    auto eraseUnlessUsedBySimd = [&](mlir::Operation *ompOp,
                                     mlir::StringAttr name) {
      if (auto uses =
              mlir::SymbolTable::getSymbolUses(name, op->getParentOp())) {
        for (auto &use : *uses)
          if (mlir::isa<mlir::omp::SimdOp>(use.getUser()))
            return rewriter.notifyMatchFailure(op,
                                               "Op used by a simd construct");
      }
      rewriter.eraseOp(ompOp);
      return mlir::success();
    };

    if (auto ompOp = mlir::dyn_cast<mlir::omp::PrivateClauseOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());
    if (auto ompOp = mlir::dyn_cast<mlir::omp::DeclareReductionOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());

    // Might be left over from rewriting composite simd with target map
    if (mlir::isa<mlir::omp::MapBoundsOp>(op)) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (auto mapInfoOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(op)) {
      rewriter.replaceOp(mapInfoOp, {mapInfoOp.getVarPtr()});
      return mlir::success();
    }

    // Might be leftover after parse tree rewriting
    if (auto threadPrivateOp = mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op)) {
      rewriter.replaceOp(threadPrivateOp, {threadPrivateOp.getSymAddr()});
      return mlir::success();
    }

    fir::FirOpBuilder builder(rewriter, op);
    mlir::Location loc = op->getLoc();

    auto inlineSimpleOp = [&](mlir::Operation *ompOp) -> bool {
      if (!ompOp)
        return false;

      assert("OpenMP operation has one region" && ompOp->getNumRegions() == 1);

      llvm::SmallVector<std::pair<mlir::Value, mlir::BlockArgument>>
          blockArgsPairs;
      if (auto iface =
              mlir::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(op)) {
        iface.getBlockArgsPairs(blockArgsPairs);
        for (auto [value, argument] : blockArgsPairs)
          rewriter.replaceAllUsesWith(argument, value);
      }

      if (ompOp->getRegion(0).getBlocks().size() == 1) {
        auto &block = *ompOp->getRegion(0).getBlocks().begin();
        // This block is about to be removed so any arguments should have been
        // replaced by now.
        block.eraseArguments(0, block.getNumArguments());
        if (auto terminatorOp =
                mlir::dyn_cast<mlir::omp::TerminatorOp>(block.back())) {
          rewriter.eraseOp(terminatorOp);
        }
        rewriter.inlineBlockBefore(&block, ompOp, {});
      } else {
        // When dealing with multi-block regions we need to fix up the control
        // flow
        auto *origBlock = ompOp->getBlock();
        auto *newBlock = rewriter.splitBlock(origBlock, ompOp->getIterator());
        auto *innerFrontBlock = &ompOp->getRegion(0).getBlocks().front();
        builder.setInsertionPointToEnd(origBlock);
        mlir::cf::BranchOp::create(builder, loc, innerFrontBlock);
        // We are no longer passing any arguments to the first block in the
        // region, so this should be safe to erase.
        innerFrontBlock->eraseArguments(0, innerFrontBlock->getNumArguments());

        for (auto &innerBlock : ompOp->getRegion(0).getBlocks()) {
          // Remove now-unused block arguments
          for (auto arg : innerBlock.getArguments()) {
            if (arg.getUses().empty())
              innerBlock.eraseArgument(arg.getArgNumber());
          }
          if (auto terminatorOp =
                  mlir::dyn_cast<mlir::omp::TerminatorOp>(innerBlock.back())) {
            builder.setInsertionPointToEnd(&innerBlock);
            mlir::cf::BranchOp::create(builder, loc, newBlock);
            rewriter.eraseOp(terminatorOp);
          }
        }

        rewriter.inlineRegionBefore(ompOp->getRegion(0), newBlock);
      }

      rewriter.eraseOp(op);
      return true;
    };

    // Remove ops that will be surrounding simd once a composite simd construct
    // goes through the codegen stage. All of the other ones should have alredy
    // been removed in the parse tree rewriting stage.
    if (inlineSimpleOp(mlir::dyn_cast<mlir::omp::TeamsOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::ParallelOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::TargetOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::WsloopOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::DistributeOp>(op)))
      return mlir::success();

    op->emitOpError("left unhandled after SimdOnly pass.");
    return mlir::failure();
  }
};

class SimdOnlyPass : public flangomp::impl::SimdOnlyPassBase<SimdOnlyPass> {

public:
  SimdOnlyPass() = default;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<SimdOnlyConversionPattern>(context);

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
      mlir::emitError(module.getLoc(), "Error in SimdOnly conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace
