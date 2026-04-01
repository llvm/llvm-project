//===-- SimdOnly.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
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
class SimdOnlyConversionPattern : public aiir::RewritePattern {
public:
  SimdOnlyConversionPattern(aiir::AIIRContext *ctx)
      : aiir::RewritePattern(MatchAnyOpTypeTag{}, 1, ctx) {}

  aiir::LogicalResult
  matchAndRewrite(aiir::Operation *op,
                  aiir::PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() !=
        aiir::omp::OpenMPDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(op, "Not an OpenMP op");

    if (auto simdOp = aiir::dyn_cast<aiir::omp::SimdOp>(op)) {
      // Remove the composite attr given that the op will no longer be composite
      if (simdOp.isComposite()) {
        simdOp.setComposite(false);
        return aiir::success();
      }

      return rewriter.notifyMatchFailure(op, "Op is a plain SimdOp");
    }

    if (op->getParentOfType<aiir::omp::SimdOp>() &&
        (aiir::isa<aiir::omp::YieldOp>(op) ||
         aiir::isa<aiir::omp::ScanOp>(op) ||
         aiir::isa<aiir::omp::LoopNestOp>(op) ||
         aiir::isa<aiir::omp::TerminatorOp>(op)))
      return rewriter.notifyMatchFailure(op, "Op is part of a simd construct");

    if (!aiir::isa<aiir::func::FuncOp>(op->getParentOp()) &&
        (aiir::isa<aiir::omp::TerminatorOp>(op) ||
         aiir::isa<aiir::omp::YieldOp>(op)))
      return rewriter.notifyMatchFailure(op,
                                         "Non top-level yield or terminator");

    LLVM_DEBUG(llvm::dbgs() << "SimdOnlyPass matched OpenMP op:\n");
    LLVM_DEBUG(op->dump());

    auto eraseUnlessUsedBySimd = [&](aiir::Operation *ompOp,
                                     aiir::StringAttr name) {
      if (auto uses =
              aiir::SymbolTable::getSymbolUses(name, op->getParentOp())) {
        for (auto &use : *uses)
          if (aiir::isa<aiir::omp::SimdOp>(use.getUser()))
            return rewriter.notifyMatchFailure(op,
                                               "Op used by a simd construct");
      }
      rewriter.eraseOp(ompOp);
      return aiir::success();
    };

    if (auto ompOp = aiir::dyn_cast<aiir::omp::PrivateClauseOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());
    if (auto ompOp = aiir::dyn_cast<aiir::omp::DeclareReductionOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());

    // Might be left over from rewriting composite simd with target map
    if (aiir::isa<aiir::omp::MapBoundsOp>(op)) {
      rewriter.eraseOp(op);
      return aiir::success();
    }
    if (auto mapInfoOp = aiir::dyn_cast<aiir::omp::MapInfoOp>(op)) {
      rewriter.replaceOp(mapInfoOp, {mapInfoOp.getVarPtr()});
      return aiir::success();
    }

    // Might be leftover after parse tree rewriting
    if (auto threadPrivateOp = aiir::dyn_cast<aiir::omp::ThreadprivateOp>(op)) {
      rewriter.replaceOp(threadPrivateOp, {threadPrivateOp.getSymAddr()});
      return aiir::success();
    }

    fir::FirOpBuilder builder(rewriter, op);
    aiir::Location loc = op->getLoc();

    auto inlineSimpleOp = [&](aiir::Operation *ompOp) -> bool {
      if (!ompOp)
        return false;

      assert("OpenMP operation has one region" && ompOp->getNumRegions() == 1);

      llvm::SmallVector<std::pair<aiir::Value, aiir::BlockArgument>>
          blockArgsPairs;
      if (auto iface =
              aiir::dyn_cast<aiir::omp::BlockArgOpenMPOpInterface>(op)) {
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
                aiir::dyn_cast<aiir::omp::TerminatorOp>(block.back())) {
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
        aiir::cf::BranchOp::create(builder, loc, innerFrontBlock);
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
                  aiir::dyn_cast<aiir::omp::TerminatorOp>(innerBlock.back())) {
            builder.setInsertionPointToEnd(&innerBlock);
            aiir::cf::BranchOp::create(builder, loc, newBlock);
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
    if (inlineSimpleOp(aiir::dyn_cast<aiir::omp::TeamsOp>(op)) ||
        inlineSimpleOp(aiir::dyn_cast<aiir::omp::ParallelOp>(op)) ||
        inlineSimpleOp(aiir::dyn_cast<aiir::omp::TargetOp>(op)) ||
        inlineSimpleOp(aiir::dyn_cast<aiir::omp::WsloopOp>(op)) ||
        inlineSimpleOp(aiir::dyn_cast<aiir::omp::DistributeOp>(op)))
      return aiir::success();

    op->emitOpError("left unhandled after SimdOnly pass.");
    return aiir::failure();
  }
};

class SimdOnlyPass : public flangomp::impl::SimdOnlyPassBase<SimdOnlyPass> {

public:
  SimdOnlyPass() = default;

  void runOnOperation() override {
    aiir::ModuleOp module = getOperation();

    aiir::AIIRContext *context = &getContext();
    aiir::RewritePatternSet patterns(context);
    patterns.insert<SimdOnlyConversionPattern>(context);

    aiir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        aiir::GreedySimplifyRegionLevel::Disabled);

    if (aiir::failed(
            aiir::applyPatternsGreedily(module, std::move(patterns), config))) {
      aiir::emitError(module.getLoc(), "Error in SimdOnly conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace
