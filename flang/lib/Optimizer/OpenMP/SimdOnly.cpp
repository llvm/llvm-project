//===-- SimdOnly.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Transforms/Utils.h"
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
         mlir::isa<mlir::omp::LoopNestOp>(op) ||
         mlir::isa<mlir::omp::TerminatorOp>(op)))
      return rewriter.notifyMatchFailure(op, "Op is part of a simd construct");

    if (!mlir::isa<mlir::func::FuncOp>(op->getParentOp()) &&
        (mlir::isa<mlir::omp::TerminatorOp>(op) ||
         mlir::isa<mlir::omp::YieldOp>(op)))
      return rewriter.notifyMatchFailure(op,
                                         "Non top-level yield or terminator");

    if (mlir::isa<mlir::omp::UnrollHeuristicOp>(op))
      return rewriter.notifyMatchFailure(
          op, "UnrollHeuristic has special handling");

    // SectionOp overrides its BlockArgInterface based on the parent SectionsOp.
    // We need to make sure we only rewrite omp.sections once all omp.section
    // ops inside it have been rewritten, otherwise the individual omp.section
    // ops will not be able to access their argument values.
    if (auto sectionsOp = mlir::dyn_cast<mlir::omp::SectionsOp>(op)) {
      for (auto &opInSections : sectionsOp.getRegion().getOps())
        if (mlir::isa<mlir::omp::SectionOp>(opInSections))
          return rewriter.notifyMatchFailure(
              op, "SectionsOp still contains individual sections");
    }

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
    if (auto ompOp = mlir::dyn_cast<mlir::omp::CriticalDeclareOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());
    if (auto ompOp = mlir::dyn_cast<mlir::omp::DeclareMapperOp>(op))
      return eraseUnlessUsedBySimd(ompOp, ompOp.getSymNameAttr());

    // Erase ops that don't need any special handling
    if (mlir::isa<mlir::omp::BarrierOp>(op) ||
        mlir::isa<mlir::omp::FlushOp>(op) ||
        mlir::isa<mlir::omp::TaskyieldOp>(op) ||
        mlir::isa<mlir::omp::MapBoundsOp>(op) ||
        mlir::isa<mlir::omp::TargetEnterDataOp>(op) ||
        mlir::isa<mlir::omp::TargetExitDataOp>(op) ||
        mlir::isa<mlir::omp::TargetUpdateOp>(op) ||
        mlir::isa<mlir::omp::OrderedOp>(op) ||
        mlir::isa<mlir::omp::CancelOp>(op) ||
        mlir::isa<mlir::omp::CancellationPointOp>(op) ||
        mlir::isa<mlir::omp::ScanOp>(op) ||
        mlir::isa<mlir::omp::TaskwaitOp>(op)) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    fir::FirOpBuilder builder(rewriter, op);
    mlir::Location loc = op->getLoc();

    if (auto ompOp = mlir::dyn_cast<mlir::omp::LoopNestOp>(op)) {
      mlir::Type indexType = builder.getIndexType();
      mlir::Type oldIndexType = ompOp.getIVs().begin()->getType();
      builder.setInsertionPoint(op);
      auto one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);

      // Generate the new loop nest
      mlir::Block *nestBody = nullptr;
      fir::DoLoopOp outerLoop = nullptr;
      llvm::SmallVector<mlir::Value> loopIndArgs;
      for (auto extent : ompOp.getLoopUpperBounds()) {
        auto ub = builder.createConvert(loc, indexType, extent);
        auto doLoop = fir::DoLoopOp::create(builder, loc, one, ub, one, false);
        nestBody = doLoop.getBody();
        builder.setInsertionPointToStart(nestBody);
        // Convert the indices to the type used inside the loop if needed
        if (oldIndexType != indexType) {
          auto convertedIndVar = builder.createConvert(
              loc, oldIndexType, doLoop.getInductionVar());
          loopIndArgs.push_back(convertedIndVar);
        } else {
          loopIndArgs.push_back(doLoop.getInductionVar());
        }
        if (!outerLoop)
          outerLoop = doLoop;
      }

      // Move the omp loop body into the new loop body
      if (ompOp->getRegion(0).getBlocks().size() == 1) {
        auto &block = *ompOp->getRegion(0).getBlocks().begin();
        rewriter.mergeBlocks(&block, nestBody, loopIndArgs);

        // Find the new loop block terminator and move it before the end of the
        // block
        for (auto &loopBodyOp : nestBody->getOperations()) {
          if (auto resultOp = mlir::dyn_cast<fir::ResultOp>(loopBodyOp)) {
            rewriter.moveOpBefore(resultOp.getOperation(), &nestBody->back());
            break;
          }
        }

        // Remove omp.yield at the end of the loop body
        if (auto yieldOp =
                mlir::dyn_cast<mlir::omp::YieldOp>(nestBody->back())) {
          assert("omp.loop_nests's omp.yield has no operands" &&
                 yieldOp->getNumOperands() == 0);
          rewriter.eraseOp(yieldOp);
        }
      } else {
        rewriter.inlineRegionBefore(ompOp->getRegion(0), nestBody);
        auto indVarArg = outerLoop->getRegion(0).front().getArgument(0);
        // fir::convertDoLoopToCFG expects the induction variable to be of type
        // index while the OpenMP LoopNestOp can have indices of different
        // types. We need to work around it.
        if (indVarArg.getType() != indexType)
          indVarArg.setType(indexType);

        // fir.do_loop, unlike omp.loop_nest does not support multi-block
        // regions. If we're dealing with multiple blocks inside omp.loop_nest,
        // we need to convert it into basic control-flow operations instead.
        auto loopBlocks =
            fir::convertDoLoopToCFG(outerLoop, rewriter, false, false);
        auto *conditionalBlock = loopBlocks.first;
        auto *firstBlock =
            conditionalBlock->getNextNode(); // Start of the loop body
        auto *lastBlock = loopBlocks.second; // Incrementing induction variables

        // If the induction variable is used within the loop and was originally
        // not of type index, then we need to add a convert to the original type
        // and replace its uses inside the loop body.
        if (oldIndexType != indexType) {
          indVarArg = conditionalBlock->getArgument(0);
          builder.setInsertionPointToStart(firstBlock);
          auto convertedIndVar =
              builder.createConvert(loc, oldIndexType, indVarArg);
          rewriter.replaceUsesWithIf(
              indVarArg, convertedIndVar, [&](auto &use) -> bool {
                return use.getOwner() != convertedIndVar.getDefiningOp() &&
                       use.getOwner()->getBlock() != lastBlock;
              });
        }

        // There might be an unused convert and an unused argument to the block.
        // If so, remove them.
        if (lastBlock->front().getUses().empty())
          lastBlock->front().erase();
        for (auto arg : lastBlock->getArguments()) {
          if (arg.getUses().empty())
            lastBlock->eraseArgument(arg.getArgNumber());
        }

        // Any loop blocks that end in omp.yield should just branch to
        // lastBlock.
        for (auto *loopBlock = conditionalBlock; loopBlock != lastBlock;
             loopBlock = loopBlock->getNextNode()) {
          if (auto yieldOp =
                  mlir::dyn_cast<mlir::omp::YieldOp>(loopBlock->back())) {
            builder.setInsertionPointToEnd(loopBlock);
            mlir::cf::BranchOp::create(builder, loc, lastBlock);
            assert("omp.loop_nests's omp.yield has no operands" &&
                   yieldOp->getNumOperands() == 0);
            rewriter.eraseOp(yieldOp);
          }
        }
      }

      rewriter.eraseOp(ompOp);
      return mlir::success();
    }

    if (auto mapInfoOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(op)) {
      mapInfoOp.getResult().replaceAllUsesWith(mapInfoOp.getVarPtr());
      rewriter.eraseOp(mapInfoOp);
      return mlir::success();
    }

    if (auto atomicReadOp = mlir::dyn_cast<mlir::omp::AtomicReadOp>(op)) {
      builder.setInsertionPoint(op);
      auto loadOp = fir::LoadOp::create(builder, loc, atomicReadOp.getX());
      auto storeOp = fir::StoreOp::create(builder, loc, loadOp.getResult(),
                                          atomicReadOp.getV());
      rewriter.replaceOp(op, storeOp);
      return mlir::success();
    }

    if (auto atomicWriteOp = mlir::dyn_cast<mlir::omp::AtomicWriteOp>(op)) {
      auto storeOp = fir::StoreOp::create(builder, loc, atomicWriteOp.getExpr(),
                                          atomicWriteOp.getX());
      rewriter.replaceOp(op, storeOp);
      return mlir::success();
    }

    if (auto atomicUpdateOp = mlir::dyn_cast<mlir::omp::AtomicUpdateOp>(op)) {
      assert("one block in region" &&
             atomicUpdateOp.getRegion().getBlocks().size() == 1);
      auto &block = *atomicUpdateOp.getRegion().getBlocks().begin();
      builder.setInsertionPointToStart(&block);

      // Load the update `x` operand and replace its uses within the block
      auto loadOp = fir::LoadOp::create(builder, loc, atomicUpdateOp.getX());
      rewriter.replaceUsesWithIf(
          block.getArgument(0), loadOp.getResult(),
          [&](auto &op) { return op.get().getParentBlock() == &block; });

      // Store the result back into `x` in line with omp.yield semantics for
      // this op
      auto yieldOp = mlir::cast<mlir::omp::YieldOp>(block.back());
      assert("only one yield operand" && yieldOp->getNumOperands() == 1);
      builder.setInsertionPointAfter(yieldOp);
      fir::StoreOp::create(builder, loc, yieldOp->getOperand(0),
                           atomicUpdateOp.getX());
      rewriter.eraseOp(yieldOp);

      // Inline the final block and remove the now-empty op
      assert("only one block argument" && block.getNumArguments() == 1);
      block.eraseArguments(0, block.getNumArguments());
      rewriter.inlineBlockBefore(&block, atomicUpdateOp, {});
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (auto threadPrivateOp = mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op)) {
      threadPrivateOp.getTlsAddr().replaceAllUsesWith(
          threadPrivateOp.getSymAddr());
      rewriter.eraseOp(threadPrivateOp);
      return mlir::success();
    }

    if (auto cLoopOp = mlir::dyn_cast<mlir::omp::CanonicalLoopOp>(op)) {
      assert("CanonicalLoopOp has one region" && cLoopOp->getNumRegions() == 1);
      auto cli = cLoopOp.getCli();
      auto tripCount = cLoopOp.getTripCount();

      builder.setInsertionPoint(cLoopOp);
      mlir::Type indexType = builder.getIndexType();
      mlir::Type oldIndexType = tripCount.getType();
      auto one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      auto ub = builder.createConvert(loc, indexType, tripCount);

      llvm::SmallVector<mlir::Value> loopIndArgs;
      auto doLoop = fir::DoLoopOp::create(builder, loc, one, ub, one, false);
      builder.setInsertionPointToStart(doLoop.getBody());
      if (oldIndexType != indexType) {
        auto convertedIndVar =
            builder.createConvert(loc, oldIndexType, doLoop.getInductionVar());
        loopIndArgs.push_back(convertedIndVar);
      } else {
        loopIndArgs.push_back(doLoop.getInductionVar());
      }

      if (cLoopOp.getRegion().getBlocks().size() == 1) {
        auto &block = *cLoopOp.getRegion().getBlocks().begin();
        // DoLoopOp will handle incrementing the induction variable
        if (auto addIOp = mlir::dyn_cast<mlir::arith::AddIOp>(block.front())) {
          rewriter.replaceOpUsesWithinBlock(addIOp, addIOp.getLhs(), &block);
          rewriter.eraseOp(addIOp);
        }

        rewriter.mergeBlocks(&block, doLoop.getBody(), loopIndArgs);

        // Find the new loop block terminator and move it before the end of the
        // block
        for (auto &loopBodyOp : doLoop.getBody()->getOperations()) {
          if (auto resultOp = mlir::dyn_cast<fir::ResultOp>(loopBodyOp)) {
            rewriter.moveOpBefore(resultOp.getOperation(),
                                  &doLoop.getBody()->back());
            break;
          }
        }

        // Remove omp.terminator at the end of the loop body
        if (auto terminatorOp = mlir::dyn_cast<mlir::omp::TerminatorOp>(
                doLoop.getBody()->back())) {
          rewriter.eraseOp(terminatorOp);
        }
      } else {
        rewriter.inlineRegionBefore(cLoopOp->getRegion(0), doLoop.getBody());
        auto indVarArg = doLoop.getBody()->getArgument(0);
        // fir::convertDoLoopToCFG expects the induction variable to be of type
        // index while the OpenMP CanonicalLoopOp can have indices of different
        // types. We need to work around it.
        if (indVarArg.getType() != indexType)
          indVarArg.setType(indexType);

        // fir.do_loop, unlike omp.canonical_loop does not support multi-block
        // regions. If we're dealing with multiple blocks inside omp.loop_nest,
        // we need to convert it into basic control-flow operations instead.
        auto loopBlocks =
            fir::convertDoLoopToCFG(doLoop, rewriter, false, false);
        auto *conditionalBlock = loopBlocks.first;
        auto *firstBlock =
            conditionalBlock->getNextNode(); // Start of the loop body
        auto *lastBlock = loopBlocks.second; // Incrementing induction variables

        // Incrementing the induction variable is handled elsewhere
        if (auto addIOp =
                mlir::dyn_cast<mlir::arith::AddIOp>(firstBlock->front())) {
          rewriter.replaceOpUsesWithinBlock(addIOp, addIOp.getLhs(),
                                            firstBlock);
          rewriter.eraseOp(addIOp);
        }

        // If the induction variable is used within the loop and was originally
        // not of type index, then we need to add a convert to the original type
        // and replace its uses inside the loop body.
        if (oldIndexType != indexType) {
          indVarArg = conditionalBlock->getArgument(0);
          builder.setInsertionPointToStart(firstBlock);
          auto convertedIndVar =
              builder.createConvert(loc, oldIndexType, indVarArg);
          rewriter.replaceUsesWithIf(
              indVarArg, convertedIndVar, [&](auto &use) -> bool {
                return use.getOwner() != convertedIndVar.getDefiningOp() &&
                       use.getOwner()->getBlock() != lastBlock;
              });
        }

        // There might be an unused convert and an unused argument to the block.
        // If so, remove them.
        if (lastBlock->front().getUses().empty())
          lastBlock->front().erase();
        for (auto arg : lastBlock->getArguments()) {
          if (arg.getUses().empty())
            lastBlock->eraseArgument(arg.getArgNumber());
        }

        // Any loop blocks that end in omp.terminator should just branch to
        // lastBlock.
        for (auto *loopBlock = conditionalBlock; loopBlock != lastBlock;
             loopBlock = loopBlock->getNextNode()) {
          if (auto terminatorOp =
                  mlir::dyn_cast<mlir::omp::TerminatorOp>(loopBlock->back())) {
            builder.setInsertionPointToEnd(loopBlock);
            mlir::cf::BranchOp::create(builder, loc, lastBlock);
            rewriter.eraseOp(terminatorOp);
          }
        }
      }

      rewriter.eraseOp(cLoopOp);
      // Handle the optional omp.new_cli op
      if (cli) {
        // cli will be used by omp.unroll_heuristic ops
        for (auto *user : cli.getUsers())
          rewriter.eraseOp(user);
        rewriter.eraseOp(cli.getDefiningOp());
      }

      return mlir::success();
    }

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

    if (inlineSimpleOp(mlir::dyn_cast<mlir::omp::TeamsOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::ParallelOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::SingleOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::SectionOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::SectionsOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::WsloopOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::LoopOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::TargetOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::TargetDataOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::DistributeOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::TaskOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::TaskloopOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::MasterOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::CriticalOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::OrderedRegionOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::AtomicCaptureOp>(op)) ||
        inlineSimpleOp(mlir::dyn_cast<mlir::omp::MaskedOp>(op)))
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
      mlir::emitError(module.getLoc(), "error in simd-only conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace
