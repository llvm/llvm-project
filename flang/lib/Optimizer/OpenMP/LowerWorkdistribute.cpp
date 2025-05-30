//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and optimisations of omp.workdistribute.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute"

using namespace mlir;

namespace {

static bool isRuntimeCall(Operation *op) {
  if (auto callOp = dyn_cast<fir::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    auto *func = op->getParentOfType<ModuleOp>().lookupSymbol(*callee);
    if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
      return true;
  }
  return false;
}

/// This is the single source of truth about whether we should parallelize an
/// operation nested in an omp.execute region.
static bool shouldParallelize(Operation *op) {
  if (llvm::any_of(op->getResults(),
                   [](OpResult v) -> bool { return !v.use_empty(); }))
    return false;
  // We will parallelize unordered loops - these come from array syntax
  if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
    auto unordered = loop.getUnordered();
    if (!unordered)
      return false;
    return *unordered;
  }
  if (isRuntimeCall(op)) {
    return true;
  }
  // We cannot parallise anything else
  return false;
}

template <typename T>
static T getPerfectlyNested(Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  auto &region = op->getRegion(0);
  if (region.getBlocks().size() != 1)
    return nullptr;
  auto *block = &region.front();
  auto *firstOp = &block->front();
  if (auto nested = dyn_cast<T>(firstOp))
    if (firstOp->getNextNode() == block->getTerminator())
      return nested;
  return nullptr;
}

/// If B() and D() are parallelizable,
///
/// omp.teams {
///   omp.workdistribute {
///     A()
///     B()
///     C()
///     D()
///     E()
///   }
/// }
///
/// becomes
///
/// A()
/// omp.teams {
///   omp.workdistribute {
///     B()
///   }
/// }
/// C()
/// omp.teams {
///   omp.workdistribute {
///     D()
///   }
/// }
/// E()

struct FissionWorkdistribute : public OpRewritePattern<omp::WorkdistributeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::WorkdistributeOp workdistribute,
                                PatternRewriter &rewriter) const override {
    auto loc = workdistribute->getLoc();
    auto teams = dyn_cast<omp::TeamsOp>(workdistribute->getParentOp());
    if (!teams) {
      emitError(loc, "workdistribute not nested in teams\n");
      return failure();
    }
    if (workdistribute.getRegion().getBlocks().size() != 1) {
      emitError(loc, "workdistribute with multiple blocks\n");
      return failure();
    }
    if (teams.getRegion().getBlocks().size() != 1) {
      emitError(loc, "teams with multiple blocks\n");
      return failure();
    }

    auto *teamsBlock = &teams.getRegion().front();
    bool changed = false;
    // Move the ops inside teams and before workdistribute outside.
    IRMapping irMapping;
    llvm::SmallVector<Operation *> teamsHoisted;
    for (auto &op : teams.getOps()) {
      if (&op == workdistribute) {
        break;
      }
      if (shouldParallelize(&op)) {
        emitError(loc,
                  "teams has parallelize ops before first workdistribute\n");
        return failure();
      } else {
        rewriter.setInsertionPoint(teams);
        rewriter.clone(op, irMapping);
        teamsHoisted.push_back(&op);
        changed = true;
      }
    }
    for (auto *op : teamsHoisted)
      rewriter.replaceOp(op, irMapping.lookup(op));

    // While we have unhandled operations in the original workdistribute
    auto *workdistributeBlock = &workdistribute.getRegion().front();
    auto *terminator = workdistributeBlock->getTerminator();
    while (&workdistributeBlock->front() != terminator) {
      rewriter.setInsertionPoint(teams);
      IRMapping mapping;
      llvm::SmallVector<Operation *> hoisted;
      Operation *parallelize = nullptr;
      for (auto &op : workdistribute.getOps()) {
        if (&op == terminator) {
          break;
        }
        if (shouldParallelize(&op)) {
          parallelize = &op;
          break;
        } else {
          rewriter.clone(op, mapping);
          hoisted.push_back(&op);
          changed = true;
        }
      }

      for (auto *op : hoisted)
        rewriter.replaceOp(op, mapping.lookup(op));

      if (parallelize && hoisted.empty() &&
          parallelize->getNextNode() == terminator)
        break;
      if (parallelize) {
        auto newTeams = rewriter.cloneWithoutRegions(teams);
        auto *newTeamsBlock = rewriter.createBlock(
            &newTeams.getRegion(), newTeams.getRegion().begin(), {}, {});
        for (auto arg : teamsBlock->getArguments())
          newTeamsBlock->addArgument(arg.getType(), arg.getLoc());
        auto newWorkdistribute = rewriter.create<omp::WorkdistributeOp>(loc);
        rewriter.create<omp::TerminatorOp>(loc);
        rewriter.createBlock(&newWorkdistribute.getRegion(),
                             newWorkdistribute.getRegion().begin(), {}, {});
        auto *cloned = rewriter.clone(*parallelize);
        rewriter.replaceOp(parallelize, cloned);
        rewriter.create<omp::TerminatorOp>(loc);
        changed = true;
      }
    }
    return success(changed);
  }
};

/// If fir.do_loop is present inside teams workdistribute
///
/// omp.teams {
///   omp.workdistribute {
///     fir.do_loop unoredered {
///       ...
///     }
///   }
/// }
///
/// Then, its lowered to
///
/// omp.teams {
///   omp.parallel {
///     omp.distribute {
///     omp.wsloop {
///       omp.loop_nest
///         ...
///       }
///     }
///   }
/// }

static void genParallelOp(Location loc, PatternRewriter &rewriter,
                          bool composite) {
  auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
  parallelOp.setComposite(composite);
  rewriter.createBlock(&parallelOp.getRegion());
  rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));
  return;
}

static void genDistributeOp(Location loc, PatternRewriter &rewriter,
                            bool composite) {
  mlir::omp::DistributeOperands distributeClauseOps;
  auto distributeOp =
      rewriter.create<mlir::omp::DistributeOp>(loc, distributeClauseOps);
  distributeOp.setComposite(composite);
  auto distributeBlock = rewriter.createBlock(&distributeOp.getRegion());
  rewriter.setInsertionPointToStart(distributeBlock);
  return;
}

static void
genLoopNestClauseOps(mlir::PatternRewriter &rewriter, fir::DoLoopOp loop,
                     mlir::omp::LoopNestOperands &loopNestClauseOps) {
  assert(loopNestClauseOps.loopLowerBounds.empty() &&
         "Loop nest bounds were already emitted!");
  loopNestClauseOps.loopLowerBounds.push_back(loop.getLowerBound());
  loopNestClauseOps.loopUpperBounds.push_back(loop.getUpperBound());
  loopNestClauseOps.loopSteps.push_back(loop.getStep());
  loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
}

static void genWsLoopOp(mlir::PatternRewriter &rewriter, fir::DoLoopOp doLoop,
                        const mlir::omp::LoopNestOperands &clauseOps,
                        bool composite) {

  auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(doLoop.getLoc());
  wsloopOp.setComposite(composite);
  rewriter.createBlock(&wsloopOp.getRegion());

  auto loopNestOp =
      rewriter.create<mlir::omp::LoopNestOp>(doLoop.getLoc(), clauseOps);

  // Clone the loop's body inside the loop nest construct using the
  // mapped values.
  rewriter.cloneRegionBefore(doLoop.getRegion(), loopNestOp.getRegion(),
                             loopNestOp.getRegion().begin());
  Block *clonedBlock = &loopNestOp.getRegion().back();
  mlir::Operation *terminatorOp = clonedBlock->getTerminator();

  // Erase fir.result op of do loop and create yield op.
  if (auto resultOp = dyn_cast<fir::ResultOp>(terminatorOp)) {
    rewriter.setInsertionPoint(terminatorOp);
    rewriter.create<mlir::omp::YieldOp>(doLoop->getLoc());
    rewriter.eraseOp(terminatorOp);
  }
  return;
}

struct WorkdistributeDoLower : public OpRewritePattern<omp::WorkdistributeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::WorkdistributeOp workdistribute,
                                PatternRewriter &rewriter) const override {
    auto doLoop = getPerfectlyNested<fir::DoLoopOp>(workdistribute);
    auto wdLoc = workdistribute->getLoc();
    if (doLoop && shouldParallelize(doLoop)) {
      assert(doLoop.getReduceOperands().empty());
      genParallelOp(wdLoc, rewriter, true);
      genDistributeOp(wdLoc, rewriter, true);
      mlir::omp::LoopNestOperands loopNestClauseOps;
      genLoopNestClauseOps(rewriter, doLoop, loopNestClauseOps);
      genWsLoopOp(rewriter, doLoop, loopNestClauseOps, true);
      rewriter.eraseOp(workdistribute);
      return success();
    }
    return failure();
  }
};

/// If A() and B () are present inside teams workdistribute
///
/// omp.teams {
///   omp.workdistribute {
///     A()
///     B()
///   }
/// }
///
/// Then, its lowered to
///
/// A()
/// B()
///

struct TeamsWorkdistributeToSingle : public OpRewritePattern<omp::TeamsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::TeamsOp teamsOp,
                                PatternRewriter &rewriter) const override {
    auto workdistributeOp = getPerfectlyNested<omp::WorkdistributeOp>(teamsOp);
    if (!workdistributeOp) {
      LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " No workdistribute nested\n");
      return failure();
    }
    Block *workdistributeBlock = &workdistributeOp.getRegion().front();
    rewriter.eraseOp(workdistributeBlock->getTerminator());
    rewriter.inlineBlockBefore(workdistributeBlock, teamsOp);
    rewriter.eraseOp(workdistributeOp);
    return success();
  }
};

static std::optional<std::tuple<Operation *, bool, bool>>
getNestedOpToIsolate(omp::TargetOp targetOp) {
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto &op : *targetBlock) {
    bool first = &op == &*targetBlock->begin();
    bool last = op.getNextNode() == targetBlock->getTerminator();
    if (first && last)
      return std::nullopt;

    if (isa<omp::TeamsOp, omp::ParallelOp>(&op))
      return {{&op, first, last}};
  }
  return std::nullopt;
}

struct SplitTargetResult {
  omp::TargetOp targetOp;
  omp::TargetDataOp dataOp;
};

/// If multiple coexecutes are nested in a target regions, we will need to split
/// the target region, but we want to preserve the data semantics of the
/// original data region and avoid unnecessary data movement at each of the
/// subkernels - we split the target region into a target_data{target}
/// nest where only the outer one moves the data
std::optional<SplitTargetResult> splitTargetData(omp::TargetOp targetOp,
                                                 RewriterBase &rewriter) {

  auto loc = targetOp->getLoc();
  if (targetOp.getMapVars().empty()) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " target region has no data maps\n");
    return std::nullopt;
  }

  // Collect all map_entries with capture(ByRef)
  SmallVector<mlir::Value> byRefMapInfos;
  SmallVector<omp::MapInfoOp> MapInfos;
  for (auto opr : targetOp.getMapVars()) {
    auto mapInfo = cast<omp::MapInfoOp>(opr.getDefiningOp());
    MapInfos.push_back(mapInfo);
    if (mapInfo.getMapCaptureType() == omp::VariableCaptureKind::ByRef)
      byRefMapInfos.push_back(opr);
  }

  // Create the new omp.target_data op with these collected map_entries
  auto targetLoc = targetOp.getLoc();
  rewriter.setInsertionPoint(targetOp);
  auto device = targetOp.getDevice();
  auto ifExpr = targetOp.getIfExpr();
  auto deviceAddrVars = targetOp.getHasDeviceAddrVars();
  auto devicePtrVars = targetOp.getIsDevicePtrVars();
  auto targetDataOp = rewriter.create<omp::TargetDataOp>(loc, device, ifExpr, 
                                                          mlir::ValueRange{byRefMapInfos},
                                                          deviceAddrVars,
                                                          devicePtrVars);

  auto taregtDataBlock = rewriter.createBlock(&targetDataOp.getRegion());
  rewriter.create<mlir::omp::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(taregtDataBlock);

  // Clone mapInfo ops inside omp.target_data region
  IRMapping mapping;
  for (auto mapInfo : MapInfos) {
    rewriter.clone(*mapInfo, mapping);
  }
  // Clone omp.target from exisiting targetOp inside target_data region.
  auto newTargetOp = rewriter.clone(*targetOp, mapping);

  // Erase TargetOp and its MapInfoOps
  rewriter.eraseOp(targetOp);
  
  for (auto mapInfo : MapInfos) {
    auto mapInfoRes = mapInfo.getResult();
    if (mapInfoRes.getUsers().empty()) 
      rewriter.eraseOp(mapInfo);
  }
  return SplitTargetResult{targetOp, targetDataOp};
}                                                  

class LowerWorkdistributePass
    : public flangomp::impl::LowerWorkdistributeBase<LowerWorkdistributePass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    GreedyRewriteConfig config;
    // prevent the pattern driver form merging blocks
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    Operation *op = getOperation();
    {
      RewritePatternSet patterns(&context);
      patterns.insert<FissionWorkdistribute, WorkdistributeDoLower>(&context);
      if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
        emitError(op->getLoc(), DEBUG_TYPE " pass failed\n");
        signalPassFailure();
      }
    }
    {
      RewritePatternSet patterns(&context);
      patterns.insert<TeamsWorkdistributeToSingle>(&context);
      if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
        emitError(op->getLoc(), DEBUG_TYPE " pass failed\n");
        signalPassFailure();
      }
    }
    {
      SmallVector<omp::TargetOp> targetOps;
      op->walk([&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
      IRRewriter rewriter(&context);
      for (auto targetOp : targetOps) {
        auto res = splitTargetData(targetOp, rewriter);
      }
    }

  }
};
} // namespace
