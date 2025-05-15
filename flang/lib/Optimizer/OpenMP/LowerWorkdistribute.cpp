//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of omp.workdistribute.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

/// This is the single source of truth about whether we should parallelize an
/// operation nested in an omp.workdistribute region.
static bool shouldParallelize(Operation *op) {
    // Currently we cannot parallelize operations with results that have uses
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
    if (auto callOp = dyn_cast<fir::CallOp>(op)) {
      auto callee = callOp.getCallee();
      if (!callee)
        return false;
      auto *func = op->getParentOfType<ModuleOp>().lookupSymbol(*callee);
      // TODO need to insert a check here whether it is a call we can actually
      // parallelize currently
      if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
        return true;
      return false;
    }
    // We cannot parallise anything else
    return false;
}

struct WorkdistributeToSingle : public OpRewritePattern<omp::TeamsOp> {
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
        rewriter.eraseOp(teamsOp);
        workdistributeOp.emitWarning("unable to parallelize coexecute");
        return success();
    }
};

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

struct FissionWorkdistribute
    : public OpRewritePattern<omp::WorkdistributeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(omp::WorkdistributeOp workdistribute,
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
    if (teams.getRegion().getBlocks().front().getOperations().size() != 2) {
      emitError(loc, "teams with multiple nested ops\n");
      return failure();
    }

    auto *teamsBlock = &teams.getRegion().front();

    // While we have unhandled operations in the original workdistribute
    auto *workdistributeBlock = &workdistribute.getRegion().front();
    auto *terminator = workdistributeBlock->getTerminator();
    bool changed = false;
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

class LowerWorkdistributePass
    : public flangomp::impl::LowerWorkdistributeBase<LowerWorkdistributePass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);
    GreedyRewriteConfig config;
    // prevent the pattern driver form merging blocks
    config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);
  
    patterns.insert<FissionWorkdistribute, WorkdistributeToSingle>(&context);
    Operation *op = getOperation();
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc(), DEBUG_TYPE " pass failed\n");
      signalPassFailure();
    }
  }
};
}
