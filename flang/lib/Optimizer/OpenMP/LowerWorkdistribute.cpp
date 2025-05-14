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

#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Dialect/FIROps.h>
#include <flang/Optimizer/Dialect/FIRType.h>
#include <flang/Optimizer/HLFIR/HLFIROps.h>
#include <flang/Optimizer/OpenMP/Passes.h>
#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute"

using namespace mlir;

namespace {

struct WorkdistributeToSingle : public mlir::OpRewritePattern<mlir::omp::WorkdistributeOp> {
using OpRewritePattern::OpRewritePattern;
mlir::LogicalResult
    matchAndRewrite(mlir::omp::WorkdistributeOp workdistribute,
                       mlir::PatternRewriter &rewriter) const override {
        auto loc = workdistribute->getLoc();
        auto teams = llvm::dyn_cast<mlir::omp::TeamsOp>(workdistribute->getParentOp());
        if (!teams) {
            mlir::emitError(loc, "workdistribute not nested in teams\n");
            return mlir::failure();
        }
        if (workdistribute.getRegion().getBlocks().size() != 1) {
            mlir::emitError(loc, "workdistribute with multiple blocks\n");
            return mlir::failure();
        }
        if (teams.getRegion().getBlocks().size() != 1) {
            mlir::emitError(loc, "teams with multiple blocks\n");
           return mlir::failure();
        }
        if (teams.getRegion().getBlocks().front().getOperations().size() != 2) {
            mlir::emitError(loc, "teams with multiple nested ops\n");
            return mlir::failure();
        }
        mlir::Block *workdistributeBlock = &workdistribute.getRegion().front();
        rewriter.eraseOp(workdistributeBlock->getTerminator());
        rewriter.inlineBlockBefore(workdistributeBlock, teams);
        rewriter.eraseOp(teams);
        return mlir::success();
    }
};

class LowerWorkdistributePass
    : public flangomp::impl::LowerWorkdistributeBase<LowerWorkdistributePass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    mlir::GreedyRewriteConfig config;
    // prevent the pattern driver form merging blocks
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
  
    patterns.insert<WorkdistributeToSingle>(&context);
    mlir::Operation *op = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns), config))) {
      mlir::emitError(op->getLoc(), DEBUG_TYPE " pass failed\n");
      signalPassFailure();
    }
  }
};
}
