//===-- AffinePromotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#define DEBUG_TYPE "flang-affine-promotion"

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    disableAffinePromo("disable-affine-promotion",
                       llvm::cl::desc("disable FIR to Affine pass"),
                       llvm::cl::init(true));

using namespace fir;

namespace {

class AffineLoopAnalysis {
public:
  AffineLoopAnalysis(fir::LoopOp op) : legality(analyzeLoop(op)) {}
  bool canPromoteToAffine() { return legality; }
  Optional<int64_t> step;

private:
  bool legality;
  bool analyzeLoop(fir::LoopOp op) {
    return analyzeStep(op.step()) && analyzeMemoryAccess(op);
  }
  bool analyzeStep(const mlir::Value stepValue) {
    auto stepDefinition = stepValue.getDefiningOp<ConstantOp>();
    if (stepDefinition) {
      if (auto stepAttr = stepDefinition.getValue().dyn_cast<IntegerAttr>()) {
        step = stepAttr.getInt();
        return true;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                   "step not integer\n";
                   stepAttr.print(llvm::dbgs()););
        return false;
      }
    } else {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineLoopAnalysis: cannot promote loop, step not constant\n";
          if (stepValue.getDefiningOp()) stepValue.getDefiningOp()->print(
              llvm::dbgs()));
      return false;
    }
  }
  bool analyzeMemoryAccess(fir::LoopOp loop) {
    llvm_unreachable("not yet implemented");
    return true;
  }
};

/// builds analysis for all loop operations within a function
class AffineFunctionAnalysis {
public:
  AffineFunctionAnalysis(mlir::FuncOp funcOp) {
    for (fir::LoopOp op : funcOp.getOps<fir::LoopOp>()) {
      loopAnalysisMap.try_emplace(op, op);
    }
  }
  AffineLoopAnalysis getChildLoopAnalysis(fir::LoopOp op) const {
    auto it = loopAnalysisMap.find_as(op);
    if (it == loopAnalysisMap.end()) {
      op.emitError("error in fetching loop analysis during affine promotion\n");
    } else {
      return it->getSecond();
    }
  }

private:
  DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
};

/// Convert `fir.loop` to `affine.for`
class AffineLoopConversion : public mlir::OpRewritePattern<fir::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineLoopConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}

  mlir::LogicalResult
  matchAndRewrite(fir::LoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    auto loopAnalysis = functionAnalysis.getChildLoopAnalysis(loop);
    if (loopAnalysis.step.getValue() <= 0) {
      LLVM_DEBUG(llvm::dbgs()
                     << "AffineLoopAnalysis: cannot promote loop for now, "
                        "step not postive\n";);
      return failure();
    }
    auto loopOps = &loop.getBody()->getOperations();
    for (auto loadOp : loop.getOps<fir::LoadOp>()) {
      if (failed(rewriteLoad(loadOp, rewriter)))
        return failure();
    }
    auto affineFor = rewriter.create<mlir::AffineForOp>(
        loop.getLoc(), ValueRange(loop.lowerBound()),
        AffineMap::getMultiDimIdentityMap(1, loop.getContext()),
        ValueRange(loop.upperBound()),
        AffineMap::getMultiDimIdentityMap(1, loop.getContext()),
        loopAnalysis.step.getValue());
    loop.step().dropAllUses();
    loop.getBody()->getOperations().pop_back(); // remove fir.result
    affineFor.getBody()->getOperations().splice(affineFor.getBody()->begin(),
                                                *loopOps, loopOps->begin(),
                                                loopOps->end());
    rewriter.eraseOp(loop);
    loop.getInductionVar().replaceAllUsesWith(affineFor.getInductionVar());
    return success();
  }

private:
  mlir::LogicalResult rewriteLoad(fir::LoadOp op,
                                  mlir::PatternRewriter &rewriter) const {
    return success();
  }
  AffineFunctionAnalysis &functionAnalysis;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public AffineDialectPromotionBase<AffineDialectPromotion> {
public:
  void runOnFunction() override {
    if (disableAffinePromo)
      return;

    auto *context = &getContext();
    auto function = getFunction();
    auto functionAnalysis = AffineFunctionAnalysis(function);
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConversion>(context, functionAnalysis);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::scf::SCFDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<LoopOp>([&functionAnalysis](fir::LoopOp op) {
      return !(functionAnalysis.getChildLoopAnalysis(op).canPromoteToAffine());
    });
    LLVM_DEBUG(llvm::dbgs()
                   << "AffineDialectPromotion: running promotion on: \n";
               function.print(llvm::dbgs()););
    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};
} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}
