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
#include "flang/Optimizer/Dialect/FIRType.h"
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
class AffineFunctionAnalysis;
class AffineLoopAnalysis;

class AffineLoopAnalysis {
public:
  AffineLoopAnalysis(fir::LoopOp op, AffineFunctionAnalysis &afa)
      : legality(analyzeLoop(op, afa)) {}
  bool canPromoteToAffine() { return legality; }
  friend AffineFunctionAnalysis;

private:
  bool legality;
  struct MemoryLoadAnalysis {};
  DenseMap<Operation *, MemoryLoadAnalysis> loadAnalysis;
  AffineLoopAnalysis(bool forcedLegality) : legality(forcedLegality) {}
  bool analyzeBody(fir::LoopOp, AffineFunctionAnalysis &);
  bool analyzeLoop(fir::LoopOp loopOperation,
                   AffineFunctionAnalysis &functionAnalysis) {
    LLVM_DEBUG(llvm::dbgs() << "AffinLoopAnalysis: \n"; loopOperation.dump(););
    return analyzeMemoryAccess(loopOperation) &&
           analyzeBody(loopOperation, functionAnalysis);
  }
  bool analyzeArrayReference(mlir::Value);
  bool analyzeMemoryAccess(fir::LoopOp loopOperation) {
    for (auto loadOp : loopOperation.getOps<fir::LoadOp>())
      if (!analyzeArrayReference(loadOp.memref()))
        return false;
    for (auto storeOp : loopOperation.getOps<fir::StoreOp>())
      if (!analyzeArrayReference(storeOp.memref()))
        return false;
    return true;
  }
};

/// builds analysis for all loop operations within a function
class AffineFunctionAnalysis {
public:
  AffineFunctionAnalysis(mlir::FuncOp funcOp) {
    for (fir::LoopOp op : funcOp.getOps<fir::LoopOp>())
      loopAnalysisMap.try_emplace(op, op, *this);
  }
  AffineLoopAnalysis getChildLoopAnalysis(fir::LoopOp op) const {
    auto it = loopAnalysisMap.find_as(op);
    if (it == loopAnalysisMap.end()) {
      LLVM_DEBUG(llvm::dbgs() << "AffineFunctionAnalysis: not computed for:\n";
                 op.dump(););
      op.emitError(
          "error in fetching loop analysis in AffineFunctionAnalysis\n");
      return AffineLoopAnalysis(false);
    }
    return it->getSecond();
  }
  friend AffineLoopAnalysis;

private:
  llvm::DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
};
bool analyzeCoordinate(mlir::Value coordinate) {
  if (auto blockArg = coordinate.dyn_cast<mlir::BlockArgument>()) {
    if (isa<fir::LoopOp>(blockArg.getOwner()->getParentOp())) {
      return true;
    } else {
      llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a "
                      "loop induction variable (owner not loopOp)\n";
      return false;
    }
  } else {
    llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a loop "
                    "induction variable (not a block argument)\n";
    return false;
  }
}
bool AffineLoopAnalysis::analyzeArrayReference(mlir::Value arrayRef) {
  bool canPromote = true;
  if (auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>()) {
    for (auto coordinate : acoOp.coor())
      canPromote = canPromote && analyzeCoordinate(coordinate);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                               "array reference uses non ArrayCoorOp\n";);
    canPromote = false;
  }
  return canPromote;
}

bool AffineLoopAnalysis::analyzeBody(fir::LoopOp loopOperation,
                                     AffineFunctionAnalysis &functionAnalysis) {
  for (auto loopOp : loopOperation.getOps<fir::LoopOp>()) {
    auto analysis = functionAnalysis.loopAnalysisMap
                        .try_emplace(loopOp, loopOp, functionAnalysis)
                        .first->getSecond();
    if (!analysis.canPromoteToAffine())
      return false;
  }
  return true;
}

mlir::AffineMap createArrayIndexAffineMap(unsigned dimensions,
                                          MLIRContext *context) {
  auto index = mlir::getAffineConstantExpr(0, context);
  auto accuExtent = mlir::getAffineConstantExpr(1, context);
  for (unsigned i = 0; i < dimensions; ++i) {
    mlir::AffineExpr idx = mlir::getAffineDimExpr(i, context),
                     lowerBound = mlir::getAffineSymbolExpr(i * 3, context),
                     currentExtent =
                         mlir::getAffineSymbolExpr(i * 3 + 1, context),
                     stride = mlir::getAffineSymbolExpr(i * 3 + 2, context),
                     currentPart = (idx * stride - lowerBound) * accuExtent;
    index = currentPart + index;
    accuExtent = accuExtent * currentExtent;
  }
  return mlir::AffineMap::get(dimensions, dimensions * 3, index);
}
Optional<int64_t> constantIntegerLike(const mlir::Value value) {
  if (auto definition = value.getDefiningOp<ConstantOp>())
    if (auto stepAttr = definition.getValue().dyn_cast<IntegerAttr>())
      return stepAttr.getInt();
  return {};
}

/// Convert `fir.loop` to `affine.for`
class AffineLoopConversion : public mlir::OpRewritePattern<fir::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineLoopConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}

  mlir::LogicalResult
  matchAndRewrite(fir::LoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: rewriting loop:\n";
               loop.dump(););
    auto loopAnalysis = functionAnalysis.getChildLoopAnalysis(loop);
    auto &loopOps = loop.getBody()->getOperations();
    auto loopAndIndex = createAffineFor(loop, rewriter);
    auto affineFor = loopAndIndex.first;
    auto inductionVar = loopAndIndex.second;
    rewriter.startRootUpdate(affineFor.getOperation());
    affineFor.getBody()->getOperations().splice(--affineFor.getBody()->end(),
                                                loopOps, loopOps.begin(),
                                                --loopOps.end());
    rewriter.finalizeRootUpdate(affineFor.getOperation());

    rewriter.startRootUpdate(loop.getOperation());
    loop.getInductionVar().replaceAllUsesWith(inductionVar);
    rewriter.finalizeRootUpdate(loop.getOperation());

    for (auto &bodyOp : affineFor.getBody()->getOperations()) {
      if (isa<fir::LoadOp>(bodyOp)) {
        if (failed(rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter))) {
          return failure();
        }
      }
      if (isa<fir::StoreOp>(bodyOp)) {
        if (failed(rewriteStore(cast<fir::StoreOp>(bodyOp), rewriter))) {
          return failure();
        }
      }
    }

    rewriter.replaceOp(loop, affineFor.getOperation()->getResults());

    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: loop rewriten to:\n";
               affineFor.dump(););
    return success();
  }

private:
  std::pair<mlir::AffineForOp, mlir::Value>
  createAffineFor(fir::LoopOp op, mlir::PatternRewriter &rewriter) const {
    if (auto constantStep = constantIntegerLike(op.step()))
      if (constantStep.getValue() > 0)
        return positiveConstantStep(op, constantStep.getValue(), rewriter);
    return genericBounds(op, rewriter);
  }
  std::pair<mlir::AffineForOp, mlir::Value>
  positiveConstantStep(fir::LoopOp op, int64_t step,
                       mlir::PatternRewriter &rewriter) const {
    auto affineFor = rewriter.create<mlir::AffineForOp>(
        op.getLoc(), ValueRange(op.lowerBound()),
        mlir::AffineMap::get(0, 1,
                             mlir::getAffineSymbolExpr(0, op.getContext())),
        ValueRange(op.upperBound()),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
        step);
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }
  std::pair<mlir::AffineForOp, mlir::Value>
  genericBounds(fir::LoopOp op, mlir::PatternRewriter &rewriter) const {
    auto lowerBound = mlir::getAffineSymbolExpr(0, op.getContext());
    auto upperBound = mlir::getAffineSymbolExpr(1, op.getContext());
    auto step = mlir::getAffineSymbolExpr(2, op.getContext());
    mlir::AffineMap upperBoundMap =
        mlir::AffineMap::get(0, 3, (upperBound - lowerBound + step).floorDiv(step));
    auto genericUpperBound = rewriter.create<mlir::AffineApplyOp>(
        op.getLoc(), upperBoundMap,
        ValueRange({op.lowerBound(), op.upperBound(), op.step()}));
    auto actualIndexMap = mlir::AffineMap::get(
        1, 2,
        (lowerBound + mlir::getAffineDimExpr(0, op.getContext())) *
            mlir::getAffineSymbolExpr(1, op.getContext()));

    auto affineFor = rewriter.create<mlir::AffineForOp>(
        op.getLoc(), ValueRange(),
        AffineMap::getConstantMap(0, op.getContext()),
        genericUpperBound.getResult(),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
        1);
    rewriter.setInsertionPointToStart(affineFor.getBody());
    auto actualIndex = rewriter.create<mlir::AffineApplyOp>(
        op.getLoc(), actualIndexMap,
        ValueRange({affineFor.getInductionVar(), op.lowerBound(), op.step()}));
    return std::make_pair(affineFor, actualIndex.getResult());
  }
  mlir::Type coordinateArrayElement(fir::ArrayCoorOp op) const {
    if (auto refType = op.ref().getType().dyn_cast_or_null<ReferenceType>()) {
      if (auto seqType = refType.getEleTy().dyn_cast_or_null<SequenceType>()) {
        return seqType.getEleTy();
      }
    }
    op.emitError(
        "AffineLoopConversion: array type in coordinate operation not valid\n");
    return mlir::Type();
  }
  std::pair<mlir::AffineApplyOp, fir::ConvertOp>
  createAffineOps(mlir::Value arrayRef, mlir::PatternRewriter &rewriter) const {
    auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>();
    auto genDim = acoOp.dims().getDefiningOp<GenDimsOp>();
    auto affineMap =
        createArrayIndexAffineMap(acoOp.coor().size(), acoOp.getContext());
    SmallVector<mlir::Value, 4> indexArgs;
    indexArgs.append(acoOp.coor().begin(), acoOp.coor().end());
    indexArgs.append(genDim.triples().begin(), genDim.triples().end());
    auto affineApply = rewriter.create<mlir::AffineApplyOp>(
        acoOp.getLoc(), affineMap, indexArgs);
    auto arrayElementType = coordinateArrayElement(acoOp);
    auto newType = mlir::MemRefType::get({-1}, arrayElementType);
    auto arrayConvert =
        rewriter.create<fir::ConvertOp>(acoOp.getLoc(), newType, acoOp.ref());
    return std::make_pair(affineApply, arrayConvert);
  }

  mlir::LogicalResult rewriteLoad(fir::LoadOp loadOp,
                                  mlir::PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(loadOp);
    auto affineOps = createAffineOps(loadOp.memref(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::AffineLoadOp>(
        loadOp, affineOps.second.getResult(), affineOps.first.getResult());
    return success();
  }
  mlir::LogicalResult rewriteStore(fir::StoreOp storeOp,
                                   mlir::PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(storeOp);
    auto affineOps = createAffineOps(storeOp.memref(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::AffineStoreOp>(
        storeOp, storeOp.value(), affineOps.second.getResult(),
        affineOps.first.getResult());
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
