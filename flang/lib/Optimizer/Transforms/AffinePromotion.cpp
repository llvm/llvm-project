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
#include "mlir/IR/IntegerSet.h"
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
class AffineIfAnalysis;
class AffineIfConversion;

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
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: \n"; loopOperation.dump(););
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

/// Calculates arguments for creating an IntegerSet symCount, dimCount are the
/// final number of symbols and dimensions of the affine map. If integer set if
/// possible is in Optional IntegerSet
class AffineIfCondition {
public:
  typedef Optional<mlir::AffineExpr> MaybeAffineExpr;
  AffineIfCondition(mlir::Value fc)
      : firCondition(fc), symCount(0), dimCount(0) {
    if (auto condDef = firCondition.getDefiningOp<mlir::CmpIOp>())
      fromCmpIOp(condDef);
  }
  AffineIfCondition() {}
  llvm::SmallVector<mlir::Value, 8> affineArgs;
  friend AffineIfAnalysis;
  friend AffineIfConversion;

private:
  mlir::Value firCondition;
  Optional<mlir::IntegerSet> integerSet;
  unsigned symCount, dimCount;

  MaybeAffineExpr affineBinaryOp(mlir::AffineExprKind kind, mlir::Value lhs,
                                 mlir::Value rhs) {
    return affineBinaryOp(kind, toAffineExpr(lhs), toAffineExpr(rhs));
  }

  MaybeAffineExpr affineBinaryOp(mlir::AffineExprKind kind, MaybeAffineExpr lhs,
                                 MaybeAffineExpr rhs) {
    if (lhs.hasValue() && rhs.hasValue())
      return mlir::getAffineBinaryOpExpr(kind, lhs.getValue(), rhs.getValue());
    else
      return {};
  }
  MaybeAffineExpr toAffineExpr(MaybeAffineExpr e) { return e; }
  MaybeAffineExpr toAffineExpr(int64_t value) {
    return {mlir::getAffineConstantExpr(value, firCondition.getContext())};
  }
  /// Returns an AffineExpr if it is a result of operations that can be done
  /// in an affine expression, this includes -, +, *, rem, constant.
  /// block arguments of a loopOp or forOp are used as dimensions
  MaybeAffineExpr toAffineExpr(mlir::Value value) {
    if (auto op = value.getDefiningOp<mlir::SubIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Add, toAffineExpr(op.lhs()),
                            affineBinaryOp(mlir::AffineExprKind::Mul,
                                           toAffineExpr(op.rhs()),
                                           toAffineExpr(-1)));
    if (auto op = value.getDefiningOp<mlir::AddIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Add, op.lhs(), op.rhs());
    if (auto op = value.getDefiningOp<mlir::MulIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Mul, op.lhs(), op.rhs());
    if (auto op = value.getDefiningOp<mlir::UnsignedRemIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Mod, op.lhs(), op.rhs());
    if (auto op = value.getDefiningOp<mlir::ConstantOp>())
      if (auto intConstant = op.getValue().dyn_cast<IntegerAttr>())
        return toAffineExpr(intConstant.getInt());
    if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
      affineArgs.push_back(value);
      if (isa<fir::LoopOp>(blockArg.getOwner()->getParentOp()) ||
          isa<mlir::AffineForOp>(blockArg.getOwner()->getParentOp()))
        return {mlir::getAffineDimExpr(dimCount++, value.getContext())};
      return {mlir::getAffineSymbolExpr(symCount++, value.getContext())};
    }
    return {};
  }
  void fromCmpIOp(mlir::CmpIOp cmpOp) {
    auto lhsAffine = toAffineExpr(cmpOp.lhs());
    auto rhsAffine = toAffineExpr(cmpOp.rhs());
    if (!lhsAffine.hasValue() || !rhsAffine.hasValue())
      return;
    auto constraintPair = constraint(
        cmpOp.predicate(), rhsAffine.getValue() - lhsAffine.getValue());
    if (!constraintPair)
      return;
    integerSet = mlir::IntegerSet::get(dimCount, symCount,
                                       {constraintPair.getValue().first},
                                       {constraintPair.getValue().second});
    return;
  }

  Optional<std::pair<AffineExpr, bool>>
  constraint(mlir::CmpIPredicate predicate, mlir::AffineExpr basic) {
    switch (predicate) {
    case mlir::CmpIPredicate::slt:
      return {std::make_pair(basic - 1, false)};
    case mlir::CmpIPredicate::sle:
      return {std::make_pair(basic, false)};
    case mlir::CmpIPredicate::sgt:
      return {std::make_pair(1 - basic, false)};
    case mlir::CmpIPredicate::sge:
      return {std::make_pair(0 - basic, false)};
    case mlir::CmpIPredicate::eq:
      return {std::make_pair(basic, true)};
    default:
      return {};
    }
  }
};

/// Analysis for affine promotion of fir.if
class AffineIfAnalysis {
public:
  AffineIfAnalysis(fir::WhereOp op, AffineFunctionAnalysis &afa)
      : legality(analyzeIf(op, afa)) {}
  bool canPromoteToAffine() { return legality; }
  friend AffineFunctionAnalysis;

private:
  bool legality;
  AffineIfAnalysis(bool forcedLegality) : legality(forcedLegality) {}
  bool analyzeIf(fir::WhereOp, AffineFunctionAnalysis &);
};

/// Stores analysis objects for all loops and where operations inside a function
///  these analysis are used twice, first for marking operations for rewrite and
///  second when doing rewrite.
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
  AffineIfAnalysis getChildIfAnalysis(fir::WhereOp op) const {
    auto it = ifAnalysisMap.find_as(op);
    if (it == ifAnalysisMap.end()) {
      LLVM_DEBUG(llvm::dbgs() << "AffineFunctionAnalysis: not computed for:\n";
                 op.dump(););
      op.emitError("error in fetching if analysis in AffineFunctionAnalysis\n");
      return AffineIfAnalysis(false);
    }
    return it->getSecond();
  }
  friend AffineLoopAnalysis;
  friend AffineIfAnalysis;

private:
  llvm::DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
  llvm::DenseMap<mlir::Operation *, AffineIfAnalysis> ifAnalysisMap;
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
  for (auto whereOp : loopOperation.getOps<fir::WhereOp>())
    functionAnalysis.ifAnalysisMap.try_emplace(whereOp, whereOp,
                                               functionAnalysis);
  return true;
}

bool AffineIfAnalysis::analyzeIf(fir::WhereOp op, AffineFunctionAnalysis &afa) {
  if (op.getNumResults() == 0)
    return true;
  LLVM_DEBUG(
      llvm::dbgs() << "AffineIfAnalysis: not promoting as op has results\n";);
  return false;
}

/// AffineMap rewriting fir.array_coor operation to affine apply,
/// %dim = fir.gendim %lowerBound, %upperBound, %stride
/// %a = fir.array_coor %arr(%dim) %i
/// returning affineMap = affine_map<(i)[lb, ub, st] -> (i*st - lb)>
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

mlir::Type coordinateArrayElement(fir::ArrayCoorOp op) {
  if (auto refType = op.ref().getType().dyn_cast_or_null<ReferenceType>()) {
    if (auto seqType = refType.getEleTy().dyn_cast_or_null<SequenceType>()) {
      return seqType.getEleTy();
    }
  }
  op.emitError(
      "AffineLoopConversion: array type in coordinate operation not valid\n");
  return mlir::Type();
}

/// Returns affine.apply and fir.convert from array_coor and gendims
std::pair<mlir::AffineApplyOp, fir::ConvertOp>
createAffineOps(mlir::Value arrayRef, mlir::PatternRewriter &rewriter) {
  auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>();
  auto genDim = acoOp.dims().getDefiningOp<GenDimsOp>();
  auto affineMap =
      createArrayIndexAffineMap(acoOp.coor().size(), acoOp.getContext());
  SmallVector<mlir::Value, 4> indexArgs;
  indexArgs.append(acoOp.coor().begin(), acoOp.coor().end());
  indexArgs.append(genDim.triples().begin(), genDim.triples().end());
  auto affineApply = rewriter.create<mlir::AffineApplyOp>(acoOp.getLoc(),
                                                          affineMap, indexArgs);
  auto arrayElementType = coordinateArrayElement(acoOp);
  auto newType = mlir::MemRefType::get({-1}, arrayElementType);
  auto arrayConvert =
      rewriter.create<fir::ConvertOp>(acoOp.getLoc(), newType, acoOp.ref());
  return std::make_pair(affineApply, arrayConvert);
}

void rewriteLoad(fir::LoadOp loadOp, mlir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(loadOp);
  auto affineOps = createAffineOps(loadOp.memref(), rewriter);
  rewriter.replaceOpWithNewOp<mlir::AffineLoadOp>(
      loadOp, affineOps.second.getResult(), affineOps.first.getResult());
}

void rewriteStore(fir::StoreOp storeOp, mlir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(storeOp);
  auto affineOps = createAffineOps(storeOp.memref(), rewriter);
  rewriter.replaceOpWithNewOp<mlir::AffineStoreOp>(storeOp, storeOp.value(),
                                                   affineOps.second.getResult(),
                                                   affineOps.first.getResult());
}

void rewriteMemoryOps(Block *block, mlir::PatternRewriter &rewriter) {
  for (auto &bodyOp : block->getOperations()) {
    if (isa<fir::LoadOp>(bodyOp))
      rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter);
    if (isa<fir::StoreOp>(bodyOp))
      rewriteStore(cast<fir::StoreOp>(bodyOp), rewriter);
  }
}

/// Convert `fir.loop` to `affine.for`, creates fir.convert for arrays to
/// memref, rewrites array_coor to affine.apply with affine_map. Rewrites fir
/// loads and stores to affine.
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

    rewriteMemoryOps(affineFor.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: loop rewriten to:\n";
               affineFor.dump(););
    rewriter.replaceOp(loop, affineFor.getOperation()->getResults());
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
  // when step for the loop is positive compile time constant
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
    mlir::AffineMap upperBoundMap = mlir::AffineMap::get(
        0, 3, (upperBound - lowerBound + step).floorDiv(step));
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
  AffineFunctionAnalysis &functionAnalysis;
};

class AffineIfConversion : public mlir::OpRewritePattern<fir::WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineIfConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}
  mlir::LogicalResult
  matchAndRewrite(fir::WhereOp op,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: rewriting where:\n";
               op.dump(););
    auto &whereOps = op.whereRegion().front().getOperations();
    auto affineCondition = AffineIfCondition(op.condition());
    if (!affineCondition.integerSet) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineIfConversion: couldn't calculate affine condition\n";);
      return failure();
    }
    auto affineIf = rewriter.create<mlir::AffineIfOp>(
        op.getLoc(), affineCondition.integerSet.getValue(),
        affineCondition.affineArgs, !op.otherRegion().empty());
    rewriter.startRootUpdate(affineIf);
    affineIf.getThenBlock()->getOperations().splice(
        --affineIf.getThenBlock()->end(), whereOps, whereOps.begin(),
        --whereOps.end());
    if (!op.otherRegion().empty()) {
      auto &otherOps = op.otherRegion().front().getOperations();
      affineIf.getElseBlock()->getOperations().splice(
          --affineIf.getElseBlock()->end(), otherOps, otherOps.begin(),
          --otherOps.end());
    }
    rewriter.finalizeRootUpdate(affineIf);
    rewriteMemoryOps(affineIf.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: where converted to:\n";
               affineIf.dump(););
    rewriter.replaceOp(op, affineIf.getOperation()->getResults());
    return success();
  }

private:
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
    patterns.insert<AffineIfConversion>(context, functionAnalysis);
    patterns.insert<AffineLoopConversion>(context, functionAnalysis);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::scf::SCFDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<WhereOp>([&functionAnalysis](fir::WhereOp op) {
      return !(functionAnalysis.getChildIfAnalysis(op).canPromoteToAffine());
    });
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
