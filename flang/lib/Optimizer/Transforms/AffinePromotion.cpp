//===-- AffinePromotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation is a prototype that promote FIR loops operations
// to affine dialect operations.
// It is not part of the production pipeline and would need more work in order
// to be used in production.
// More information can be found in this presentation:
// https://slides.com/rajanwalia/deck
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/IntegerSet.h"
#include "aiir/IR/Visitors.h"
#include "aiir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_AFFINEDIALECTPROMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-affine-promotion"

using namespace fir;
using namespace aiir;

namespace {
struct AffineLoopAnalysis;
struct AffineIfAnalysis;

/// Stores analysis objects for all loops and if operations inside a function
/// these analysis are used twice, first for marking operations for rewrite and
/// second when doing rewrite.
struct AffineFunctionAnalysis {
  explicit AffineFunctionAnalysis(aiir::func::FuncOp funcOp) {
    funcOp->walk([&](fir::DoLoopOp doloop) {
      loopAnalysisMap.try_emplace(doloop, doloop, *this);
    });
  }

  AffineLoopAnalysis getChildLoopAnalysis(fir::DoLoopOp op) const;

  AffineIfAnalysis getChildIfAnalysis(fir::IfOp op) const;

  llvm::DenseMap<aiir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
  llvm::DenseMap<aiir::Operation *, AffineIfAnalysis> ifAnalysisMap;
};
} // namespace

static bool analyzeCoordinate(aiir::Value coordinate, aiir::Operation *op) {
  if (auto blockArg = aiir::dyn_cast<aiir::BlockArgument>(coordinate)) {
    if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()))
      return true;
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a "
                               "loop induction variable (owner not loopOp)\n";
               op->dump());
    return false;
  }
  LLVM_DEBUG(
      llvm::dbgs() << "AffineLoopAnalysis: array coordinate is not a loop "
                      "induction variable (not a block argument)\n";
      op->dump(); coordinate.getDefiningOp()->dump());
  return false;
}

namespace {
struct AffineLoopAnalysis {
  AffineLoopAnalysis() = default;

  explicit AffineLoopAnalysis(fir::DoLoopOp op, AffineFunctionAnalysis &afa)
      : legality(analyzeLoop(op, afa)) {}

  bool canPromoteToAffine() { return legality; }

private:
  bool analyzeBody(fir::DoLoopOp loopOperation,
                   AffineFunctionAnalysis &functionAnalysis) {
    for (auto loopOp : loopOperation.getOps<fir::DoLoopOp>()) {
      auto analysis = functionAnalysis.loopAnalysisMap
                          .try_emplace(loopOp, loopOp, functionAnalysis)
                          .first->getSecond();
      if (!analysis.canPromoteToAffine())
        return false;
    }
    for (auto ifOp : loopOperation.getOps<fir::IfOp>())
      functionAnalysis.ifAnalysisMap.try_emplace(ifOp, ifOp, functionAnalysis);
    return true;
  }

  bool analysisResults(fir::DoLoopOp loopOperation) {
    if (loopOperation.getFinalValue() &&
        !loopOperation.getResult(0).use_empty()) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineLoopAnalysis: cannot promote loop final value\n";);
      return false;
    }

    return true;
  }

  bool analyzeLoop(fir::DoLoopOp loopOperation,
                   AffineFunctionAnalysis &functionAnalysis) {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: \n"; loopOperation.dump(););
    return analyzeMemoryAccess(loopOperation) &&
           analysisResults(loopOperation) &&
           analyzeBody(loopOperation, functionAnalysis);
  }

  bool analyzeReference(aiir::Value memref, aiir::Operation *op) {
    if (auto acoOp = memref.getDefiningOp<ArrayCoorOp>()) {
      if (aiir::isa<fir::BoxType>(acoOp.getMemref().getType())) {
        // TODO: Look if and how fir.box can be promoted to affine.
        LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                   "array memory operation uses fir.box\n";
                   op->dump(); acoOp.dump(););
        return false;
      }
      bool canPromote = true;
      for (auto coordinate : acoOp.getIndices())
        canPromote = canPromote && analyzeCoordinate(coordinate, op);
      return canPromote;
    }
    if (auto coOp = memref.getDefiningOp<CoordinateOp>()) {
      LLVM_DEBUG(llvm::dbgs()
                     << "AffineLoopAnalysis: cannot promote loop, "
                        "array memory operation uses non ArrayCoorOp\n";
                 op->dump(); coOp.dump(););

      return false;
    }
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: unknown type of memory "
                               "reference for array load\n";
               op->dump(););
    return false;
  }

  bool analyzeMemoryAccess(fir::DoLoopOp loopOperation) {
    for (auto loadOp : loopOperation.getOps<fir::LoadOp>())
      if (!analyzeReference(loadOp.getMemref(), loadOp))
        return false;
    for (auto storeOp : loopOperation.getOps<fir::StoreOp>())
      if (!analyzeReference(storeOp.getMemref(), storeOp))
        return false;
    return true;
  }

  bool legality{};
};
} // namespace

AffineLoopAnalysis
AffineFunctionAnalysis::getChildLoopAnalysis(fir::DoLoopOp op) const {
  auto it = loopAnalysisMap.find_as(op);
  if (it == loopAnalysisMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "AffineFunctionAnalysis: not computed for:\n";
               op.dump(););
    op.emitError("error in fetching loop analysis in AffineFunctionAnalysis\n");
    return {};
  }
  return it->getSecond();
}

namespace {
/// Calculates arguments for creating an IntegerSet. symCount, dimCount are the
/// final number of symbols and dimensions of the affine map. Integer set if
/// possible is in Optional IntegerSet.
struct AffineIfCondition {
  using MaybeAffineExpr = std::optional<aiir::AffineExpr>;

  explicit AffineIfCondition(aiir::Value fc) : firCondition(fc) {
    if (auto condDef = firCondition.getDefiningOp<aiir::arith::CmpIOp>())
      fromCmpIOp(condDef);
  }

  bool hasIntegerSet() const { return integerSet.has_value(); }

  aiir::IntegerSet getIntegerSet() const {
    assert(hasIntegerSet() && "integer set is missing");
    return *integerSet;
  }

  aiir::ValueRange getAffineArgs() const { return affineArgs; }

private:
  MaybeAffineExpr affineBinaryOp(aiir::AffineExprKind kind, aiir::Value lhs,
                                 aiir::Value rhs) {
    return affineBinaryOp(kind, toAffineExpr(lhs), toAffineExpr(rhs));
  }

  MaybeAffineExpr affineBinaryOp(aiir::AffineExprKind kind, MaybeAffineExpr lhs,
                                 MaybeAffineExpr rhs) {
    if (lhs && rhs)
      return aiir::getAffineBinaryOpExpr(kind, *lhs, *rhs);
    return {};
  }

  MaybeAffineExpr toAffineExpr(MaybeAffineExpr e) { return e; }

  MaybeAffineExpr toAffineExpr(int64_t value) {
    return {aiir::getAffineConstantExpr(value, firCondition.getContext())};
  }

  /// Returns an AffineExpr if it is a result of operations that can be done
  /// in an affine expression, this includes -, +, *, rem, constant.
  /// block arguments of a loopOp or forOp are used as dimensions
  MaybeAffineExpr toAffineExpr(aiir::Value value) {
    if (auto op = value.getDefiningOp<aiir::arith::SubIOp>())
      return affineBinaryOp(
          aiir::AffineExprKind::Add, toAffineExpr(op.getLhs()),
          affineBinaryOp(aiir::AffineExprKind::Mul, toAffineExpr(op.getRhs()),
                         toAffineExpr(-1)));
    if (auto op = value.getDefiningOp<aiir::arith::AddIOp>())
      return affineBinaryOp(aiir::AffineExprKind::Add, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<aiir::arith::MulIOp>())
      return affineBinaryOp(aiir::AffineExprKind::Mul, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<aiir::arith::RemUIOp>())
      return affineBinaryOp(aiir::AffineExprKind::Mod, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<aiir::arith::ConstantOp>())
      if (auto intConstant = aiir::dyn_cast<IntegerAttr>(op.getValue()))
        return toAffineExpr(intConstant.getInt());
    if (auto blockArg = aiir::dyn_cast<aiir::BlockArgument>(value)) {
      affineArgs.push_back(value);
      if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()) ||
          isa<aiir::affine::AffineForOp>(blockArg.getOwner()->getParentOp()))
        return {aiir::getAffineDimExpr(dimCount++, value.getContext())};
      return {aiir::getAffineSymbolExpr(symCount++, value.getContext())};
    }
    return {};
  }

  void fromCmpIOp(aiir::arith::CmpIOp cmpOp) {
    auto lhsAffine = toAffineExpr(cmpOp.getLhs());
    auto rhsAffine = toAffineExpr(cmpOp.getRhs());
    if (!lhsAffine || !rhsAffine)
      return;
    auto constraintPair =
        constraint(cmpOp.getPredicate(), *rhsAffine - *lhsAffine);
    if (!constraintPair)
      return;
    integerSet = aiir::IntegerSet::get(
        dimCount, symCount, {constraintPair->first}, {constraintPair->second});
  }

  std::optional<std::pair<AffineExpr, bool>>
  constraint(aiir::arith::CmpIPredicate predicate, aiir::AffineExpr basic) {
    switch (predicate) {
    case aiir::arith::CmpIPredicate::slt:
      return {std::make_pair(basic - 1, false)};
    case aiir::arith::CmpIPredicate::sle:
      return {std::make_pair(basic, false)};
    case aiir::arith::CmpIPredicate::sgt:
      return {std::make_pair(1 - basic, false)};
    case aiir::arith::CmpIPredicate::sge:
      return {std::make_pair(0 - basic, false)};
    case aiir::arith::CmpIPredicate::eq:
      return {std::make_pair(basic, true)};
    default:
      return {};
    }
  }

  llvm::SmallVector<aiir::Value> affineArgs;
  std::optional<aiir::IntegerSet> integerSet;
  aiir::Value firCondition;
  unsigned symCount{0u};
  unsigned dimCount{0u};
};
} // namespace

namespace {
/// Analysis for affine promotion of fir.if
struct AffineIfAnalysis {
  AffineIfAnalysis() = default;

  explicit AffineIfAnalysis(fir::IfOp op, AffineFunctionAnalysis &afa)
      : legality(analyzeIf(op, afa)) {}

  bool canPromoteToAffine() { return legality; }

private:
  bool analyzeIf(fir::IfOp op, AffineFunctionAnalysis &afa) {
    if (op.getNumResults() == 0)
      return true;
    LLVM_DEBUG(llvm::dbgs()
                   << "AffineIfAnalysis: not promoting as op has results\n";);
    return false;
  }

  bool legality{};
};
} // namespace

AffineIfAnalysis
AffineFunctionAnalysis::getChildIfAnalysis(fir::IfOp op) const {
  auto it = ifAnalysisMap.find_as(op);
  if (it == ifAnalysisMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "AffineFunctionAnalysis: not computed for:\n";
               op.dump(););
    op.emitError("error in fetching if analysis in AffineFunctionAnalysis\n");
    return {};
  }
  return it->getSecond();
}

/// AffineMap rewriting fir.array_coor operation to affine apply,
/// %dim = fir.gendim %lowerBound, %upperBound, %stride
/// %a = fir.array_coor %arr(%dim) %i
/// returning affineMap = affine_map<(i)[lb, ub, st] -> (i*st - lb)>
static aiir::AffineMap createArrayIndexAffineMap(unsigned dimensions,
                                                 AIIRContext *context) {
  auto index = aiir::getAffineConstantExpr(0, context);
  auto accuExtent = aiir::getAffineConstantExpr(1, context);
  for (unsigned i = 0; i < dimensions; ++i) {
    aiir::AffineExpr idx = aiir::getAffineDimExpr(i, context),
                     lowerBound = aiir::getAffineSymbolExpr(i * 3, context),
                     currentExtent =
                         aiir::getAffineSymbolExpr(i * 3 + 1, context),
                     stride = aiir::getAffineSymbolExpr(i * 3 + 2, context),
                     currentPart = (idx * stride - lowerBound) * accuExtent;
    index = currentPart + index;
    accuExtent = accuExtent * currentExtent;
  }
  return aiir::AffineMap::get(dimensions, dimensions * 3, index);
}

static std::optional<int64_t> constantIntegerLike(const aiir::Value value) {
  if (auto definition = value.getDefiningOp<aiir::arith::ConstantOp>())
    if (auto stepAttr = aiir::dyn_cast<IntegerAttr>(definition.getValue()))
      return stepAttr.getInt();
  return {};
}

static aiir::Type coordinateArrayElement(fir::ArrayCoorOp op) {
  if (auto refType =
          aiir::dyn_cast_or_null<ReferenceType>(op.getMemref().getType())) {
    if (auto seqType =
            aiir::dyn_cast_or_null<SequenceType>(refType.getEleTy())) {
      return seqType.getEleTy();
    }
  }
  op.emitError(
      "AffineLoopConversion: array type in coordinate operation not valid\n");
  return aiir::Type();
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::ShapeOp shape,
                              SmallVectorImpl<aiir::Value> &indexArgs,
                              aiir::PatternRewriter &rewriter) {
  auto one = aiir::arith::ConstantOp::create(rewriter, acoOp.getLoc(),
                                             rewriter.getIndexType(),
                                             rewriter.getIndexAttr(1));
  auto extents = shape.getExtents();
  for (auto i = extents.begin(); i < extents.end(); i++) {
    indexArgs.push_back(one);
    indexArgs.push_back(*i);
    indexArgs.push_back(one);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::ShapeShiftOp shape,
                              SmallVectorImpl<aiir::Value> &indexArgs,
                              aiir::PatternRewriter &rewriter) {
  auto one = aiir::arith::ConstantOp::create(rewriter, acoOp.getLoc(),
                                             rewriter.getIndexType(),
                                             rewriter.getIndexAttr(1));
  auto extents = shape.getPairs();
  for (auto i = extents.begin(); i < extents.end();) {
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
    indexArgs.push_back(one);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::SliceOp slice,
                              SmallVectorImpl<aiir::Value> &indexArgs,
                              aiir::PatternRewriter &rewriter) {
  auto extents = slice.getTriples();
  for (auto i = extents.begin(); i < extents.end();) {
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp,
                              SmallVectorImpl<aiir::Value> &indexArgs,
                              aiir::PatternRewriter &rewriter) {
  if (auto shape = acoOp.getShape().getDefiningOp<ShapeOp>())
    return populateIndexArgs(acoOp, shape, indexArgs, rewriter);
  if (auto shapeShift = acoOp.getShape().getDefiningOp<ShapeShiftOp>())
    return populateIndexArgs(acoOp, shapeShift, indexArgs, rewriter);
  if (auto slice = acoOp.getShape().getDefiningOp<SliceOp>())
    return populateIndexArgs(acoOp, slice, indexArgs, rewriter);
}

/// Returns affine.apply and fir.convert from array_coor and gendims
static std::pair<affine::AffineApplyOp, fir::ConvertOp>
createAffineOps(aiir::Value arrayRef, aiir::PatternRewriter &rewriter) {
  auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>();
  auto affineMap =
      createArrayIndexAffineMap(acoOp.getIndices().size(), acoOp.getContext());
  SmallVector<aiir::Value> indexArgs;
  indexArgs.append(acoOp.getIndices().begin(), acoOp.getIndices().end());

  populateIndexArgs(acoOp, indexArgs, rewriter);

  auto affineApply = affine::AffineApplyOp::create(rewriter, acoOp.getLoc(),
                                                   affineMap, indexArgs);
  auto arrayElementType = coordinateArrayElement(acoOp);
  auto newType =
      aiir::MemRefType::get({aiir::ShapedType::kDynamic}, arrayElementType);
  auto arrayConvert = fir::ConvertOp::create(rewriter, acoOp.getLoc(), newType,
                                             acoOp.getMemref());
  return std::make_pair(affineApply, arrayConvert);
}

static void rewriteLoad(fir::LoadOp loadOp, aiir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(loadOp);
  auto affineOps = createAffineOps(loadOp.getMemref(), rewriter);
  rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
      loadOp, affineOps.second.getResult(), affineOps.first.getResult());
}

static void rewriteStore(fir::StoreOp storeOp,
                         aiir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(storeOp);
  auto affineOps = createAffineOps(storeOp.getMemref(), rewriter);
  rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
      storeOp, storeOp.getValue(), affineOps.second.getResult(),
      affineOps.first.getResult());
}

static void rewriteMemoryOps(Block *block, aiir::PatternRewriter &rewriter) {
  for (auto &bodyOp : llvm::make_early_inc_range(block->getOperations())) {
    if (isa<fir::LoadOp>(bodyOp))
      rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter);
    else if (isa<fir::StoreOp>(bodyOp))
      rewriteStore(cast<fir::StoreOp>(bodyOp), rewriter);
  }
}

namespace {
/// Convert `fir.do_loop` to `affine.for`, creates fir.convert for arrays to
/// memref, rewrites array_coor to affine.apply with affine_map. Rewrites fir
/// loads and stores to affine.
class AffineLoopConversion : public aiir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineLoopConversion(aiir::AIIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}

  llvm::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  aiir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: rewriting loop:\n";
               loop.dump(););
    [[maybe_unused]] auto loopAnalysis =
        functionAnalysis.getChildLoopAnalysis(loop);
    if (!loopAnalysis.canPromoteToAffine())
      return rewriter.notifyMatchFailure(loop, "cannot promote to affine");
    auto &loopOps = loop.getBody()->getOperations();
    auto resultOp = cast<fir::ResultOp>(loop.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    auto loopResults = loop->getResults();
    auto loopAndIndex = createAffineFor(loop, rewriter);
    auto affineFor = loopAndIndex.first;
    auto inductionVar = loopAndIndex.second;

    if (loop.getFinalValue()) {
      results = results.drop_front();
      loopResults = loopResults.drop_front();
    }

    rewriter.startOpModification(affineFor.getOperation());
    affineFor.getBody()->getOperations().splice(
        std::prev(affineFor.getBody()->end()), loopOps, loopOps.begin(),
        std::prev(loopOps.end()));
    rewriter.replaceAllUsesWith(loop.getRegionIterArgs(),
                                affineFor.getRegionIterArgs());
    if (!results.empty()) {
      rewriter.setInsertionPointToEnd(affineFor.getBody());
      affine::AffineYieldOp::create(rewriter, resultOp->getLoc(), results);
    }
    rewriter.finalizeOpModification(affineFor.getOperation());

    rewriter.startOpModification(loop.getOperation());
    loop.getInductionVar().replaceAllUsesWith(inductionVar);
    rewriter.finalizeOpModification(loop.getOperation());

    rewriteMemoryOps(affineFor.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: loop rewriten to:\n";
               affineFor.dump(););
    rewriter.replaceAllUsesWith(loopResults, affineFor->getResults());
    rewriter.eraseOp(loop);
    return success();
  }

private:
  std::pair<affine::AffineForOp, aiir::Value>
  createAffineFor(fir::DoLoopOp op, aiir::PatternRewriter &rewriter) const {
    if (auto constantStep = constantIntegerLike(op.getStep()))
      if (*constantStep > 0)
        return positiveConstantStep(op, *constantStep, rewriter);
    return genericBounds(op, rewriter);
  }

  // when step for the loop is positive compile time constant
  std::pair<affine::AffineForOp, aiir::Value>
  positiveConstantStep(fir::DoLoopOp op, int64_t step,
                       aiir::PatternRewriter &rewriter) const {
    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), ValueRange(op.getLowerBound()),
        aiir::AffineMap::get(0, 1,
                             aiir::getAffineSymbolExpr(0, op.getContext())),
        ValueRange(op.getUpperBound()),
        aiir::AffineMap::get(0, 1,
                             1 + aiir::getAffineSymbolExpr(0, op.getContext())),
        step, op.getIterOperands());
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }

  std::pair<affine::AffineForOp, aiir::Value>
  genericBounds(fir::DoLoopOp op, aiir::PatternRewriter &rewriter) const {
    auto lowerBound = aiir::getAffineSymbolExpr(0, op.getContext());
    auto upperBound = aiir::getAffineSymbolExpr(1, op.getContext());
    auto step = aiir::getAffineSymbolExpr(2, op.getContext());
    aiir::AffineMap upperBoundMap = aiir::AffineMap::get(
        0, 3, (upperBound - lowerBound + step).floorDiv(step));
    auto genericUpperBound = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), upperBoundMap,
        ValueRange({op.getLowerBound(), op.getUpperBound(), op.getStep()}));
    auto actualIndexMap = aiir::AffineMap::get(
        1, 2,
        (lowerBound + aiir::getAffineDimExpr(0, op.getContext())) *
            aiir::getAffineSymbolExpr(1, op.getContext()));

    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), ValueRange(),
        AffineMap::getConstantMap(0, op.getContext()),
        genericUpperBound.getResult(),
        aiir::AffineMap::get(0, 1,
                             1 + aiir::getAffineSymbolExpr(0, op.getContext())),
        1, op.getIterOperands());
    rewriter.setInsertionPointToStart(affineFor.getBody());
    auto actualIndex = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), actualIndexMap,
        ValueRange(
            {affineFor.getInductionVar(), op.getLowerBound(), op.getStep()}));
    return std::make_pair(affineFor, actualIndex.getResult());
  }

  AffineFunctionAnalysis &functionAnalysis;
};

/// Convert `fir.if` to `affine.if`.
class AffineIfConversion : public aiir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineIfConversion(aiir::AIIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}
  llvm::LogicalResult
  matchAndRewrite(fir::IfOp op,
                  aiir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: rewriting if:\n";
               op.dump(););
    if (!functionAnalysis.getChildIfAnalysis(op).canPromoteToAffine())
      return rewriter.notifyMatchFailure(op, "cannot promote to affine");
    auto &ifOps = op.getThenRegion().front().getOperations();
    auto affineCondition = AffineIfCondition(op.getCondition());
    if (!affineCondition.hasIntegerSet()) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineIfConversion: couldn't calculate affine condition\n";);
      return failure();
    }
    auto affineIf = affine::AffineIfOp::create(
        rewriter, op.getLoc(), affineCondition.getIntegerSet(),
        affineCondition.getAffineArgs(), !op.getElseRegion().empty());
    rewriter.startOpModification(affineIf);
    affineIf.getThenBlock()->getOperations().splice(
        std::prev(affineIf.getThenBlock()->end()), ifOps, ifOps.begin(),
        std::prev(ifOps.end()));
    if (!op.getElseRegion().empty()) {
      auto &otherOps = op.getElseRegion().front().getOperations();
      affineIf.getElseBlock()->getOperations().splice(
          std::prev(affineIf.getElseBlock()->end()), otherOps, otherOps.begin(),
          std::prev(otherOps.end()));
    }
    rewriter.finalizeOpModification(affineIf);
    rewriteMemoryOps(affineIf.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: if converted to:\n";
               affineIf.dump(););
    rewriter.replaceOp(op, affineIf.getOperation()->getResults());
    return success();
  }

  AffineFunctionAnalysis &functionAnalysis;
};

/// Promote fir.do_loop and fir.if to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public fir::impl::AffineDialectPromotionBase<AffineDialectPromotion> {
public:
  void runOnOperation() override {

    auto *context = &getContext();
    auto function = getOperation();
    markAllAnalysesPreserved();
    auto functionAnalysis = AffineFunctionAnalysis(function);
    aiir::RewritePatternSet patterns(context);
    patterns.insert<AffineIfConversion>(context, functionAnalysis);
    patterns.insert<AffineLoopConversion>(context, functionAnalysis);
    LLVM_DEBUG(llvm::dbgs()
                   << "AffineDialectPromotion: running promotion on: \n";
               function.print(llvm::dbgs()););
    // apply the patterns
    walkAndApplyPatterns(function, std::move(patterns));
  }
};
} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<aiir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}
