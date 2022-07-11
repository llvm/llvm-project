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

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-affine-promotion"

using namespace fir;
using namespace mlir;

namespace {
struct AffineLoopAnalysis;
struct AffineIfAnalysis;

/// Stores analysis objects for all loops and if operations inside a function
/// these analysis are used twice, first for marking operations for rewrite and
/// second when doing rewrite.
struct AffineFunctionAnalysis {
  explicit AffineFunctionAnalysis(mlir::func::FuncOp funcOp) {
    for (fir::DoLoopOp op : funcOp.getOps<fir::DoLoopOp>())
      loopAnalysisMap.try_emplace(op, op, *this);
  }

  AffineLoopAnalysis getChildLoopAnalysis(fir::DoLoopOp op) const;

  AffineIfAnalysis getChildIfAnalysis(fir::IfOp op) const;

  llvm::DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
  llvm::DenseMap<mlir::Operation *, AffineIfAnalysis> ifAnalysisMap;
};
} // namespace

static bool analyzeCoordinate(mlir::Value coordinate, mlir::Operation *op) {
  if (auto blockArg = coordinate.dyn_cast<mlir::BlockArgument>()) {
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

  bool analyzeLoop(fir::DoLoopOp loopOperation,
                   AffineFunctionAnalysis &functionAnalysis) {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: \n"; loopOperation.dump(););
    return analyzeMemoryAccess(loopOperation) &&
           analyzeBody(loopOperation, functionAnalysis);
  }

  bool analyzeReference(mlir::Value memref, mlir::Operation *op) {
    if (auto acoOp = memref.getDefiningOp<ArrayCoorOp>()) {
      if (acoOp.getMemref().getType().isa<fir::BoxType>()) {
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
  using MaybeAffineExpr = llvm::Optional<mlir::AffineExpr>;

  explicit AffineIfCondition(mlir::Value fc) : firCondition(fc) {
    if (auto condDef = firCondition.getDefiningOp<mlir::arith::CmpIOp>())
      fromCmpIOp(condDef);
  }

  bool hasIntegerSet() const { return integerSet.hasValue(); }

  mlir::IntegerSet getIntegerSet() const {
    assert(hasIntegerSet() && "integer set is missing");
    return integerSet.getValue();
  }

  mlir::ValueRange getAffineArgs() const { return affineArgs; }

private:
  MaybeAffineExpr affineBinaryOp(mlir::AffineExprKind kind, mlir::Value lhs,
                                 mlir::Value rhs) {
    return affineBinaryOp(kind, toAffineExpr(lhs), toAffineExpr(rhs));
  }

  MaybeAffineExpr affineBinaryOp(mlir::AffineExprKind kind, MaybeAffineExpr lhs,
                                 MaybeAffineExpr rhs) {
    if (lhs && rhs)
      return mlir::getAffineBinaryOpExpr(kind, lhs.getValue(), rhs.getValue());
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
    if (auto op = value.getDefiningOp<mlir::arith::SubIOp>())
      return affineBinaryOp(
          mlir::AffineExprKind::Add, toAffineExpr(op.getLhs()),
          affineBinaryOp(mlir::AffineExprKind::Mul, toAffineExpr(op.getRhs()),
                         toAffineExpr(-1)));
    if (auto op = value.getDefiningOp<mlir::arith::AddIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Add, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<mlir::arith::MulIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Mul, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<mlir::arith::RemUIOp>())
      return affineBinaryOp(mlir::AffineExprKind::Mod, op.getLhs(),
                            op.getRhs());
    if (auto op = value.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto intConstant = op.getValue().dyn_cast<IntegerAttr>())
        return toAffineExpr(intConstant.getInt());
    if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
      affineArgs.push_back(value);
      if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()) ||
          isa<mlir::AffineForOp>(blockArg.getOwner()->getParentOp()))
        return {mlir::getAffineDimExpr(dimCount++, value.getContext())};
      return {mlir::getAffineSymbolExpr(symCount++, value.getContext())};
    }
    return {};
  }

  void fromCmpIOp(mlir::arith::CmpIOp cmpOp) {
    auto lhsAffine = toAffineExpr(cmpOp.getLhs());
    auto rhsAffine = toAffineExpr(cmpOp.getRhs());
    if (!lhsAffine || !rhsAffine)
      return;
    auto constraintPair = constraint(
        cmpOp.getPredicate(), rhsAffine.getValue() - lhsAffine.getValue());
    if (!constraintPair)
      return;
    integerSet = mlir::IntegerSet::get(dimCount, symCount,
                                       {constraintPair.getValue().first},
                                       {constraintPair.getValue().second});
    return;
  }

  llvm::Optional<std::pair<AffineExpr, bool>>
  constraint(mlir::arith::CmpIPredicate predicate, mlir::AffineExpr basic) {
    switch (predicate) {
    case mlir::arith::CmpIPredicate::slt:
      return {std::make_pair(basic - 1, false)};
    case mlir::arith::CmpIPredicate::sle:
      return {std::make_pair(basic, false)};
    case mlir::arith::CmpIPredicate::sgt:
      return {std::make_pair(1 - basic, false)};
    case mlir::arith::CmpIPredicate::sge:
      return {std::make_pair(0 - basic, false)};
    case mlir::arith::CmpIPredicate::eq:
      return {std::make_pair(basic, true)};
    default:
      return {};
    }
  }

  llvm::SmallVector<mlir::Value> affineArgs;
  llvm::Optional<mlir::IntegerSet> integerSet;
  mlir::Value firCondition;
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
static mlir::AffineMap createArrayIndexAffineMap(unsigned dimensions,
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

static Optional<int64_t> constantIntegerLike(const mlir::Value value) {
  if (auto definition = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto stepAttr = definition.getValue().dyn_cast<IntegerAttr>())
      return stepAttr.getInt();
  return {};
}

static mlir::Type coordinateArrayElement(fir::ArrayCoorOp op) {
  if (auto refType =
          op.getMemref().getType().dyn_cast_or_null<ReferenceType>()) {
    if (auto seqType = refType.getEleTy().dyn_cast_or_null<SequenceType>()) {
      return seqType.getEleTy();
    }
  }
  op.emitError(
      "AffineLoopConversion: array type in coordinate operation not valid\n");
  return mlir::Type();
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::ShapeOp shape,
                              SmallVectorImpl<mlir::Value> &indexArgs,
                              mlir::PatternRewriter &rewriter) {
  auto one = rewriter.create<mlir::arith::ConstantOp>(
      acoOp.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
  auto extents = shape.getExtents();
  for (auto i = extents.begin(); i < extents.end(); i++) {
    indexArgs.push_back(one);
    indexArgs.push_back(*i);
    indexArgs.push_back(one);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::ShapeShiftOp shape,
                              SmallVectorImpl<mlir::Value> &indexArgs,
                              mlir::PatternRewriter &rewriter) {
  auto one = rewriter.create<mlir::arith::ConstantOp>(
      acoOp.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
  auto extents = shape.getPairs();
  for (auto i = extents.begin(); i < extents.end();) {
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
    indexArgs.push_back(one);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp, fir::SliceOp slice,
                              SmallVectorImpl<mlir::Value> &indexArgs,
                              mlir::PatternRewriter &rewriter) {
  auto extents = slice.getTriples();
  for (auto i = extents.begin(); i < extents.end();) {
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
    indexArgs.push_back(*i++);
  }
}

static void populateIndexArgs(fir::ArrayCoorOp acoOp,
                              SmallVectorImpl<mlir::Value> &indexArgs,
                              mlir::PatternRewriter &rewriter) {
  if (auto shape = acoOp.getShape().getDefiningOp<ShapeOp>())
    return populateIndexArgs(acoOp, shape, indexArgs, rewriter);
  if (auto shapeShift = acoOp.getShape().getDefiningOp<ShapeShiftOp>())
    return populateIndexArgs(acoOp, shapeShift, indexArgs, rewriter);
  if (auto slice = acoOp.getShape().getDefiningOp<SliceOp>())
    return populateIndexArgs(acoOp, slice, indexArgs, rewriter);
  return;
}

/// Returns affine.apply and fir.convert from array_coor and gendims
static std::pair<mlir::AffineApplyOp, fir::ConvertOp>
createAffineOps(mlir::Value arrayRef, mlir::PatternRewriter &rewriter) {
  auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>();
  auto affineMap =
      createArrayIndexAffineMap(acoOp.getIndices().size(), acoOp.getContext());
  SmallVector<mlir::Value> indexArgs;
  indexArgs.append(acoOp.getIndices().begin(), acoOp.getIndices().end());

  populateIndexArgs(acoOp, indexArgs, rewriter);

  auto affineApply = rewriter.create<mlir::AffineApplyOp>(acoOp.getLoc(),
                                                          affineMap, indexArgs);
  auto arrayElementType = coordinateArrayElement(acoOp);
  auto newType = mlir::MemRefType::get({-1}, arrayElementType);
  auto arrayConvert = rewriter.create<fir::ConvertOp>(acoOp.getLoc(), newType,
                                                      acoOp.getMemref());
  return std::make_pair(affineApply, arrayConvert);
}

static void rewriteLoad(fir::LoadOp loadOp, mlir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(loadOp);
  auto affineOps = createAffineOps(loadOp.getMemref(), rewriter);
  rewriter.replaceOpWithNewOp<mlir::AffineLoadOp>(
      loadOp, affineOps.second.getResult(), affineOps.first.getResult());
}

static void rewriteStore(fir::StoreOp storeOp,
                         mlir::PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(storeOp);
  auto affineOps = createAffineOps(storeOp.getMemref(), rewriter);
  rewriter.replaceOpWithNewOp<mlir::AffineStoreOp>(storeOp, storeOp.getValue(),
                                                   affineOps.second.getResult(),
                                                   affineOps.first.getResult());
}

static void rewriteMemoryOps(Block *block, mlir::PatternRewriter &rewriter) {
  for (auto &bodyOp : block->getOperations()) {
    if (isa<fir::LoadOp>(bodyOp))
      rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter);
    if (isa<fir::StoreOp>(bodyOp))
      rewriteStore(cast<fir::StoreOp>(bodyOp), rewriter);
  }
}

namespace {
/// Convert `fir.do_loop` to `affine.for`, creates fir.convert for arrays to
/// memref, rewrites array_coor to affine.apply with affine_map. Rewrites fir
/// loads and stores to affine.
class AffineLoopConversion : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineLoopConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: rewriting loop:\n";
               loop.dump(););
    LLVM_ATTRIBUTE_UNUSED auto loopAnalysis =
        functionAnalysis.getChildLoopAnalysis(loop);
    auto &loopOps = loop.getBody()->getOperations();
    auto loopAndIndex = createAffineFor(loop, rewriter);
    auto affineFor = loopAndIndex.first;
    auto inductionVar = loopAndIndex.second;

    rewriter.startRootUpdate(affineFor.getOperation());
    affineFor.getBody()->getOperations().splice(
        std::prev(affineFor.getBody()->end()), loopOps, loopOps.begin(),
        std::prev(loopOps.end()));
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
  createAffineFor(fir::DoLoopOp op, mlir::PatternRewriter &rewriter) const {
    if (auto constantStep = constantIntegerLike(op.getStep()))
      if (constantStep.getValue() > 0)
        return positiveConstantStep(op, constantStep.getValue(), rewriter);
    return genericBounds(op, rewriter);
  }

  // when step for the loop is positive compile time constant
  std::pair<mlir::AffineForOp, mlir::Value>
  positiveConstantStep(fir::DoLoopOp op, int64_t step,
                       mlir::PatternRewriter &rewriter) const {
    auto affineFor = rewriter.create<mlir::AffineForOp>(
        op.getLoc(), ValueRange(op.getLowerBound()),
        mlir::AffineMap::get(0, 1,
                             mlir::getAffineSymbolExpr(0, op.getContext())),
        ValueRange(op.getUpperBound()),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
        step);
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }

  std::pair<mlir::AffineForOp, mlir::Value>
  genericBounds(fir::DoLoopOp op, mlir::PatternRewriter &rewriter) const {
    auto lowerBound = mlir::getAffineSymbolExpr(0, op.getContext());
    auto upperBound = mlir::getAffineSymbolExpr(1, op.getContext());
    auto step = mlir::getAffineSymbolExpr(2, op.getContext());
    mlir::AffineMap upperBoundMap = mlir::AffineMap::get(
        0, 3, (upperBound - lowerBound + step).floorDiv(step));
    auto genericUpperBound = rewriter.create<mlir::AffineApplyOp>(
        op.getLoc(), upperBoundMap,
        ValueRange({op.getLowerBound(), op.getUpperBound(), op.getStep()}));
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
        ValueRange(
            {affineFor.getInductionVar(), op.getLowerBound(), op.getStep()}));
    return std::make_pair(affineFor, actualIndex.getResult());
  }

  AffineFunctionAnalysis &functionAnalysis;
};

/// Convert `fir.if` to `affine.if`.
class AffineIfConversion : public mlir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineIfConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context) {}
  mlir::LogicalResult
  matchAndRewrite(fir::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: rewriting if:\n";
               op.dump(););
    auto &ifOps = op.getThenRegion().front().getOperations();
    auto affineCondition = AffineIfCondition(op.getCondition());
    if (!affineCondition.hasIntegerSet()) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineIfConversion: couldn't calculate affine condition\n";);
      return failure();
    }
    auto affineIf = rewriter.create<mlir::AffineIfOp>(
        op.getLoc(), affineCondition.getIntegerSet(),
        affineCondition.getAffineArgs(), !op.getElseRegion().empty());
    rewriter.startRootUpdate(affineIf);
    affineIf.getThenBlock()->getOperations().splice(
        std::prev(affineIf.getThenBlock()->end()), ifOps, ifOps.begin(),
        std::prev(ifOps.end()));
    if (!op.getElseRegion().empty()) {
      auto &otherOps = op.getElseRegion().front().getOperations();
      affineIf.getElseBlock()->getOperations().splice(
          std::prev(affineIf.getElseBlock()->end()), otherOps, otherOps.begin(),
          std::prev(otherOps.end()));
    }
    rewriter.finalizeRootUpdate(affineIf);
    rewriteMemoryOps(affineIf.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: if converted to:\n";
               affineIf.dump(););
    rewriter.replaceOp(op, affineIf.getOperation()->getResults());
    return success();
  }
};

/// Promote fir.do_loop and fir.if to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public AffineDialectPromotionBase<AffineDialectPromotion> {
public:
  void runOnOperation() override {

    auto *context = &getContext();
    auto function = getOperation();
    markAllAnalysesPreserved();
    auto functionAnalysis = AffineFunctionAnalysis(function);
    mlir::RewritePatternSet patterns(context);
    patterns.insert<AffineIfConversion>(context, functionAnalysis);
    patterns.insert<AffineLoopConversion>(context, functionAnalysis);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<
        mlir::AffineDialect, FIROpsDialect, mlir::scf::SCFDialect,
        mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();
    target.addDynamicallyLegalOp<IfOp>([&functionAnalysis](fir::IfOp op) {
      return !(functionAnalysis.getChildIfAnalysis(op).canPromoteToAffine());
    });
    target.addDynamicallyLegalOp<DoLoopOp>([&functionAnalysis](
                                               fir::DoLoopOp op) {
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
