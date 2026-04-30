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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_AFFINEDIALECTPROMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

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
    funcOp->walk([&](fir::DoLoopOp doloop) {
      fir::DoLoopOp outermost = doloop;
      while (auto parent = outermost->getParentOfType<fir::DoLoopOp>())
        outermost = parent;
      outermostLoopMap[doloop] = outermost;
      loopAnalysisMap.try_emplace(doloop, doloop, *this);
    });
  }

  AffineLoopAnalysis getChildLoopAnalysis(fir::DoLoopOp op) const;

  AffineIfAnalysis getChildIfAnalysis(fir::IfOp op) const;

  fir::DoLoopOp getOutermostLoop(fir::DoLoopOp op) const {
    auto it = outermostLoopMap.find(op.getOperation());
    assert(it != outermostLoopMap.end());
    return it->second;
  }

  llvm::DenseMap<mlir::Operation *, AffineLoopAnalysis> loopAnalysisMap;
  llvm::DenseMap<mlir::Operation *, AffineIfAnalysis> ifAnalysisMap;
  llvm::DenseMap<mlir::Operation *, fir::DoLoopOp> outermostLoopMap;
};
} // namespace

/// Recursively checks whether a value can be expressed as an affine function
/// of loop induction variables, integer constants, and loop-invariant symbols
/// (values defined outside the outermost loop of the nest).
///
/// When \p outermost is provided, values defined outside it are accepted as
/// valid affine symbols.  When nullptr, only loop IVs and constants are
/// accepted (legacy behavior).
static bool isAffineIndex(mlir::Value val, fir::DoLoopOp outermost = nullptr,
                          unsigned depth = 0) {
  if (depth > 16)
    return false;

  if (auto conv = val.getDefiningOp<fir::ConvertOp>())
    return isAffineIndex(conv.getValue(), outermost, depth + 1);

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()) ||
        isa<mlir::affine::AffineForOp>(blockArg.getOwner()->getParentOp()))
      return true;
    if (outermost && !outermost->isAncestor(blockArg.getOwner()->getParentOp()))
      return true;
    return false;
  }

  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  if (isa<mlir::arith::ConstantOp>(defOp))
    return true;

  if (auto add = dyn_cast<mlir::arith::AddIOp>(defOp))
    return isAffineIndex(add.getLhs(), outermost, depth + 1) &&
           isAffineIndex(add.getRhs(), outermost, depth + 1);

  if (auto sub = dyn_cast<mlir::arith::SubIOp>(defOp))
    return isAffineIndex(sub.getLhs(), outermost, depth + 1) &&
           isAffineIndex(sub.getRhs(), outermost, depth + 1);

  if (auto mul = dyn_cast<mlir::arith::MulIOp>(defOp)) {
    auto *lhsDef = mul.getLhs().getDefiningOp();
    auto *rhsDef = mul.getRhs().getDefiningOp();
    if ((lhsDef && isa<mlir::arith::ConstantOp>(lhsDef)) ||
        (rhsDef && isa<mlir::arith::ConstantOp>(rhsDef)))
      return isAffineIndex(mul.getLhs(), outermost, depth + 1) &&
             isAffineIndex(mul.getRhs(), outermost, depth + 1);
    return false;
  }

  // Value defined outside the outermost loop → valid affine symbol.
  if (outermost && !outermost->isAncestor(defOp))
    return true;

  LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: index is not an affine "
                             "expression of loop IVs\n";
             defOp->dump());
  return false;
}

/// Builds an mlir::AffineExpr by recursively walking the FIR/arith expression
/// tree rooted at a fir.array_coor index value.  Loop induction variables
/// become affine dimensions; integer constants are folded into the expression.
/// Values defined outside the outermost enclosing loop are classified as
/// affine symbols — they are loop-invariant across the entire nest.
struct AffineIndexBuilder {
  using MaybeExpr = std::optional<mlir::AffineExpr>;

  explicit AffineIndexBuilder(mlir::MLIRContext *ctx,
                              fir::DoLoopOp outermost = nullptr)
      : context(ctx), outermostLoop(outermost) {}

  MaybeExpr build(mlir::Value val) {
    if (auto conv = val.getDefiningOp<fir::ConvertOp>())
      return build(conv.getValue());

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
      if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()) ||
          isa<mlir::affine::AffineForOp>(blockArg.getOwner()->getParentOp())) {
        for (unsigned i = 0; i < dims.size(); ++i)
          if (dims[i] == val)
            return mlir::getAffineDimExpr(i, context);
        unsigned idx = dims.size();
        dims.push_back(val);
        return mlir::getAffineDimExpr(idx, context);
      }
      if (outermostLoop &&
          !outermostLoop->isAncestor(blockArg.getOwner()->getParentOp()))
        return addSymbol(val);
      return {};
    }

    auto *defOp = val.getDefiningOp();
    if (!defOp)
      return {};

    if (auto op = dyn_cast<mlir::arith::ConstantOp>(defOp))
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue()))
        return mlir::getAffineConstantExpr(intAttr.getInt(), context);

    if (auto op = dyn_cast<mlir::arith::AddIOp>(defOp)) {
      auto lhs = build(op.getLhs());
      auto rhs = build(op.getRhs());
      if (lhs && rhs)
        return *lhs + *rhs;
      return {};
    }

    if (auto op = dyn_cast<mlir::arith::SubIOp>(defOp)) {
      auto lhs = build(op.getLhs());
      auto rhs = build(op.getRhs());
      if (lhs && rhs)
        return *lhs - *rhs;
      return {};
    }

    if (auto op = dyn_cast<mlir::arith::MulIOp>(defOp)) {
      auto lhs = build(op.getLhs());
      auto rhs = build(op.getRhs());
      if (lhs && rhs)
        return *lhs * *rhs;
      return {};
    }

    // Value defined outside the outermost loop → affine symbol.
    if (outermostLoop && !outermostLoop->isAncestor(defOp))
      return addSymbol(val);

    return {};
  }

  mlir::MLIRContext *context;
  fir::DoLoopOp outermostLoop;
  llvm::SmallVector<mlir::Value> dims;
  llvm::SmallVector<mlir::Value> syms;

private:
  MaybeExpr addSymbol(mlir::Value val) {
    for (unsigned i = 0; i < syms.size(); ++i)
      if (syms[i] == val)
        return mlir::getAffineSymbolExpr(i, context);
    unsigned idx = syms.size();
    syms.push_back(val);
    return mlir::getAffineSymbolExpr(idx, context);
  }
};

/// Ensure a value is index-typed, inserting a fir.convert immediately after
/// the value's definition point if needed.  Affine operations require all
/// dimension and symbol operands to be of index type.
static mlir::Value castToIndex(mlir::Value val,
                               mlir::PatternRewriter &rewriter) {
  if (val.getType().isIndex())
    return val;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  if (auto *defOp = val.getDefiningOp())
    rewriter.setInsertionPointAfter(defOp);
  else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val))
    rewriter.setInsertionPointToStart(blockArg.getOwner());
  return fir::ConvertOp::create(rewriter, val.getLoc(), rewriter.getIndexType(),
                                val);
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
    // Reject loops containing fir.if until full fir.if → affine.if
    // promotion is available.
    if (!loopOperation.getOps<fir::IfOp>().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: loop contains fir.if, "
                                 "skipping (if-promotion not yet enabled)\n");
      return false;
    }
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
    auto outermost = functionAnalysis.getOutermostLoop(loopOperation);
    return analyzeBounds(loopOperation, outermost) &&
           analyzeMemoryAccess(loopOperation, outermost) &&
           analysisResults(loopOperation) &&
           analyzeBody(loopOperation, functionAnalysis);
  }

  bool analyzeReference(mlir::Value memref, mlir::Operation *op,
                        fir::DoLoopOp outermost) {
    if (auto acoOp = memref.getDefiningOp<ArrayCoorOp>()) {
      if (mlir::isa<fir::BoxType>(acoOp.getMemref().getType())) {
        // TODO: Look if and how fir.box can be promoted to affine.
        LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                   "array memory operation uses fir.box\n";
                   op->dump(); acoOp.dump(););
        return false;
      }
      // TODO: Support fir.array_coor with a fir.slice operand. The current
      // promotion path only inspects acoOp.getShape() and silently ignores
      // acoOp.getSlice(). Reject these loops until full slice
      // support is implemented.
      if (acoOp.getSlice()) {
        LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: cannot promote loop, "
                                   "fir.array_coor has a fir.slice operand "
                                   "(not yet supported)\n";
                   op->dump(); acoOp.dump(););
        return false;
      }

      // Reject element types that `mlir::MemRefType` cannot hold (e.g.
      // `!fir.char`) — promotion
      // would later build an invalid `MemRefType`.
      fir::SequenceType seqType;
      mlir::Type baseTy = acoOp.getMemref().getType();
      if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(baseTy))
        seqType = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy());
      else if (auto heapTy = mlir::dyn_cast<fir::HeapType>(baseTy))
        seqType = mlir::dyn_cast<fir::SequenceType>(heapTy.getEleTy());
      if (!seqType ||
          !mlir::MemRefType::isValidElementType(seqType.getEleTy())) {
        LLVM_DEBUG(llvm::dbgs()
                       << "AffineLoopAnalysis: array element type is not a "
                          "valid MemRef element type, cannot promote\n";
                   op->dump(); acoOp.dump(););
        return false;
      }
      bool canPromote = true;
      for (auto coordinate : acoOp.getIndices())
        canPromote = canPromote && isAffineIndex(coordinate, outermost);
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

  bool analyzeMemoryAccess(fir::DoLoopOp loopOperation,
                           fir::DoLoopOp outermost) {
    for (auto loadOp : loopOperation.getOps<fir::LoadOp>())
      if (!analyzeReference(loadOp.getMemref(), loadOp, outermost))
        return false;
    for (auto storeOp : loopOperation.getOps<fir::StoreOp>())
      if (!analyzeReference(storeOp.getMemref(), storeOp, outermost))
        return false;
    return true;
  }

  bool analyzeBounds(fir::DoLoopOp loopOperation, fir::DoLoopOp outermost) {
    // Only promote loops with a positive constant step. The genericBounds
    // fallback (which attempts to handle variable/negative steps) is broken
    // — see the comment on that function — so we reject everything that
    // positiveConstantStep cannot handle.
    bool hasPositiveConstantStep = false;
    if (auto defOp =
            loopOperation.getStep().getDefiningOp<mlir::arith::ConstantOp>())
      if (auto attr = mlir::dyn_cast<IntegerAttr>(defOp.getValue()))
        hasPositiveConstantStep = attr.getInt() > 0;
    if (!hasPositiveConstantStep) {
      LLVM_DEBUG(llvm::dbgs() << "AffineLoopAnalysis: step is not a positive "
                                 "constant, cannot promote\n");
      return false;
    }
    if (!isAffineIndex(loopOperation.getLowerBound(), outermost)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "AffineLoopAnalysis: lower bound not affine\n");
      return false;
    }
    if (!isAffineIndex(loopOperation.getUpperBound(), outermost)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "AffineLoopAnalysis: upper bound not affine\n");
      return false;
    }
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
  using MaybeAffineExpr = std::optional<mlir::AffineExpr>;

  explicit AffineIfCondition(mlir::Value fc) : firCondition(fc) {
    if (auto condDef = firCondition.getDefiningOp<mlir::arith::CmpIOp>())
      fromCmpIOp(condDef);
  }

  bool hasIntegerSet() const { return integerSet.has_value(); }

  mlir::IntegerSet getIntegerSet() const {
    assert(hasIntegerSet() && "integer set is missing");
    return *integerSet;
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
      return mlir::getAffineBinaryOpExpr(kind, *lhs, *rhs);
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
      if (auto intConstant = mlir::dyn_cast<IntegerAttr>(op.getValue()))
        return toAffineExpr(intConstant.getInt());
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      affineArgs.push_back(value);
      if (isa<fir::DoLoopOp>(blockArg.getOwner()->getParentOp()) ||
          isa<mlir::affine::AffineForOp>(blockArg.getOwner()->getParentOp()))
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
    auto constraintPair =
        constraint(cmpOp.getPredicate(), *rhsAffine - *lhsAffine);
    if (!constraintPair)
      return;
    integerSet = mlir::IntegerSet::get(
        dimCount, symCount, {constraintPair->first}, {constraintPair->second});
  }

  std::optional<std::pair<AffineExpr, bool>>
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
  std::optional<mlir::IntegerSet> integerSet;
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

    return {};
  }
  return it->getSecond();
}

static std::optional<int64_t> constantIntegerLike(const mlir::Value value) {
  if (auto definition = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto stepAttr = mlir::dyn_cast<IntegerAttr>(definition.getValue()))
      return stepAttr.getInt();
  return {};
}

/// Holds the result of creating multi-dimensional affine operations.
struct MultiDimAffineResult {
  SmallVector<mlir::Value> indices;
  fir::ConvertOp arrayConvert;
};

/// Creates multi-dimensional affine operations preserving array dimensionality.
/// Instead of linearizing all indices into a single 1D offset, this extracts
/// the array shape from the FIR SequenceType, creates a matching multi-dim
/// MemRefType, and adjusts each per-dimension index from Fortran 1-based to
/// memref 0-based indexing.
static MultiDimAffineResult
createMultiDimAffineOps(mlir::Value arrayRef, mlir::PatternRewriter &rewriter,
                        fir::DoLoopOp outermost) {
  auto acoOp = arrayRef.getDefiningOp<ArrayCoorOp>();
  auto loc = acoOp.getLoc();
  auto *context = acoOp.getContext();

  fir::SequenceType seqType;
  if (auto refType =
          mlir::dyn_cast<fir::ReferenceType>(acoOp.getMemref().getType()))
    seqType = mlir::dyn_cast<fir::SequenceType>(refType.getEleTy());
  else if (auto heapType =
               mlir::dyn_cast<fir::HeapType>(acoOp.getMemref().getType()))
    seqType = mlir::dyn_cast<fir::SequenceType>(heapType.getEleTy());

  // need change because memref is row major order but fir.array is column major
  // order
  SmallVector<int64_t> reversedShape(seqType.getShape().rbegin(),
                                     seqType.getShape().rend());

  auto newType = mlir::MemRefType::get(reversedShape, seqType.getEleTy());
  auto arrayConvert =
      fir::ConvertOp::create(rewriter, loc, newType, acoOp.getMemref());

  SmallVector<mlir::Value> adjustedIndices;
  auto indices = acoOp.getIndices();

  auto buildOperands = [&](AffineIndexBuilder &builder) {
    SmallVector<mlir::Value> operands;
    for (auto &d : builder.dims)
      operands.push_back(castToIndex(d, rewriter));
    for (auto &s : builder.syms)
      operands.push_back(castToIndex(s, rewriter));
    return operands;
  };

  if (auto shapeOp = acoOp.getShape().getDefiningOp<ShapeOp>()) {
    for (auto idx : indices) {
      AffineIndexBuilder builder(context, outermost);
      auto expr = builder.build(idx);
      assert(expr && "analysis guaranteed index is affine");
      auto adjustedExpr = *expr - 1;
      auto map = mlir::AffineMap::get(builder.dims.size(), builder.syms.size(),
                                      adjustedExpr);
      auto operands = buildOperands(builder);
      auto adjusted =
          affine::AffineApplyOp::create(rewriter, loc, map, operands);
      adjustedIndices.push_back(adjusted.getResult());
    }
  } else if (auto shapeShiftOp =
                 acoOp.getShape().getDefiningOp<ShapeShiftOp>()) {
    auto pairs = shapeShiftOp.getPairs();
    for (unsigned i = 0; i < indices.size(); ++i) {
      AffineIndexBuilder builder(context, outermost);
      auto expr = builder.build(indices[i]);
      assert(expr && "analysis guaranteed index is affine");
      unsigned extraSymIdx = builder.syms.size();
      auto adjustedExpr =
          *expr - mlir::getAffineSymbolExpr(extraSymIdx, context);
      auto map = mlir::AffineMap::get(builder.dims.size(),
                                      builder.syms.size() + 1, adjustedExpr);
      auto operands = buildOperands(builder);
      operands.push_back(pairs[i * 2]);
      auto adjusted =
          affine::AffineApplyOp::create(rewriter, loc, map, operands);
      adjustedIndices.push_back(adjusted.getResult());
    }
  } else {
    llvm::report_fatal_error(
        "unsupported fir.array_coor shape kind; "
        "AffineLoopAnalysis::analyzeReference should have rejected this");
  }

  // need reverse because memref is row major order but fir.array is column
  // major order
  std::reverse(adjustedIndices.begin(), adjustedIndices.end());

  return {std::move(adjustedIndices), arrayConvert};
}

static void rewriteLoad(fir::LoadOp loadOp, mlir::PatternRewriter &rewriter,
                        fir::DoLoopOp outermost) {
  rewriter.setInsertionPoint(loadOp);
  auto result =
      createMultiDimAffineOps(loadOp.getMemref(), rewriter, outermost);
  rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
      loadOp, result.arrayConvert.getResult(), result.indices);
}

static void rewriteStore(fir::StoreOp storeOp, mlir::PatternRewriter &rewriter,
                         fir::DoLoopOp outermost) {
  rewriter.setInsertionPoint(storeOp);
  auto result =
      createMultiDimAffineOps(storeOp.getMemref(), rewriter, outermost);
  rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
      storeOp, storeOp.getValue(), result.arrayConvert.getResult(),
      result.indices);
}

static void rewriteMemoryOps(Block *block, mlir::PatternRewriter &rewriter,
                             fir::DoLoopOp outermost = {}) {
  for (auto &bodyOp : llvm::make_early_inc_range(block->getOperations())) {
    if (isa<fir::LoadOp>(bodyOp))
      rewriteLoad(cast<fir::LoadOp>(bodyOp), rewriter, outermost);
    else if (isa<fir::StoreOp>(bodyOp))
      rewriteStore(cast<fir::StoreOp>(bodyOp), rewriter, outermost);
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

  llvm::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: rewriting loop:\n";
               loop.dump(););
    [[maybe_unused]] auto loopAnalysis =
        functionAnalysis.getChildLoopAnalysis(loop);
    if (!loopAnalysis.canPromoteToAffine())
      return rewriter.notifyMatchFailure(loop, "cannot promote to affine");

    // All enclosing fir.do_loop ops must also be promotable.  Otherwise
    // this loop's affine operations would reference fir.do_loop block args
    // (not affine.for IVs) as dimension ids, which is invalid.
    for (auto *parent = loop->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto parentLoop = dyn_cast<fir::DoLoopOp>(parent)) {
        auto parentAnalysis = functionAnalysis.getChildLoopAnalysis(parentLoop);
        if (!parentAnalysis.canPromoteToAffine())
          return rewriter.notifyMatchFailure(
              loop, "enclosing fir.do_loop is not promotable");
      }
    }
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

    auto outermost = functionAnalysis.getOutermostLoop(loop);
    rewriteMemoryOps(affineFor.getBody(), rewriter, outermost);

    LLVM_DEBUG(llvm::dbgs() << "AffineLoopConversion: loop rewriten to:\n";
               affineFor.dump(););
    rewriter.replaceAllUsesWith(loopResults, affineFor->getResults());
    rewriter.eraseOp(loop);
    return success();
  }

private:
  std::pair<affine::AffineForOp, mlir::Value>
  createAffineFor(fir::DoLoopOp op, mlir::PatternRewriter &rewriter) const {
    if (auto constantStep = constantIntegerLike(op.getStep()))
      if (*constantStep > 0)
        return positiveConstantStep(op, *constantStep, rewriter);
    return genericBounds(op, rewriter);
  }

  mlir::AffineMap boundMap(mlir::Value operand, int64_t offset,
                           mlir::MLIRContext *ctx,
                           SmallVectorImpl<mlir::Value> &mapOperands,
                           fir::DoLoopOp outermost,
                           mlir::PatternRewriter &rewriter) const {
    AffineIndexBuilder builder(ctx, outermost);
    if (auto expr = builder.build(operand)) {
      for (auto &d : builder.dims)
        mapOperands.push_back(castToIndex(d, rewriter));
      for (auto &s : builder.syms)
        mapOperands.push_back(castToIndex(s, rewriter));
      return mlir::AffineMap::get(builder.dims.size(), builder.syms.size(),
                                  *expr + offset);
    }
    llvm_unreachable("analysis should have rejected non-affine bounds");
  }

  std::pair<affine::AffineForOp, mlir::Value>
  positiveConstantStep(fir::DoLoopOp op, int64_t step,
                       mlir::PatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto outermost = functionAnalysis.getOutermostLoop(op);
    SmallVector<mlir::Value> lbOperands, ubOperands;
    auto lbMap =
        boundMap(op.getLowerBound(), 0, ctx, lbOperands, outermost, rewriter);
    auto ubMap =
        boundMap(op.getUpperBound(), 1, ctx, ubOperands, outermost, rewriter);
    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), lbOperands, lbMap, ubOperands, ubMap, step,
        op.getIterOperands());
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }

  // KNOWN FLAWED: This function attempts to normalize any fir.do_loop
  // (variable step, negative step) into a 0-based step-1 affine.for by
  // computing trip_count = (ub - lb + step) / step and reconstructing
  // the original index as actual_i = lb + ii * step.
  //
  // Flaws:
  //   1. Operand classification: lb, ub, step are passed as affine symbols,
  //      but they may be dimensions (e.g. enclosing loop IVs), causing the
  //      MLIR verifier to reject with "dimensional operand cannot be used
  //      as a symbol".
  //   2. The index reconstruction (lb + ii * step) involves multiplying a
  //      dimension (ii) by a variable (step), which is not a valid affine
  //      expression when step is not a compile-time constant.
  //
  // Currently unreachable: analyzeBounds rejects loops with non-positive-
  // constant steps, so createAffineFor always takes the
  // positiveConstantStep path. This function is kept for reference only.
  std::pair<affine::AffineForOp, mlir::Value>
  genericBounds(fir::DoLoopOp op, mlir::PatternRewriter &rewriter) const {
    auto lowerBound = mlir::getAffineSymbolExpr(0, op.getContext());
    auto upperBound = mlir::getAffineSymbolExpr(1, op.getContext());
    auto step = mlir::getAffineSymbolExpr(2, op.getContext());
    mlir::AffineMap upperBoundMap = mlir::AffineMap::get(
        0, 3, (upperBound - lowerBound + step).floorDiv(step));
    auto genericUpperBound = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), upperBoundMap,
        ValueRange({op.getLowerBound(), op.getUpperBound(), op.getStep()}));
    auto actualIndexMap = mlir::AffineMap::get(
        1, 2,
        (lowerBound + mlir::getAffineDimExpr(0, op.getContext())) *
            mlir::getAffineSymbolExpr(1, op.getContext()));

    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), ValueRange(),
        AffineMap::getConstantMap(0, op.getContext()),
        genericUpperBound.getResult(),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
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
class AffineIfConversion : public mlir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineIfConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context), functionAnalysis(afa) {}
  llvm::LogicalResult
  matchAndRewrite(fir::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
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
    mlir::RewritePatternSet patterns(context);
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
std::unique_ptr<mlir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}
