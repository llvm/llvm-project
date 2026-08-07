//===- LoopNestPattern.cpp - Loop nest classification ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopNestPattern.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace cir;
using namespace cir::loopopt;

// Bounds compile time on pathological expressions.
static constexpr unsigned kMaxExprDepth = 16;

// Two values may only be compared or combined when they agree on width and
// signedness. The frontend guarantees that within one expression, but not
// across the separately emitted parts of a nest.
static bool sameForm(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return lhs.getBitWidth() == rhs.getBitWidth() &&
         lhs.isUnsigned() == rhs.isUnsigned();
}

static llvm::APSInt like(const llvm::APSInt &model, int64_t value) {
  return llvm::APSInt(
      llvm::APInt(model.getBitWidth(), value, /*isSigned=*/true),
      model.isUnsigned());
}

static mlir::FailureOr<llvm::APSInt> constantOfType(mlir::Type type,
                                                    int64_t value) {
  auto intType = mlir::dyn_cast<cir::IntType>(type);
  if (!intType)
    return mlir::failure();
  return llvm::APSInt(llvm::APInt(intType.getWidth(), value, /*isSigned=*/true),
                      intType.isUnsigned());
}

// Two limits are the same limit when they are the same value, or when both
// fold to the same number. The frontend emits a separate constant for each
// mention of a bound, so identity alone would miss the common case.
static bool limitsEquivalent(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return true;
  mlir::FailureOr<llvm::APSInt> lhsValue = evaluateConstantIntExpr(lhs);
  mlir::FailureOr<llvm::APSInt> rhsValue = evaluateConstantIntExpr(rhs);
  if (mlir::failed(lhsValue) || mlir::failed(rhsValue))
    return false;
  return sameForm(*lhsValue, *rhsValue) && *lhsValue == *rhsValue;
}

namespace {
/// An expression read as `coefficient * outerIV + offset`. A coefficient of
/// zero means the expression does not read the outer induction variable.
struct AffineTerm {
  llvm::APSInt coefficient;
  llvm::APSInt offset;
};
} // namespace

static mlir::FailureOr<AffineTerm>
matchAffineTerm(mlir::Value expr, mlir::Value outerIvSlot, cir::ForOp outerFor,
                unsigned &ivLoads, unsigned depth) {
  if (depth > kMaxExprDepth)
    return mlir::failure();

  // A subexpression that folds contributes only an offset. This is what
  // collapses limits the frontend leaves unfolded, such as n - k.
  if (mlir::FailureOr<llvm::APSInt> folded = evaluateConstantIntExpr(expr);
      mlir::succeeded(folded)) {
    mlir::FailureOr<llvm::APSInt> zero = constantOfType(expr.getType(), 0);
    if (mlir::failed(zero))
      return mlir::failure();
    return AffineTerm{*zero, *folded};
  }

  mlir::Operation *def = expr.getDefiningOp();
  if (!def)
    return mlir::failure();

  if (mlir::isa<cir::LoadOp>(def)) {
    // The read has to happen inside the outer loop, so it observes the
    // current iteration rather than a value snapshotted before the nest.
    if (!isOrdinaryLoadOfSlotIn(expr, outerIvSlot, outerFor.getOperation()))
      return mlir::failure();
    mlir::FailureOr<llvm::APSInt> one = constantOfType(expr.getType(), 1);
    mlir::FailureOr<llvm::APSInt> zero = constantOfType(expr.getType(), 0);
    if (mlir::failed(one) || mlir::failed(zero))
      return mlir::failure();
    ++ivLoads;
    return AffineTerm{*one, *zero};
  }

  // Checked here as well as in the condition vocabulary, because an affine
  // limit may be computed outside the condition block.
  if (!mlir::isa<cir::AddOp, cir::SubOp, cir::MulOp>(def) || isSaturating(def))
    return mlir::failure();
  mlir::FailureOr<AffineTerm> lhs = matchAffineTerm(
      def->getOperand(0), outerIvSlot, outerFor, ivLoads, depth + 1);
  mlir::FailureOr<AffineTerm> rhs = matchAffineTerm(
      def->getOperand(1), outerIvSlot, outerFor, ivLoads, depth + 1);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();

  mlir::FailureOr<llvm::APSInt> coefficient = mlir::failure();
  mlir::FailureOr<llvm::APSInt> offset = mlir::failure();
  if (mlir::isa<cir::AddOp>(def)) {
    coefficient = checkedAdd(lhs->coefficient, rhs->coefficient);
    offset = checkedAdd(lhs->offset, rhs->offset);
  } else if (mlir::isa<cir::SubOp>(def)) {
    coefficient = checkedSub(lhs->coefficient, rhs->coefficient);
    offset = checkedSub(lhs->offset, rhs->offset);
  } else {
    // Only one factor may vary, otherwise the expression is not affine.
    bool lhsIsConstant = lhs->coefficient.isZero();
    bool rhsIsConstant = rhs->coefficient.isZero();
    if (!lhsIsConstant && !rhsIsConstant)
      return mlir::failure();
    coefficient = lhsIsConstant ? checkedMul(rhs->coefficient, lhs->offset)
                                : checkedMul(lhs->coefficient, rhs->offset);
    offset = checkedMul(lhs->offset, rhs->offset);
  }
  if (mlir::failed(coefficient) || mlir::failed(offset))
    return mlir::failure();
  return AffineTerm{*coefficient, *offset};
}

// Read expr as a positive affine function of the outer induction variable,
// normalized to coefficient * outerIV + offset.
static mlir::FailureOr<AffineOuterIVRelation>
matchAffineInSlot(mlir::Value expr, mlir::Value outerIvSlot,
                  cir::ForOp outerFor) {
  unsigned ivLoads = 0;
  mlir::FailureOr<AffineTerm> term =
      matchAffineTerm(expr, outerIvSlot, outerFor, ivLoads, /*depth=*/0);
  if (mlir::failed(term))
    return mlir::failure();
  // Require exactly one mention of the outer variable, so a repeated form
  // such as i + i is not recognized.
  if (ivLoads != 1)
    return mlir::failure();
  if (term->coefficient.isZero() || term->coefficient.isNegative())
    return mlir::failure();
  return AffineOuterIVRelation{term->coefficient, term->offset};
}

static mlir::Value resolveInvariantSource(mlir::Value slot, cir::ForOp loop,
                                          mlir::Operation *use,
                                          mlir::DominanceInfo &dominance) {
  if (!slot.getDefiningOp<cir::AllocaOp>())
    return {};

  // The alloca use list is complete. Require one ordinary store and only
  // ordinary loads to prove that the stored argument remains fixed.
  cir::StoreOp writer;
  for (mlir::Operation *user : slot.getUsers()) {
    if (auto load = mlir::dyn_cast<cir::LoadOp>(user)) {
      if (!isOrdinaryAccess(load) || load.getAddr() != slot)
        return {};
      continue;
    }
    auto store = mlir::dyn_cast<cir::StoreOp>(user);
    if (!store || store.getAddr() != slot || store.getValue() == slot)
      return {};
    if (!isOrdinaryAccess(store) || writer)
      return {};
    writer = store;
  }
  if (!writer || loop->isAncestor(writer))
    return {};
  if (!dominance.dominates(writer.getOperation(), use) ||
      !dominance.dominates(writer.getOperation(), loop.getOperation()))
    return {};

  // Only function arguments are recognized as invariant sources.
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(writer.getValue());
  if (!argument || !mlir::isa<cir::FuncOp>(argument.getOwner()->getParentOp()))
    return {};
  return argument;
}

static bool isIncreasingStrictLess(const CountedLoop &loop) {
  return loop.direction == StepDirection::Increment &&
         loop.condition.predicate == cir::CmpOpKind::lt;
}

// Every recognized domain is reasoned about with signed arithmetic, so an
// unsigned counter is left unrecognized rather than assumed to behave.
static bool hasSignedControlType(const CountedLoop &loop) {
  auto type =
      mlir::dyn_cast<cir::IntType>(loop.condition.ivDependentExpr.getType());
  return type && type.isSigned();
}

static bool comparesDirectIv(const CountedLoop &loop) {
  cir::ForOp forOp = loop.forOp;
  return isOrdinaryLoadOfSlotIn(loop.condition.ivDependentExpr, loop.ivSlot,
                                forOp.getOperation());
}

// The loop counts from a known value up to a known value, over at least one
// iteration.
static mlir::FailureOr<std::pair<llvm::APSInt, llvm::APSInt>>
constantDomain(const CountedLoop &loop) {
  mlir::FailureOr<llvm::APSInt> first =
      evaluateConstantIntExpr(loop.initialExpr);
  mlir::FailureOr<llvm::APSInt> limit =
      evaluateConstantIntExpr(loop.condition.nonIvExpr);
  if (mlir::failed(first) || mlir::failed(limit) || !sameForm(*first, *limit))
    return mlir::failure();
  if (!(*first < *limit))
    return mlir::failure();
  return std::make_pair(*first, *limit);
}

static bool startsAtZero(const CountedLoop &loop) {
  mlir::FailureOr<llvm::APSInt> first =
      evaluateConstantIntExpr(loop.initialExpr);
  return mlir::succeeded(first) && first->isZero();
}

// Shared entry conditions for every pattern. Both loops count up under a
// strict less-than over signed values, and the outer loop compares its own
// variable directly. Each pattern decides separately whether it also needs a
// constant outer range.
static bool matchCommonPreconditions(const CountedLoop &outer,
                                     const CountedLoop &inner) {
  return isIncreasingStrictLess(outer) && isIncreasingStrictLess(inner) &&
         hasSignedControlType(outer) && hasSignedControlType(inner) &&
         comparesDirectIv(outer);
}

static mlir::FailureOr<LoopNestPattern>
matchInnerAffineUpper(const CountedLoop &outer, const CountedLoop &inner) {
  if (!comparesDirectIv(inner))
    return mlir::failure();
  // The outer range is needed here to bound the inner limit.
  mlir::FailureOr<std::pair<llvm::APSInt, llvm::APSInt>> outerDomain =
      constantDomain(outer);
  if (mlir::failed(outerDomain))
    return mlir::failure();

  mlir::FailureOr<AffineOuterIVRelation> affine =
      matchAffineInSlot(inner.condition.nonIvExpr, outer.ivSlot, outer.forOp);
  if (mlir::failed(affine) || !startsAtZero(inner))
    return mlir::failure();

  // The limit is evaluated once per outer iteration, so it has to stay
  // representable across the whole outer range.
  const llvm::APSInt &first = outerDomain->first;
  const llvm::APSInt &limit = outerDomain->second;
  if (!sameForm(affine->coefficient, first))
    return mlir::failure();
  mlir::FailureOr<llvm::APSInt> last = checkedSub(limit, like(limit, 1));
  if (mlir::failed(last))
    return mlir::failure();
  auto evaluateAt = [&](const llvm::APSInt &point) {
    mlir::FailureOr<llvm::APSInt> scaled =
        checkedMul(affine->coefficient, point);
    if (mlir::failed(scaled))
      return mlir::FailureOr<llvm::APSInt>(mlir::failure());
    return checkedAdd(*scaled, affine->offset);
  };
  if (mlir::failed(evaluateAt(first)) || mlir::failed(evaluateAt(*last)))
    return mlir::failure();

  return LoopNestPattern{outer, inner, LoopNestPatternKind::InnerAffineUpper,
                         *affine};
}

// No constant outer range is required here. Demanding one would drop the
// middle pair of a symmetric triple nest. The shared limit keeps both loops
// counted, since it folds to a constant or is one value computed outside
// them.
static mlir::FailureOr<LoopNestPattern>
matchOuterIVInnerStart(const CountedLoop &outer, const CountedLoop &inner) {
  if (!comparesDirectIv(inner))
    return mlir::failure();

  // The inner loop walks from the current outer value to the shared limit.
  cir::ForOp outerFor = outer.forOp;
  if (!isOrdinaryLoadOfSlotIn(inner.initialExpr, outer.ivSlot,
                              outerFor.getOperation()))
    return mlir::failure();
  if (!limitsEquivalent(outer.condition.nonIvExpr, inner.condition.nonIvExpr))
    return mlir::failure();

  return LoopNestPattern{outer, inner, LoopNestPatternKind::OuterIVInnerStart,
                         std::nullopt};
}

static mlir::FailureOr<LoopNestPattern>
matchInnerProductUpper(const CountedLoop &outer, const CountedLoop &inner) {
  // The outer range is needed here to bound the largest product computed.
  mlir::FailureOr<std::pair<llvm::APSInt, llvm::APSInt>> outerDomain =
      constantDomain(outer);
  if (mlir::failed(outerDomain))
    return mlir::failure();

  // Exactly the product of the two counters, in either operand order, with
  // nothing else folded in.
  auto product = inner.condition.ivDependentExpr.getDefiningOp<cir::MulOp>();
  if (!product)
    return mlir::failure();
  mlir::Value lhs = product.getLhs();
  mlir::Value rhs = product.getRhs();
  cir::ForOp outerFor = outer.forOp;
  cir::ForOp innerFor = inner.forOp;
  bool innerThenOuter =
      isOrdinaryLoadOfSlotIn(lhs, inner.ivSlot, innerFor.getOperation()) &&
      isOrdinaryLoadOfSlotIn(rhs, outer.ivSlot, outerFor.getOperation());
  bool outerThenInner =
      isOrdinaryLoadOfSlotIn(rhs, inner.ivSlot, innerFor.getOperation()) &&
      isOrdinaryLoadOfSlotIn(lhs, outer.ivSlot, outerFor.getOperation());
  if (!innerThenOuter && !outerThenInner)
    return mlir::failure();

  if (!startsAtZero(inner))
    return mlir::failure();
  // A strictly positive outer start keeps the product increasing with the
  // inner counter, so the loop is still counted.
  if (!outerDomain->first.isStrictlyPositive())
    return mlir::failure();

  // The product is compared against the same constant the outer loop stops
  // at. At termination, j * i < N + i. Since i <= N - 1, require N + (N - 1)
  // to remain representable.
  mlir::FailureOr<llvm::APSInt> productLimit =
      evaluateConstantIntExpr(inner.condition.nonIvExpr);
  if (mlir::failed(productLimit) ||
      !sameForm(*productLimit, outerDomain->second) ||
      *productLimit != outerDomain->second)
    return mlir::failure();
  mlir::FailureOr<llvm::APSInt> largestOuter =
      checkedSub(outerDomain->second, like(outerDomain->second, 1));
  if (mlir::failed(largestOuter) ||
      mlir::failed(checkedAdd(*productLimit, *largestOuter)))
    return mlir::failure();

  return LoopNestPattern{outer, inner, LoopNestPatternKind::InnerProductUpper,
                         std::nullopt};
}

static mlir::FailureOr<LoopNestPattern>
matchInvariantInnerStart(const CountedLoop &outer, const CountedLoop &inner,
                         mlir::DominanceInfo &dominance) {
  if (!comparesDirectIv(inner))
    return mlir::failure();
  // No shared limit ties the two loops together, so each limit has to be
  // known fixed on its own.
  if (mlir::failed(constantDomain(outer)) ||
      mlir::failed(evaluateConstantIntExpr(inner.condition.nonIvExpr)))
    return mlir::failure();

  // Requiring the start to be a proven invariant read, never a constant,
  // keeps ordinary rectangular nests out of this pattern.
  cir::ForOp outerFor = outer.forOp;
  auto start = inner.initialExpr.getDefiningOp<cir::LoadOp>();
  if (!start || !isOrdinaryAccess(start) || !outerFor->isAncestor(start))
    return mlir::failure();
  if (!resolveInvariantSource(start.getAddr(), outerFor, start.getOperation(),
                              dominance))
    return mlir::failure();

  return LoopNestPattern{outer, inner, LoopNestPatternKind::InvariantInnerStart,
                         std::nullopt};
}

mlir::FailureOr<LoopNestPattern>
cir::loopopt::matchLoopNestPattern(const CountedLoop &outer,
                                   const CountedLoop &inner,
                                   mlir::DominanceInfo &dominance) {
  // Both loops must count up under a strict less-than over signed values,
  // and the outer loop must compare its own variable. Settled once here
  // rather than in each matcher.
  if (!matchCommonPreconditions(outer, inner))
    return mlir::failure();
  if (mlir::FailureOr<LoopNestPattern> pattern =
          matchInnerAffineUpper(outer, inner);
      mlir::succeeded(pattern))
    return pattern;
  if (mlir::FailureOr<LoopNestPattern> pattern =
          matchOuterIVInnerStart(outer, inner);
      mlir::succeeded(pattern))
    return pattern;
  if (mlir::FailureOr<LoopNestPattern> pattern =
          matchInnerProductUpper(outer, inner);
      mlir::succeeded(pattern))
    return pattern;
  return matchInvariantInnerStart(outer, inner, dominance);
}
