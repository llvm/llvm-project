//===- BoundsChecking.cpp - Bounds checking related APIs --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements 'checkBounds', a function that compares memory offsets
//  (that may be symbolic) and uses heuristical workarounds to provide more
//  accurate results than the 'naive' evalBinOp calls.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BoundsChecking.h"

using namespace clang;
using namespace ento;

// NOTE: This function is the "heart" of this algorithm. It simplifies
// inequalities with transformations that are valid (and very elementary) in
// pure mathematics, but become invalid if we use them in C++ number model
// where the calculations may overflow.
// Due to the overflow issues I think it's impossible (or at least not
// practical) to integrate this kind of simplification into the resolution of
// arbitrary inequalities (i.e. the code of `evalBinOp`); but this function
// produces valid results when the calculations are handling memory offsets
// and every value is well below SIZE_MAX.
// NOTE: the simplification preserves the order of the two operands in a
// mathematical sense, but it may change the result produced by a C++
// comparison operator (and the automatic type conversions).
// For example, consider a comparison "X+1 < 0", where the LHS is stored as a
// size_t and the RHS is stored in an int. (As size_t is unsigned, this
// comparison is false for all values of "X".) However, the simplification may
// turn it into "X < -1", which is still always false in a mathematical sense,
// but can produce a true result when evaluated by `evalBinOp` (which follows
// the rules of C++ and casts -1 to SIZE_MAX).
static std::pair<NonLoc, nonloc::ConcreteInt>
getSimplifiedOffsets(NonLoc Offset, nonloc::ConcreteInt Extent,
                     SValBuilder &SVB) {
  const llvm::APSInt &ExtentVal = Extent.getValue();
  std::optional<nonloc::SymbolVal> SymVal = Offset.getAs<nonloc::SymbolVal>();
  if (SymVal && SymVal->isExpression()) {
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SymVal->getSymbol())) {
      llvm::APSInt Num = APSIntType(ExtentVal).convert(SIE->getRHS());
      switch (SIE->getOpcode()) {
      case BO_Mul:
        // The Num should never be 0 here, because multiplication by zero
        // is simplified by the engine.
        if ((ExtentVal % Num) != 0)
          return std::pair<NonLoc, nonloc::ConcreteInt>(Offset, Extent);
        else
          return getSimplifiedOffsets(nonloc::SymbolVal(SIE->getLHS()),
                                      SVB.makeIntVal(ExtentVal / Num), SVB);
      case BO_Add:
        return getSimplifiedOffsets(nonloc::SymbolVal(SIE->getLHS()),
                                    SVB.makeIntVal(ExtentVal - Num), SVB);
      default:
        break;
      }
    }
  }

  return std::pair<NonLoc, nonloc::ConcreteInt>(Offset, Extent);
}

static bool isNegative(SValBuilder &SVB, ProgramStateRef State, NonLoc Value) {
  const llvm::APSInt *MaxV = SVB.getMaxValue(State, Value);
  return MaxV && MaxV->isNegative();
}

static bool isUnsigned(SValBuilder &SVB, NonLoc Value) {
  QualType T = Value.getType(SVB.getContext());
  return T->isUnsignedIntegerType();
}

std::pair<ProgramStateRef, ProgramStateRef>
bounds::compareValueToThreshold(ProgramStateRef State, SValBuilder &SVB,
                                NonLoc Value, NonLoc Threshold,
                                Comparison CmpKind) {
  if (auto ConcreteThreshold = Threshold.getAs<nonloc::ConcreteInt>()) {
    std::tie(Value, Threshold) =
        getSimplifiedOffsets(Value, *ConcreteThreshold, SVB);
  }

  // We want to perform a _mathematical_ comparison between the numbers `Value`
  // and `Threshold`; but `evalBinOpNN` evaluates a C/C++ operator that may
  // perform automatic conversions. For example the number -1 is less than the
  // number 1000, but -1 < `1000ull` will evaluate to `false` because the `int`
  // -1 is converted to ULONGLONG_MAX.
  // To avoid automatic conversions, we evaluate the "obvious" cases without
  // calling `evalBinOpNN`:
  if (isNegative(SVB, State, Value) && isUnsigned(SVB, Threshold)) {
    if (CmpKind == Comparison::EQ) {
      // negative == unsigned is always false
      return {nullptr, State};
    }
    // negative < unsigned and negative <= unsigned are always true
    return {State, nullptr};
  }
  if (isUnsigned(SVB, Value) && isNegative(SVB, State, Threshold)) {
    // unsigned == negative, unsigned < negative and unsigned <= negative are
    // all always false
    return {nullptr, State};
  }
  // FIXME: These special cases are sufficient for handling real-world
  // comparisons, but in theory there could be contrived situations where
  // automatic conversion of a symbolic value (which can be negative and can be
  // positive) leads to incorrect results.
  // NOTE: We NEED to use the `evalBinOpNN` call in the "common" case, because
  // we want to ensure that assumptions coming from this precondition and
  // assumptions coming from regular C/C++ operator calls are represented by
  // constraints on the same symbolic expression. A solution that would
  // evaluate these "mathematical" comparisons through a separate pathway would
  // be a step backwards in this sense.

  const BinaryOperatorKind OpKind = asOpcode(CmpKind);
  auto BelowThreshold =
      SVB.evalBinOpNN(State, OpKind, Value, Threshold, SVB.getConditionType())
          .getAs<NonLoc>();

  if (BelowThreshold)
    return State->assume(*BelowThreshold);

  return {nullptr, nullptr};
}

bounds::CheckResult bounds::checkBounds(ProgramStateRef State, SValBuilder &SVB,
                                        NonLoc Offset,
                                        std::optional<NonLoc> Extent,
                                        bounds::CheckFlags Flags) {

  bounds::CheckResult Res(Offset);

  // CHECK LOWER BOUND
  if (Flags.CheckUnderflow) {
    auto [PrecedesLowerBound, WithinLowerBound] = compareValueToThreshold(
        State, SVB, Offset, SVB.makeZeroArrayIndex(), Comparison::LT);

    if (PrecedesLowerBound) {
      // The analyzer thinks that the offset may be invalid (negative)...
      if (Flags.OffsetObviouslyNonnegative) {
        // ...but the offset is obviously non-negative (clear array subscript
        // with an unsigned index), so we're in a buggy situation.

        // TODO: Currently the analyzer ignores many casts (e.g. signed ->
        // unsigned casts), so it can easily reach states where it will load a
        // signed (and negative) value from an unsigned variable. This sanity
        // check is a duct tape "solution" that silences most of the ugly false
        // positives that are caused by this buggy behavior. Note that this is
        // not a complete solution: this cannot silence reports where pointer
        // arithmetic complicates the picture and cannot ensure modeling of the
        // "unsigned index is positive with highest bit set" cases which are
        // "usurped" by the nonsense "unsigned index is negative" case.
        // For more information about this topic, see the umbrella ticket
        // https://github.com/llvm/llvm-project/issues/39492
        // TODO: Remove this hack once 'SymbolCast's are modeled properly.

        if (!WithinLowerBound) {
          // The state is completely nonsense -- let's just sink it!
          Res.IsCorruptedState = true;
          return Res;
        }
        // Otherwise continue on the 'WithinLowerBound' branch where the
        // unsigned index _is_ non-negative. Don't mention this assumption as a
        // note tag, because it would just confuse the users!
      } else {
        Res.MayUnderflow = true;

        if (!WithinLowerBound) {
          // ...and it cannot be valid (>= 0), so report an error.
          return Res;
        }
      }
    }

    // Actually update the state. The "if" only fails in the extremely unlikely
    // case when compareValueToThreshold returns {nullptr, nullptr} because
    // evalBinOpNN fails to evaluate the less-than operator.
    if (WithinLowerBound)
      State = WithinLowerBound;
  }

  // CHECK UPPER BOUND
  if (Extent) {
    Comparison CK = Flags.AlsoAcceptEquality ? Comparison::LE : Comparison::LT;
    auto [WithinUpperBound, ExceedsUpperBound] =
        compareValueToThreshold(State, SVB, Offset, *Extent, /*CmpKind=*/CK);

    if (ExceedsUpperBound) {
      // The offset may be invalid (>= Size)...
      Res.ExtentIfMayOverflow = Extent;

      if (!WithinUpperBound) {
        // ...and it cannot be within bounds.
        return Res;
      }
    }
    if (WithinUpperBound)
      State = WithinUpperBound;
  }

  Res.InBoundsState = State;
  return Res;
}
