//===- BoundsChecking.cpp - Bounds checking related APIs --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs for performing a bounds check (i.e. comparing a
//  symbolic Offset value to zero and a symbolic Extent value) and composing
//  descriptions that explain its results.
//
//  This is intended as a replacement for `ProgramState::assumeInBound` to
//  avoid its incorrect logic and compensate for deficiencies of other parts of
//  the analyzer.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BoundsChecking.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"

using llvm::formatv;

namespace clang {
namespace ento {

const ArraySubscriptExpr *
getAsCleanArraySubscriptExpr(const Expr *E, const CheckerContext &C) {
  const auto *ASE = dyn_cast<ArraySubscriptExpr>(E);
  if (!ASE)
    return nullptr;

  const MemRegion *SubscriptBaseReg = C.getSVal(ASE->getBase()).getAsRegion();
  if (!SubscriptBaseReg)
    return nullptr;

  // The base of the subscript expression is affected by pointer arithmetics,
  // so we want to report byte offsets instead of indices and we don't want to
  // activate the "index is unsigned -> cannot be negative" shortcut.
  if (isa<ElementRegion>(SubscriptBaseReg->StripCasts()))
    return nullptr;

  return ASE;
}

// NOTE: This function is the "heart" of this checker. It simplifies
// inequalities with transformations that are valid (and very elementary) in
// pure mathematics, but become invalid if we use them in C++ number model
// where the calculations may overflow.
// Due to the overflow issues I think it's impossible (or at least not
// practical) to integrate this kind of simplification into the resolution of
// arbitrary inequalities (i.e. the code of `evalBinOp`); but this function
// produces valid results when the calculations are handling memory offsets
// and every value is well below SIZE_MAX.
// TODO: This algorithm should be moved to a central location where it's
// available for other checkers that need to compare memory offsets.
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
      llvm::APSInt Constant = APSIntType(ExtentVal).convert(SIE->getRHS());
      switch (SIE->getOpcode()) {
      case BO_Mul:
        // The Constant should never be 0 here, becasue multiplication by zero
        // is simplified by the engine.
        if ((ExtentVal % Constant) != 0)
          return std::pair<NonLoc, nonloc::ConcreteInt>(Offset, Extent);
        else
          return getSimplifiedOffsets(nonloc::SymbolVal(SIE->getLHS()),
                                      SVB.makeIntVal(ExtentVal / Constant),
                                      SVB);
      case BO_Add:
        return getSimplifiedOffsets(nonloc::SymbolVal(SIE->getLHS()),
                                    SVB.makeIntVal(ExtentVal - Constant), SVB);
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

// Evaluate the comparison Value < Threshold with the help of the custom
// simplification algorithm defined for this checker. Return a pair of states,
// where the first one corresponds to "value below threshold" and the second
// corresponds to "value at or above threshold". Returns {nullptr, nullptr} in
// the case when the evaluation fails.
// If the optional argument CheckEquality is true, then use BO_EQ instead of
// the default BO_LT after consistently applying the same simplification steps.
static std::pair<ProgramStateRef, ProgramStateRef>
compareValueToThreshold(ProgramStateRef State, NonLoc Value, NonLoc Threshold,
                        SValBuilder &SVB, bool CheckEquality = false) {
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
    if (CheckEquality) {
      // negative_value == unsigned_threshold is always false
      return {nullptr, State};
    }
    // negative_value < unsigned_threshold is always true
    return {State, nullptr};
  }
  if (isUnsigned(SVB, Value) && isNegative(SVB, State, Threshold)) {
    // unsigned_value == negative_threshold and
    // unsigned_value < negative_threshold are both always false
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

  const BinaryOperatorKind OpKind = CheckEquality ? BO_EQ : BO_LT;
  auto BelowThreshold =
      SVB.evalBinOpNN(State, OpKind, Value, Threshold, SVB.getConditionType())
          .getAs<NonLoc>();

  if (BelowThreshold)
    return State->assume(*BelowThreshold);

  return {nullptr, nullptr};
}

std::string getRegionName(const MemSpaceRegion *Space,
                          const SubRegion *Region) {
  if (std::string RegName = Region->getDescriptiveName(); !RegName.empty())
    return RegName;

  // Field regions only have descriptive names when their parent has a
  // descriptive name; so we provide a fallback representation for them:
  if (const auto *FR = Region->getAs<FieldRegion>()) {
    if (StringRef Name = FR->getDecl()->getName(); !Name.empty())
      return formatv("the field '{0}'", Name);
    return "the unnamed field";
  }

  if (isa<AllocaRegion>(Region))
    return "the memory returned by 'alloca'";

  if (isa<SymbolicRegion>(Region) && isa<HeapSpaceRegion>(Space))
    return "the heap area";

  if (isa<StringRegion>(Region))
    return "the string literal";

  return "the region";
}

static std::optional<int64_t> getConcreteValue(NonLoc SV) {
  if (auto ConcreteVal = SV.getAs<nonloc::ConcreteInt>()) {
    return ConcreteVal->getValue()->tryExtValue();
  }
  return std::nullopt;
}

static std::optional<int64_t> getConcreteValue(std::optional<NonLoc> SV) {
  return SV ? getConcreteValue(*SV) : std::nullopt;
}

/// Try to divide `Val1` and `Val2` (in place) by `Divisor` and return true if
/// it can be performed (`Divisor` is nonzero and there is no remainder). The
/// values `Val1` and `Val2` may be nullopt and in that case the corresponding
/// division is considered to be successful.
static bool tryDividePair(std::optional<int64_t> &Val1,
                          std::optional<int64_t> &Val2, int64_t Divisor) {
  if (!Divisor)
    return false;
  const bool Val1HasRemainder = Val1 && *Val1 % Divisor;
  const bool Val2HasRemainder = Val2 && *Val2 % Divisor;
  if (Val1HasRemainder || Val2HasRemainder)
    return false;
  if (Val1)
    *Val1 /= Divisor;
  if (Val2)
    *Val2 /= Divisor;
  return true;
}

Messages getNonTaintMsgs(std::string RegName, SizeUnit SU, NonLoc Offset,
                         std::optional<NonLoc> Extent, BadOffsetKind Problem) {

  std::optional<int64_t> OffsetN = getConcreteValue(Offset);
  std::optional<int64_t> ExtentN = getConcreteValue(Extent);

  if (Problem == BadOffsetKind::Negative)
    ExtentN = std::nullopt;

  bool UseByteOffsets = !tryDividePair(OffsetN, ExtentN, SU.asCharUnits());
  const char *OffsetOrIndex = UseByteOffsets ? "byte offset" : "index";

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of ";
  if (OffsetN && !ExtentN && !UseByteOffsets) {
    // If the offset is reported as an index, then the report must mention the
    // element type (because it is not always clear from the code). It's more
    // natural to mention the element type later where the extent is described,
    // but if the extent is unknown/irrelevant, then the element type can be
    // inserted into the message at this point.
    Out << SU.asElementName(/*ForceBytes=*/false) << " in ";
  }
  Out << RegName << " at ";
  if (OffsetN) {
    if (Problem == BadOffsetKind::Negative)
      Out << "negative ";
    Out << OffsetOrIndex << " " << *OffsetN;
  } else {
    Out << asAdjective(Problem) << " " << OffsetOrIndex;
  }
  if (ExtentN) {
    Out << ", while it holds only ";
    if (*ExtentN != 1)
      Out << *ExtentN;
    else
      Out << "a single";

    Out << ' ' << SU.asElementName(/*ForceBytes=*/UseByteOffsets);

    if (*ExtentN > 1)
      Out << "s";
  }

  return {formatv("Out of bound access to memory {0} {1}",
                  asPreposition(Problem), RegName),
          std::string(Buf)};
}

Messages getTaintMsgs(std::string RegName, const char *OffsetName,
                      bool AlsoMentionUnderflow) {
  return {formatv("Potential out of bound access to {0} with tainted {1}",
                  RegName, OffsetName),
          formatv("Access of {0} with a tainted {1} that may be {2}too large",
                  RegName, OffsetName,
                  AlsoMentionUnderflow ? "negative or " : "")};
}

std::string BoundsCheckResult::getMessage(PathSensitiveBugReport &BR,
                                          StringRef RegName,
                                          SizeUnit SU) const {
  bool ShouldReportNonNegative = AssumedNonNegative;
  if (!providesInformationAboutInteresting(Offset, BR)) {
    if (AssumedUpperBound && providesInformationAboutInteresting(*Extent, BR)) {
      // Even if the byte offset isn't interesting (e.g. it's a constant value),
      // the assumption can still be interesting if it provides information
      // about an interesting symbolic upper bound.
      ShouldReportNonNegative = false;
    } else {
      // We don't have anything interesting, don't report the assumption.
      return "";
    }
  }

  std::optional<int64_t> OffsetN = getConcreteValue(Offset);
  std::optional<int64_t> ExtentN = getConcreteValue(Extent);

  const bool UseIndex =
      !SU.isBytes() && tryDividePair(OffsetN, ExtentN, SU.asCharUnits());

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Assuming ";
  if (UseIndex) {
    Out << "index ";
    if (OffsetN)
      Out << "'" << OffsetN << "' ";
  } else if (Extent) {
    Out << "byte offset ";
    if (OffsetN)
      Out << "'" << OffsetN << "' ";
  } else {
    Out << "offset ";
  }

  Out << "is";
  if (ShouldReportNonNegative) {
    Out << " non-negative";
  }
  if (Extent) {
    if (ShouldReportNonNegative)
      Out << " and";
    Out << " less than ";
    if (ExtentN)
      Out << *ExtentN << ", ";
    Out << SU.asExtentDesc(/*ForceBytes=*/!UseIndex) << ' ' << RegName;
  }
  return std::string(Out.str());
}

bool BoundsCheckResult::providesInformationAboutInteresting(
    SymbolRef Sym, PathSensitiveBugReport &BR) {
  if (!Sym)
    return false;
  for (SymbolRef PartSym : Sym->symbols()) {
    // The interestingess mark may appear on any layer as we're stripping off
    // the SymIntExpr, UnarySymExpr etc. layers...
    if (BR.isInteresting(PartSym))
      return true;
    // ...but if both sides of the expression are symbolic, then there is no
    // practical algorithm to produce separate constraints for the two
    // operands (from the single combined result).
    if (isa<SymSymExpr>(PartSym))
      return false;
  }
  return false;
}

BoundsCheckResult checkBounds(ProgramStateRef State, SValBuilder &SVB,
                              NonLoc Offset, std::optional<NonLoc> Extent,
                              CheckFlags Flags) {
  BoundsCheckResult Res(Offset, Extent);

  // CHECK LOWER BOUND
  if (Flags.CheckUnderflow) {
    auto [PrecedesLowerBound, WithinLowerBound] =
        compareValueToThreshold(State, Offset, SVB.makeZeroArrayIndex(), SVB);

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
          Res.finalize(BoundsCheckResult::Kind::Paradox, PrecedesLowerBound);
          return Res;
        }
        // Otherwise continue on the 'WithinLowerBound' branch where the
        // unsigned index _is_ non-negative. Don't mention this assumption as a
        // note tag, because it would just confuse the users!
      } else {
        if (!WithinLowerBound) {
          // ...and it cannot be valid (>= 0), so report an error.
          Res.finalize(BoundsCheckResult::Kind::Underflow, PrecedesLowerBound);
          return Res;
        }
        // ...but it can be valid as well, so the checker will (optimistically)
        // assume that it's valid and mention this in the note tag.
        Res.recordNonNegativeAssumption();
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
    auto [WithinUpperBound, ExceedsUpperBound] =
        compareValueToThreshold(State, Offset, *Extent, SVB);

    if (ExceedsUpperBound) {
      // The offset may be invalid (>= Size)...
      if (!WithinUpperBound) {
        // ...and it cannot be within bounds, so report an error, unless we can
        // definitely determine that this is an idiomatic `&array[size]`
        // expression that calculates the past-the-end pointer.
        if (Flags.AcceptPastTheEnd) {
          auto [EqualsToThreshold, NotEqualToThreshold] =
              compareValueToThreshold(State, Offset, *Extent, SVB,
                                      /*CheckEquality=*/true);
          if (EqualsToThreshold && !NotEqualToThreshold) {
            Res.finalize(BoundsCheckResult::Kind::Valid, State);
            return Res;
          }
        }

        Res.finalize(BoundsCheckResult::Kind::Overflow, ExceedsUpperBound);
        return Res;
      }
      // ...and it can be valid as well...
      if (taint::isTainted(State, Offset)) {
        // ...but it's tainted, so report an error.
        Res.finalize(BoundsCheckResult::Kind::TaintBug, State);
        return Res;
      }
      // ...and it isn't tainted, so the checker will (optimistically) assume
      // that the offset is in bounds and mention this in the note tag.
      Res.recordUpperBoundAssumption();
    }

    // Actually update the state. The "if" only fails in the extremely unlikely
    // case when compareValueToThreshold returns {nullptr, nullptr} because
    // evalBinOpNN fails to evaluate the less-than operator.
    if (WithinUpperBound)
      State = WithinUpperBound;
  }
  Res.finalize(BoundsCheckResult::Kind::Valid, State);
  return Res;
}

} // namespace ento
} // namespace clang
