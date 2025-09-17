//== ArrayBoundChecker.cpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines security.ArrayBound, which is a path-sensitive checker
// that looks for out of bounds access of memory regions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CharUnits.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang;
using namespace ento;
using namespace taint;
using llvm::formatv;

namespace {
/// If `E` is an array subscript expression with a base that is "clean" (= not
/// modified by pointer arithmetic = the beginning of a memory region), return
/// it as a pointer to ArraySubscriptExpr; otherwise return nullptr.
/// This helper function is used by two separate heuristics that are only valid
/// in these "clean" cases.
static const ArraySubscriptExpr *
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

/// If `E` is a "clean" array subscript expression, return the type of the
/// accessed element; otherwise return std::nullopt because that's the best (or
/// least bad) option for the diagnostic generation that relies on this.
static std::optional<QualType> determineElementType(const Expr *E,
                                                    const CheckerContext &C) {
  const auto *ASE = getAsCleanArraySubscriptExpr(E, C);
  if (!ASE)
    return std::nullopt;

  return ASE->getType();
}

static std::optional<int64_t>
determineElementSize(const std::optional<QualType> T, const CheckerContext &C) {
  if (!T)
    return std::nullopt;
  return C.getASTContext().getTypeSizeInChars(*T).getQuantity();
}

class StateUpdateReporter {
  const MemSpaceRegion *Space;
  const SubRegion *Reg;
  const NonLoc ByteOffsetVal;
  const std::optional<QualType> ElementType;
  const std::optional<int64_t> ElementSize;
  bool AssumedNonNegative = false;
  std::optional<NonLoc> AssumedUpperBound = std::nullopt;

public:
  StateUpdateReporter(const SubRegion *R, NonLoc ByteOffsVal, const Expr *E,
                      CheckerContext &C)
      : Space(R->getMemorySpace(C.getState())), Reg(R),
        ByteOffsetVal(ByteOffsVal), ElementType(determineElementType(E, C)),
        ElementSize(determineElementSize(ElementType, C)) {}

  void recordNonNegativeAssumption() { AssumedNonNegative = true; }
  void recordUpperBoundAssumption(NonLoc UpperBoundVal) {
    AssumedUpperBound = UpperBoundVal;
  }

  bool assumedNonNegative() { return AssumedNonNegative; }

  const NoteTag *createNoteTag(CheckerContext &C) const;

private:
  std::string getMessage(PathSensitiveBugReport &BR) const;
};

struct Messages {
  std::string Short, Full;
};

enum class BadOffsetKind { Negative, Overflowing, Indeterminate };

constexpr llvm::StringLiteral Adjectives[] = {"a negative", "an overflowing",
                                              "a negative or overflowing"};
static StringRef asAdjective(BadOffsetKind Problem) {
  return Adjectives[static_cast<int>(Problem)];
}

constexpr llvm::StringLiteral Prepositions[] = {"preceding", "after the end of",
                                                "around"};
static StringRef asPreposition(BadOffsetKind Problem) {
  return Prepositions[static_cast<int>(Problem)];
}

// NOTE: The `ArraySubscriptExpr` and `UnaryOperator` callbacks are `PostStmt`
// instead of `PreStmt` because the current implementation passes the whole
// expression to `CheckerContext::getSVal()` which only works after the
// symbolic evaluation of the expression. (To turn them into `PreStmt`
// callbacks, we'd need to duplicate the logic that evaluates these
// expressions.) The `MemberExpr` callback would work as `PreStmt` but it's
// defined as `PostStmt` for the sake of consistency with the other callbacks.
class ArrayBoundChecker : public Checker<check::PostStmt<ArraySubscriptExpr>,
                                         check::PostStmt<UnaryOperator>,
                                         check::PostStmt<MemberExpr>> {
  BugType BT{this, "Out-of-bound access"};
  BugType TaintBT{this, "Out-of-bound access", categories::TaintedData};

  enum class ConstantArrayIndexResult {
    Done,   //< Could model the index access based on its type
    Unknown //< Could not model the array access based on its type
  };

  // `ConstantArrayType`s have a constant size, so use it to check the access.
  ConstantArrayIndexResult
  performCheckArrayTypeIndex(const ArraySubscriptExpr *E,
                             CheckerContext &C) const;

  void performCheck(const Expr *E, CheckerContext &C) const;

  void reportOOB(CheckerContext &C, ProgramStateRef ErrorState, Messages Msgs,
                 NonLoc Offset, std::optional<NonLoc> Extent,
                 bool IsTaintBug = false) const;

  void warnFlexibleArrayAccess(CheckerContext &C, ProgramStateRef State,
                               const ArraySubscriptExpr *E, StringRef Name,
                               NonLoc Index, nonloc::ConcreteInt ArraySize,
                               QualType ArrayType) const;

  static void markPartsInteresting(PathSensitiveBugReport &BR,
                                   ProgramStateRef ErrorState, NonLoc Val,
                                   bool MarkTaint);

  static bool isFromCtypeMacro(const Expr *E, ASTContext &AC);

  static bool isOffsetObviouslyNonnegative(const Expr *E, CheckerContext &C);

  static bool isIdiomaticPastTheEndPtr(const Expr *E, ProgramStateRef State,
                                       NonLoc Offset, NonLoc Limit,
                                       CheckerContext &C);
  static bool isInAddressOf(const Stmt *S, ASTContext &AC);

public:
  bool EnableFakeFlexibleArrayWarn{false};

  void checkPostStmt(const ArraySubscriptExpr *E, CheckerContext &C) const {
    if (performCheckArrayTypeIndex(E, C) == ConstantArrayIndexResult::Unknown) {
      performCheck(E, C);
    }
  }
  void checkPostStmt(const UnaryOperator *E, CheckerContext &C) const {
    if (E->getOpcode() == UO_Deref)
      performCheck(E, C);
  }
  void checkPostStmt(const MemberExpr *E, CheckerContext &C) const {
    if (E->isArrow())
      performCheck(E->getBase(), C);
  }
};

} // anonymous namespace

/// For a given Location that can be represented as a symbolic expression
/// Arr[Idx] (or perhaps Arr[Idx1][Idx2] etc.), return the parent memory block
/// Arr and the distance of Location from the beginning of Arr (expressed in a
/// NonLoc that specifies the number of CharUnits). Returns nullopt when these
/// cannot be determined.
static std::optional<std::pair<const SubRegion *, NonLoc>>
computeOffset(ProgramStateRef State, SValBuilder &SVB, SVal Location) {
  QualType T = SVB.getArrayIndexType();
  auto EvalBinOp = [&SVB, State, T](BinaryOperatorKind Op, NonLoc L, NonLoc R) {
    // We will use this utility to add and multiply values.
    return SVB.evalBinOpNN(State, Op, L, R, T).getAs<NonLoc>();
  };

  const SubRegion *OwnerRegion = nullptr;
  std::optional<NonLoc> Offset = SVB.makeZeroArrayIndex();

  const ElementRegion *CurRegion =
      dyn_cast_or_null<ElementRegion>(Location.getAsRegion());

  while (CurRegion) {
    const auto Index = CurRegion->getIndex().getAs<NonLoc>();
    if (!Index)
      return std::nullopt;

    QualType ElemType = CurRegion->getElementType();

    // FIXME: The following early return was presumably added to safeguard the
    // getTypeSizeInChars() call (which doesn't accept an incomplete type), but
    // it seems that `ElemType` cannot be incomplete at this point.
    if (ElemType->isIncompleteType())
      return std::nullopt;

    // Calculate Delta = Index * sizeof(ElemType).
    NonLoc Size = SVB.makeArrayIndex(
        SVB.getContext().getTypeSizeInChars(ElemType).getQuantity());
    auto Delta = EvalBinOp(BO_Mul, *Index, Size);
    if (!Delta)
      return std::nullopt;

    // Perform Offset += Delta.
    Offset = EvalBinOp(BO_Add, *Offset, *Delta);
    if (!Offset)
      return std::nullopt;

    OwnerRegion = CurRegion->getSuperRegion()->getAs<SubRegion>();
    // When this is just another ElementRegion layer, we need to continue the
    // offset calculations:
    CurRegion = dyn_cast_or_null<ElementRegion>(OwnerRegion);
  }

  if (OwnerRegion)
    return std::make_pair(OwnerRegion, *Offset);

  return std::nullopt;
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
getSimplifiedOffsets(NonLoc offset, nonloc::ConcreteInt extent,
                     SValBuilder &svalBuilder) {
  const llvm::APSInt &extentVal = extent.getValue();
  std::optional<nonloc::SymbolVal> SymVal = offset.getAs<nonloc::SymbolVal>();
  if (SymVal && SymVal->isExpression()) {
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SymVal->getSymbol())) {
      llvm::APSInt constant = APSIntType(extentVal).convert(SIE->getRHS());
      switch (SIE->getOpcode()) {
      case BO_Mul:
        // The constant should never be 0 here, becasue multiplication by zero
        // is simplified by the engine.
        if ((extentVal % constant) != 0)
          return std::pair<NonLoc, nonloc::ConcreteInt>(offset, extent);
        else
          return getSimplifiedOffsets(
              nonloc::SymbolVal(SIE->getLHS()),
              svalBuilder.makeIntVal(extentVal / constant), svalBuilder);
      case BO_Add:
        return getSimplifiedOffsets(
            nonloc::SymbolVal(SIE->getLHS()),
            svalBuilder.makeIntVal(extentVal - constant), svalBuilder);
      default:
        break;
      }
    }
  }

  return std::pair<NonLoc, nonloc::ConcreteInt>(offset, extent);
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

static std::string getRegionName(const MemSpaceRegion *Space,
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

static Messages getNonTaintMsgs(const ASTContext &ACtx,
                                const MemSpaceRegion *Space,
                                const SubRegion *Region, NonLoc Offset,
                                std::optional<NonLoc> Extent, SVal Location,
                                BadOffsetKind Problem) {
  std::string RegName = getRegionName(Space, Region);
  const auto *EReg = Location.getAsRegion()->getAs<ElementRegion>();
  assert(EReg && "this checker only handles element access");
  QualType ElemType = EReg->getElementType();

  std::optional<int64_t> OffsetN = getConcreteValue(Offset);
  std::optional<int64_t> ExtentN = getConcreteValue(Extent);

  int64_t ElemSize = ACtx.getTypeSizeInChars(ElemType).getQuantity();

  bool UseByteOffsets = !tryDividePair(OffsetN, ExtentN, ElemSize);
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
    Out << "'" << ElemType.getAsString() << "' element in ";
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
    if (UseByteOffsets)
      Out << " byte";
    else
      Out << " '" << ElemType.getAsString() << "' element";

    if (*ExtentN > 1)
      Out << "s";
  }

  return {formatv("Out of bound access to memory {0} {1}",
                  asPreposition(Problem), RegName),
          std::string(Buf)};
}

static Messages getTaintMsgs(StringRef RegName, const char *OffsetName,
                             bool AlsoMentionUnderflow) {
  return {formatv("Potential out of bound access to {0} with tainted {1}",
                  RegName, OffsetName),
          formatv("Access of {0} with a tainted {1} that may be {2}too large",
                  RegName, OffsetName,
                  AlsoMentionUnderflow ? "negative or " : "")};
}

/// Return true if information about the value of `Sym` can put constraints
/// on some symbol which is interesting within the bug report `BR`.
/// In particular, this returns true when `Sym` is interesting within `BR`;
/// but it also returns true if `Sym` is an expression that contains integer
/// constants and a single symbolic operand which is interesting (in `BR`).
/// We need to use this instead of plain `BR.isInteresting()` because if we
/// are analyzing code like
///   int array[10];
///   int f(int arg) {
///     return array[arg] && array[arg + 10];
///   }
/// then the byte offsets are `arg * 4` and `(arg + 10) * 4`, which are not
/// sub-expressions of each other (but `getSimplifiedOffsets` is smart enough
/// to detect this out of bounds access).
static bool providesInformationAboutInteresting(SymbolRef Sym,
                                                PathSensitiveBugReport &BR) {
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

static bool providesInformationAboutInteresting(SVal SV,
                                                PathSensitiveBugReport &BR) {
  return providesInformationAboutInteresting(SV.getAsSymbol(), BR);
}

const NoteTag *StateUpdateReporter::createNoteTag(CheckerContext &C) const {
  // Don't create a note tag if we didn't assume anything:
  if (!AssumedNonNegative && !AssumedUpperBound)
    return nullptr;

  return C.getNoteTag([*this](PathSensitiveBugReport &BR) -> std::string {
    return getMessage(BR);
  });
}

std::string StateUpdateReporter::getMessage(PathSensitiveBugReport &BR) const {
  bool ShouldReportNonNegative = AssumedNonNegative;
  if (!providesInformationAboutInteresting(ByteOffsetVal, BR)) {
    if (AssumedUpperBound &&
        providesInformationAboutInteresting(*AssumedUpperBound, BR)) {
      // Even if the byte offset isn't interesting (e.g. it's a constant value),
      // the assumption can still be interesting if it provides information
      // about an interesting symbolic upper bound.
      ShouldReportNonNegative = false;
    } else {
      // We don't have anything interesting, don't report the assumption.
      return "";
    }
  }

  std::optional<int64_t> OffsetN = getConcreteValue(ByteOffsetVal);
  std::optional<int64_t> ExtentN = getConcreteValue(AssumedUpperBound);

  const bool UseIndex =
      ElementSize && tryDividePair(OffsetN, ExtentN, *ElementSize);

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Assuming ";
  if (UseIndex) {
    Out << "index ";
    if (OffsetN)
      Out << "'" << OffsetN << "' ";
  } else if (AssumedUpperBound) {
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
  if (AssumedUpperBound) {
    if (ShouldReportNonNegative)
      Out << " and";
    Out << " less than ";
    if (ExtentN)
      Out << *ExtentN << ", ";
    if (UseIndex && ElementType)
      Out << "the number of '" << ElementType->getAsString()
          << "' elements in ";
    else
      Out << "the extent of ";
    Out << getRegionName(Space, Reg);
  }
  return std::string(Out.str());
}

// If the base expression of `expr` refers to a `ConstantArrayType`,
// return the element type and the array size.
// Note that "real" Flexible Array Members are `IncompleteArrayType`.
// For them, this function returns `std::nullopt`.
static std::optional<std::pair<QualType, nonloc::ConcreteInt>>
getArrayTypeInfo(SValBuilder &svb, ArraySubscriptExpr const *expr) {
  auto const *arrayBaseExpr = expr->getBase()->IgnoreParenImpCasts();
  auto const arrayQualType = arrayBaseExpr->getType().getCanonicalType();
  if (arrayQualType.isNull() || !arrayQualType->isConstantArrayType()) {
    return std::nullopt;
  }
  auto *constArrayType =
      dyn_cast<ConstantArrayType>(arrayQualType->getAsArrayTypeUnsafe());
  if (!constArrayType) {
    return std::nullopt;
  }
  return std::make_pair(
      constArrayType->getElementType(),
      // Note that an array size is technically unsigned, but
      // `compareValueToThreshold` (via `getSimplifiedOffsets`)
      // will do some arithmetics that could overflow and cause FN. For instance
      // ```
      // int array[20];
      // array[unsigned_index + 21];
      // ```
      // `unsigned_index + 21 < 20` is turned into `unsigned_index < 20 - 21`,
      // and if 20 is unsigned, that will overflow to the biggest possible array
      // which is always trivially true. We obviously do not want that, so we
      // need to treat the size as signed.
      svb.makeIntVal(llvm::APSInt{constArrayType->getSize(), false}));
}

static Messages getNegativeIndexMessage(StringRef Name,
                                        nonloc::ConcreteInt ArraySize,
                                        NonLoc Index) {
  auto const ArraySizeVal = ArraySize.getValue()->getZExtValue();
  std::string const IndexStr = [&]() -> std::string {
    if (auto ConcreteIndex = getConcreteValue(Index);
        ConcreteIndex.has_value()) {
      return formatv(" {0}", ConcreteIndex);
    }
    return "";
  }();

  return {formatv("Out of bound access to {0} at a negative index", Name),
          formatv("Access of {0} containing {1} elements at negative index{2}",
                  Name, ArraySizeVal, IndexStr)};
}

static std::string truncateWithEllipsis(StringRef str, size_t maxLength) {
  if (str.size() <= maxLength)
    return str.str();

  return (str.substr(0, maxLength - 3) + "...").str();
}

static Messages getOOBIndexMessage(StringRef Name, NonLoc Index,
                                   nonloc::ConcreteInt ArraySize,
                                   QualType ElemType,
                                   bool AlsoMentionUnderflow) {
  std::optional<int64_t> IndexN = getConcreteValue(Index);
  int64_t ExtentN = ArraySize.getValue()->getZExtValue();

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of " << Name << " at ";
  if (AlsoMentionUnderflow) {
    Out << "a negative or overflowing index";
  } else if (IndexN.has_value()) {
    Out << "index " << *IndexN;
  } else {
    Out << "an overflowing index";
  }

  const auto ElemTypeStr = truncateWithEllipsis(ElemType.getAsString(), 20);

  Out << ", while it holds only ";
  if (ExtentN != 1)
    Out << ExtentN << " '" << ElemTypeStr << "' elements";
  else
    Out << "a single " << "'" << ElemTypeStr << "' element";

  return {formatv("Out of bound access to memory {0} {1}",
                  AlsoMentionUnderflow ? "around" : "after the end of", Name),
          std::string(Buf)};
}

// "True" flexible array members do not specify any size (`int elems[];`)
// However, some projects use "fake flexible arrays" (aka "struct hack"), where
// they specify a size of 0 or 1 to work around a compiler limitation.
// "True" flexible array members are `IncompleteArrayType` and will be skipped
// by `performCheckArrayTypeIndex`. We need an heuristic to identify "fake"
// ones.
static bool isFakeFlexibleArrays(const ArraySubscriptExpr *E) {
  auto getFieldDecl = [](ArraySubscriptExpr const *array) -> FieldDecl * {
    const Expr *BaseExpr = array->getBase()->IgnoreParenImpCasts();
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(BaseExpr)) {
      return dyn_cast<FieldDecl>(ME->getMemberDecl());
    }
    return nullptr;
  };
  auto const isLastField = [](RecordDecl const *Parent,
                              FieldDecl const *Field) {
    const FieldDecl *LastField = nullptr;
    for (const FieldDecl *F : Parent->fields()) {
      LastField = F;
    }

    return (LastField == Field);
  };
  // We expect placeholder constant arrays to have size 0 or 1.
  auto maybeConstArrayPlaceholder = [](QualType Type) {
    const ConstantArrayType *CAT =
        dyn_cast<ConstantArrayType>(Type->getAsArrayTypeUnsafe());
    return CAT && CAT->getSize().getZExtValue() <= 1;
  };

  if (auto const *Field = getFieldDecl(E)) {
    if (!maybeConstArrayPlaceholder(Field->getType()))
      return false;

    const RecordDecl *Parent = Field->getParent();
    return Parent && (Parent->isUnion() || isLastField(Parent, Field));
  }

  return false;
}

// Generate a representation of `Expr` suitable for diagnosis.
SmallString<128> ExprReprForDiagnosis(ArraySubscriptExpr const *E,
                                      CheckerContext &C) {
  SmallString<128> Buf;
  llvm::raw_svector_ostream Out(Buf);

  auto const *Base = E->getBase()->IgnoreParenImpCasts();
  switch (Base->getStmtClass()) {
  case Stmt::MemberExprClass:
    Out << "the field '";
    Out << dyn_cast<MemberExpr>(Base)->getMemberDecl()->getName().str();
    Out << '\'';
    break;
  case Stmt::ArraySubscriptExprClass:
    Out << "the subarray '";
    Base->printPretty(Out, nullptr, {C.getLangOpts()});
    Out << '\'';
    break;
  default:
    Out << '\'';
    Base->printPretty(Out, nullptr, {C.getLangOpts()});
    Out << '\'';
  }

  return Buf;
}

class StateIndexUpdateReporter {
  std::string Repr;
  QualType ElementType;
  NonLoc Index;
  nonloc::ConcreteInt ArraySize;
  bool AssumedNonNegative = false;
  bool AssumedInBounds = false;

  std::string getMessage(PathSensitiveBugReport &BR) const {
    SmallString<256> Buf;
    if (providesInformationAboutInteresting(Index, BR)) {
      llvm::raw_svector_ostream Out{Buf};
      Out << "Assuming index is";
      if (AssumedNonNegative)
        Out << " non-negative";
      if (AssumedInBounds) {
        if (AssumedNonNegative)
          Out << " and";
        Out << " less than " << ArraySize.getValue()->getZExtValue() << ", ";
        Out << "the number of '" << ElementType.getAsString()
            << "' elements in ";
        Out << Repr;
      }
    }
    return std::string{Buf.str()};
  }

public:
  StateIndexUpdateReporter(StringRef Repr, QualType ElementType, NonLoc Index,
                           nonloc::ConcreteInt ArraySize)
      : Repr(Repr), ElementType{ElementType}, Index{Index},
        ArraySize{ArraySize} {}

  void recordNonNegativeAssumption() { AssumedNonNegative = true; }

  void recordInBoundsAssumption() { AssumedInBounds = true; }

  const NoteTag *createNoteTag(CheckerContext &C) const {
    // Don't create a note tag if we didn't assume anything:
    if (!AssumedNonNegative && !AssumedInBounds) {
      return nullptr;
    }

    return C.getNoteTag(
        [*this](PathSensitiveBugReport &BR) { return getMessage(BR); });
  }
};

// If the array is a `ConstantArrayType`, check the axis size.
// It returns `ConstantArrayIndexResult::Unknown` if it could not reason about
// the array access, deferring to the regular check based on the region.
auto ArrayBoundChecker::performCheckArrayTypeIndex(const ArraySubscriptExpr *E,
                                                   CheckerContext &C) const
    -> ConstantArrayIndexResult {
  auto State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  auto const ArrayInfo = getArrayTypeInfo(SVB, E);
  auto const Index =
      SVB.simplifySVal(State, C.getSVal(E->getIdx())).getAs<NonLoc>();
  if (!ArrayInfo || !Index)
    return ConstantArrayIndexResult::Unknown;

  auto const &[ArrayType, ArraySize] = *ArrayInfo;

  auto const ExprAsStr = ExprReprForDiagnosis(E, C);
  bool const IsFakeFAM = isFakeFlexibleArrays(E);

  StateIndexUpdateReporter SUR(ExprAsStr, ArrayType, *Index, ArraySize);

  // Is the index negative?
  auto [NegativeIndexState, NonNegativeIndexState] =
      compareValueToThreshold(State, *Index, SVB.makeZeroArrayIndex(), SVB);
  bool const AlsoMentionUnderflow = (NegativeIndexState != nullptr);

  // Negative is possible
  if (NegativeIndexState) {
    // But it can't be
    if (E->getIdx()->getType()->isUnsignedIntegerOrEnumerationType()) {
      // And positive isn't possible
      if (!NonNegativeIndexState) {
        // The state is broken
        return ConstantArrayIndexResult::Done;
      }
      // As in `performCheck`, we add no assumptions about the index
    } else if (!NonNegativeIndexState) {
      // Positive is not possible, this is a bug
      Messages Msgs = getNegativeIndexMessage(ExprAsStr, ArraySize, *Index);
      reportOOB(C, NegativeIndexState, Msgs, *Index, ArraySize);
      return ConstantArrayIndexResult::Done;

    } else {
      // Both negative and positive are possible, assume positive
      SUR.recordNonNegativeAssumption();
    }
  } else if (!NonNegativeIndexState) {
    // Broken state
    return ConstantArrayIndexResult::Done;
  }

  // The index is greater than 0, is it within bounds?
  auto [WithinUpperBound, OutOfBounds] =
      compareValueToThreshold(NonNegativeIndexState, *Index, ArraySize, SVB);
  if (!WithinUpperBound && !OutOfBounds) {
    // Invalid state
    return ConstantArrayIndexResult::Done;
  }

  if (!WithinUpperBound) {
    if (isIdiomaticPastTheEndPtr(E, OutOfBounds, *Index, ArraySize, C)) {
      C.addTransition(OutOfBounds, SUR.createNoteTag(C));
      return ConstantArrayIndexResult::Done;
    }
    // Silence for fake flexible arrays unless explicitly enabled
    if (!IsFakeFAM) {
      Messages Msgs = getOOBIndexMessage(ExprAsStr, *Index, ArraySize,
                                         ArrayType, AlsoMentionUnderflow);
      reportOOB(C, OutOfBounds, Msgs, *Index, ArraySize);
    } else if (EnableFakeFlexibleArrayWarn) {
      warnFlexibleArrayAccess(C, OutOfBounds, E, ExprAsStr, *Index, ArraySize,
                              ArrayType);
    }
    return ConstantArrayIndexResult::Done;
  }

  // The access might be within range, but it may be tainted
  if (OutOfBounds && isTainted(OutOfBounds, *Index)) {
    Messages Msgs = getTaintMsgs(ExprAsStr, "index", AlsoMentionUnderflow);
    reportOOB(C, OutOfBounds, Msgs, *Index, ArraySize,
              /*IsTaintBug=*/true);
  }

  // When "Flexible Array Members" are involved, assume only non-negative
  // even if we want the warning for OOB FAM access.
  if (!IsFakeFAM) {
    if (WithinUpperBound != NonNegativeIndexState)
      SUR.recordInBoundsAssumption();
    C.addTransition(WithinUpperBound, SUR.createNoteTag(C));
  } else
    C.addTransition(NonNegativeIndexState, SUR.createNoteTag(C));

  return ConstantArrayIndexResult::Done;
}

void ArrayBoundChecker::performCheck(const Expr *E, CheckerContext &C) const {
  const SVal Location = C.getSVal(E);

  // The header ctype.h (from e.g. glibc) implements the isXXXXX() macros as
  //   #define isXXXXX(arg) (LOOKUP_TABLE[arg] & BITMASK_FOR_XXXXX)
  // and incomplete analysis of these leads to false positives. As even
  // accurate reports would be confusing for the users, just disable reports
  // from these macros:
  if (isFromCtypeMacro(E, C.getASTContext()))
    return;

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  const std::optional<std::pair<const SubRegion *, NonLoc>> &RawOffset =
      computeOffset(State, SVB, Location);

  if (!RawOffset)
    return;

  auto [Reg, ByteOffset] = *RawOffset;

  // The state updates will be reported as a single note tag, which will be
  // composed by this helper class.
  StateUpdateReporter SUR(Reg, ByteOffset, E, C);

  // CHECK LOWER BOUND
  const MemSpaceRegion *Space = Reg->getMemorySpace(State);
  if (!(isa<SymbolicRegion>(Reg) && isa<UnknownSpaceRegion>(Space))) {
    // A symbolic region in unknown space represents an unknown pointer that
    // may point into the middle of an array, so we don't look for underflows.
    // Both conditions are significant because we want to check underflows in
    // symbolic regions on the heap (which may be introduced by checkers like
    // MallocChecker that call SValBuilder::getConjuredHeapSymbolVal()) and
    // non-symbolic regions (e.g. a field subregion of a symbolic region) in
    // unknown space.
    auto [PrecedesLowerBound, WithinLowerBound] = compareValueToThreshold(
        State, ByteOffset, SVB.makeZeroArrayIndex(), SVB);

    if (PrecedesLowerBound) {
      // The analyzer thinks that the offset may be invalid (negative)...

      if (isOffsetObviouslyNonnegative(E, C)) {
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
          C.addSink();
          return;
        }
        // Otherwise continue on the 'WithinLowerBound' branch where the
        // unsigned index _is_ non-negative. Don't mention this assumption as a
        // note tag, because it would just confuse the users!
      } else {
        if (!WithinLowerBound) {
          // ...and it cannot be valid (>= 0), so report an error.
          Messages Msgs = getNonTaintMsgs(C.getASTContext(), Space, Reg,
                                          ByteOffset, /*Extent=*/std::nullopt,
                                          Location, BadOffsetKind::Negative);
          reportOOB(C, PrecedesLowerBound, Msgs, ByteOffset, std::nullopt);
          return;
        }
        // ...but it can be valid as well, so the checker will (optimistically)
        // assume that it's valid and mention this in the note tag.
        SUR.recordNonNegativeAssumption();
      }
    }

    // Actually update the state. The "if" only fails in the extremely unlikely
    // case when compareValueToThreshold returns {nullptr, nullptr} because
    // evalBinOpNN fails to evaluate the less-than operator.
    if (WithinLowerBound)
      State = WithinLowerBound;
  }

  // CHECK UPPER BOUND
  DefinedOrUnknownSVal Size = getDynamicExtent(State, Reg, SVB);
  if (auto KnownSize = Size.getAs<NonLoc>()) {
    // In a situation where both underflow and overflow are possible (but the
    // index is either tainted or known to be invalid), the logic of this
    // checker will first assume that the offset is non-negative, and then
    // (with this additional assumption) it will detect an overflow error.
    // In this situation the warning message should mention both possibilities.
    bool AlsoMentionUnderflow = SUR.assumedNonNegative();

    auto [WithinUpperBound, ExceedsUpperBound] =
        compareValueToThreshold(State, ByteOffset, *KnownSize, SVB);

    if (ExceedsUpperBound) {
      // The offset may be invalid (>= Size)...
      if (!WithinUpperBound) {
        // ...and it cannot be within bounds, so report an error, unless we can
        // definitely determine that this is an idiomatic `&array[size]`
        // expression that calculates the past-the-end pointer.
        if (isIdiomaticPastTheEndPtr(E, ExceedsUpperBound, ByteOffset,
                                     *KnownSize, C)) {
          C.addTransition(ExceedsUpperBound, SUR.createNoteTag(C));
          return;
        }

        BadOffsetKind Problem = AlsoMentionUnderflow
                                    ? BadOffsetKind::Indeterminate
                                    : BadOffsetKind::Overflowing;
        Messages Msgs =
            getNonTaintMsgs(C.getASTContext(), Space, Reg, ByteOffset,
                            *KnownSize, Location, Problem);
        reportOOB(C, ExceedsUpperBound, Msgs, ByteOffset, KnownSize);
        return;
      }
      // ...and it can be valid as well...
      if (isTainted(State, ByteOffset)) {
        // ...but it's tainted, so report an error.

        // Diagnostic detail: saying "tainted offset" is always correct, but
        // the common case is that 'idx' is tainted in 'arr[idx]' and then it's
        // nicer to say "tainted index".
        const char *OffsetName = "offset";
        if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
          if (isTainted(State, ASE->getIdx(), C.getLocationContext()))
            OffsetName = "index";

        Messages Msgs = getTaintMsgs(getRegionName(Space, Reg), OffsetName,
                                     AlsoMentionUnderflow);
        reportOOB(C, ExceedsUpperBound, Msgs, ByteOffset, KnownSize,
                  /*IsTaintBug=*/true);
        return;
      }
      // ...and it isn't tainted, so the checker will (optimistically) assume
      // that the offset is in bounds and mention this in the note tag.
      SUR.recordUpperBoundAssumption(*KnownSize);
    }

    // Actually update the state. The "if" only fails in the extremely unlikely
    // case when compareValueToThreshold returns {nullptr, nullptr} because
    // evalBinOpNN fails to evaluate the less-than operator.
    if (WithinUpperBound)
      State = WithinUpperBound;
  }

  // Add a transition, reporting the state updates that we accumulated.
  C.addTransition(State, SUR.createNoteTag(C));
}

void ArrayBoundChecker::markPartsInteresting(PathSensitiveBugReport &BR,
                                             ProgramStateRef ErrorState,
                                             NonLoc Val, bool MarkTaint) {
  if (SymbolRef Sym = Val.getAsSymbol()) {
    // If the offset is a symbolic value, iterate over its "parts" with
    // `SymExpr::symbols()` and mark each of them as interesting.
    // For example, if the offset is `x*4 + y` then we put interestingness onto
    // the SymSymExpr `x*4 + y`, the SymIntExpr `x*4` and the two data symbols
    // `x` and `y`.
    for (SymbolRef PartSym : Sym->symbols())
      BR.markInteresting(PartSym);
  }

  if (MarkTaint) {
    // If the issue that we're reporting depends on the taintedness of the
    // offset, then put interestingness onto symbols that could be the origin
    // of the taint. Note that this may find symbols that did not appear in
    // `Sym->symbols()` (because they're only loosely connected to `Val`).
    for (SymbolRef Sym : getTaintedSymbols(ErrorState, Val))
      BR.markInteresting(Sym);
  }
}

void ArrayBoundChecker::reportOOB(CheckerContext &C, ProgramStateRef ErrorState,
                                  Messages Msgs, NonLoc Offset,
                                  std::optional<NonLoc> Extent,
                                  bool IsTaintBug /*=false*/) const {

  ExplodedNode *ErrorNode = C.generateErrorNode(ErrorState);
  if (!ErrorNode)
    return;

  auto BR = std::make_unique<PathSensitiveBugReport>(
      IsTaintBug ? TaintBT : BT, Msgs.Short, Msgs.Full, ErrorNode);

  // FIXME: ideally we would just call trackExpressionValue() and that would
  // "do the right thing": mark the relevant symbols as interesting, track the
  // control dependencies and statements storing the relevant values and add
  // helpful diagnostic pieces. However, right now trackExpressionValue() is
  // a heap of unreliable heuristics, so it would cause several issues:
  // - Interestingness is not applied consistently, e.g. if `array[x+10]`
  //   causes an overflow, then `x` is not marked as interesting.
  // - We get irrelevant diagnostic pieces, e.g. in the code
  //   `int *p = (int*)malloc(2*sizeof(int)); p[3] = 0;`
  //   it places a "Storing uninitialized value" note on the `malloc` call
  //   (which is technically true, but irrelevant).
  // If trackExpressionValue() becomes reliable, it should be applied instead
  // of this custom markPartsInteresting().
  markPartsInteresting(*BR, ErrorState, Offset, IsTaintBug);
  if (Extent)
    markPartsInteresting(*BR, ErrorState, *Extent, IsTaintBug);

  C.emitReport(std::move(BR));
}

void ArrayBoundChecker::warnFlexibleArrayAccess(CheckerContext &C,
                                                ProgramStateRef State,
                                                const ArraySubscriptExpr *E,
                                                StringRef Name, NonLoc Index,
                                                nonloc::ConcreteInt ArraySize,
                                                QualType ElemType) const {
  ExplodedNode *WarnNode = C.generateNonFatalErrorNode(State);
  if (WarnNode) {
    int64_t ExtentN = ArraySize.getValue()->getZExtValue();

    assert(ExtentN <= 1 && "Flexible arrays are expected to have size 0 or 1");

    SmallString<256> Buf;
    llvm::raw_svector_ostream Out(Buf);
    Out << "Access of " << Name << " containing ";
    if (ExtentN != 1) {
      Out << ExtentN << " '" << ElemType.getAsString() << "' elements";
    } else {
      Out << "a single '" << ElemType.getAsString() << "' element";
    }
    Out << " as a possible 'flexible array member'";

    auto BR = std::make_unique<PathSensitiveBugReport>(
        BT,
        formatv(
            "Potential out of bound access to {0}, which may be a 'flexible "
            "array member'",
            Name)
            .str(),
        Buf, WarnNode);

    markPartsInteresting(*BR, State, Index, false);
    markPartsInteresting(*BR, State, ArraySize, false);
    C.emitReport(std::move(BR));
  }
}

bool ArrayBoundChecker::isFromCtypeMacro(const Expr *E, ASTContext &ACtx) {
  SourceLocation Loc = E->getBeginLoc();
  if (!Loc.isMacroID())
    return false;

  StringRef MacroName = Lexer::getImmediateMacroName(
      Loc, ACtx.getSourceManager(), ACtx.getLangOpts());

  if (MacroName.size() < 7 || MacroName[0] != 'i' || MacroName[1] != 's')
    return false;

  return ((MacroName == "isalnum") || (MacroName == "isalpha") ||
          (MacroName == "isblank") || (MacroName == "isdigit") ||
          (MacroName == "isgraph") || (MacroName == "islower") ||
          (MacroName == "isnctrl") || (MacroName == "isprint") ||
          (MacroName == "ispunct") || (MacroName == "isspace") ||
          (MacroName == "isupper") || (MacroName == "isxdigit"));
}

bool ArrayBoundChecker::isOffsetObviouslyNonnegative(const Expr *E,
                                                     CheckerContext &C) {
  const ArraySubscriptExpr *ASE = getAsCleanArraySubscriptExpr(E, C);
  if (!ASE)
    return false;
  return ASE->getIdx()->getType()->isUnsignedIntegerOrEnumerationType();
}

bool ArrayBoundChecker::isInAddressOf(const Stmt *S, ASTContext &ACtx) {
  ParentMapContext &ParentCtx = ACtx.getParentMapContext();
  do {
    const DynTypedNodeList Parents = ParentCtx.getParents(*S);
    if (Parents.empty())
      return false;
    S = Parents[0].get<Stmt>();
  } while (isa_and_nonnull<ParenExpr, ImplicitCastExpr>(S));
  const auto *UnaryOp = dyn_cast_or_null<UnaryOperator>(S);
  return UnaryOp && UnaryOp->getOpcode() == UO_AddrOf;
}

bool ArrayBoundChecker::isIdiomaticPastTheEndPtr(const Expr *E,
                                                 ProgramStateRef State,
                                                 NonLoc Offset, NonLoc Limit,
                                                 CheckerContext &C) {
  if (isa<ArraySubscriptExpr>(E) && isInAddressOf(E, C.getASTContext())) {
    auto [EqualsToThreshold, NotEqualToThreshold] = compareValueToThreshold(
        State, Offset, Limit, C.getSValBuilder(), /*CheckEquality=*/true);
    return EqualsToThreshold && !NotEqualToThreshold;
  }
  return false;
}

void ento::registerArrayBoundChecker(CheckerManager &mgr) {
  auto *checker = mgr.registerChecker<ArrayBoundChecker>();
  checker->EnableFakeFlexibleArrayWarn =
      mgr.getAnalyzerOptions().getCheckerBooleanOption(
          checker->getName(), "EnableFakeFlexibleArrayWarn");
}

bool ento::shouldRegisterArrayBoundChecker(const CheckerManager &mgr) {
  return true;
}
