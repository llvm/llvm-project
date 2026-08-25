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
#include "clang/StaticAnalyzer/Checkers/BoundsChecking.h"
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

class SizeUnit {
  QualType AsType;
  int64_t AsCharUnits;

  SizeUnit() : AsType(), AsCharUnits(1) {}

public:
  SizeUnit(QualType T, const ASTContext &ACtx)
      : AsType(T), AsCharUnits(ACtx.getTypeSizeInChars(T).getQuantity()) {
    assert(!T.isNull());
  }

  static SizeUnit bytes() { return SizeUnit(); }

  bool isBytes() const { return AsType.isNull(); }

  /// Return the element type that is "natural" for reporting out-of-bounds
  /// memory access to \p ER.
  static SizeUnit forElementRegion(const ElementRegion *ER,
                                   const ASTContext &ACtx) {
    return SizeUnit(ER->getElementType(), ACtx);
  }

  /// If `E` is a "clean" array subscript expression, return the type of the
  /// accessed element; otherwise return 'Bytes' because that's the best (or
  /// least bad) option for the assumption messages that use this.
  /// FIXME: It is unfortunate that this heuristic differs from the heuristic
  /// used for reporting assumption; but this difference is currently needed
  /// due to the unfortunate phrasing of the assumption messages.
  /// Get rid of this when the assumption note is rephrased and improved.
  static SizeUnit forExpr(const Expr *E, const CheckerContext &C) {
    const auto *ASE = getAsCleanArraySubscriptExpr(E, C);
    if (!ASE)
      return bytes();

    return SizeUnit(ASE->getType(), C.getASTContext());
  }

  int64_t asCharUnits() const { return AsCharUnits; }

  bool canExpress(std::optional<int64_t> Val) const {
    return asCharUnits() && (!Val || !(*Val % asCharUnits()));
  }

  std::string asExtentDesc() const {
    if (isBytes())
      return "the extent of";
    return formatv("the number of '{0}' elements in", AsType.getAsString());
  }

  std::string asElementName() const {
    if (isBytes())
      return "byte";
    return formatv("'{0}' element", AsType.getAsString());
  }
};

/// Strings that will be passed to the parameters 'desc' and 'fullDesc' of the
/// constructor of 'PathSensitiveBugReport'.
struct BugDescription {
  std::string Short;
  std::string Full;
};

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

  void handleAccessExpr(const Expr *E, CheckerContext &C) const;

  void reportOOB(CheckerContext &C, ProgramStateRef ErrorState,
                 BugDescription Desc, NonLoc Offset,
                 std::optional<NonLoc> Extent, bool IsTaintBug = false) const;

  static void markPartsInteresting(PathSensitiveBugReport &BR,
                                   ProgramStateRef ErrorState, NonLoc Val,
                                   bool MarkTaint);

  static bool isFromCtypeMacro(const Expr *E, ASTContext &AC);

  static bool isOffsetObviouslyNonnegative(const Expr *E, CheckerContext &C);

  static bool isInAddressOf(const Stmt *S, ASTContext &AC);

public:
  void checkPostStmt(const ArraySubscriptExpr *E, CheckerContext &C) const {
    handleAccessExpr(E, C);
  }
  void checkPostStmt(const UnaryOperator *E, CheckerContext &C) const {
    if (E->getOpcode() == UO_Deref)
      handleAccessExpr(E, C);
  }
  void checkPostStmt(const MemberExpr *E, CheckerContext &C) const {
    if (E->isArrow())
      handleAccessExpr(E->getBase(), C);
  }
};

} // anonymous namespace

/// Return true if information about the value of \p SV can put constraints
/// on some symbol which is interesting within the bug report \p BR.
/// In particular, this returns true when \p SV is interesting within \p BR;
/// but it also returns true if \p SV is an expression that contains integer
/// constants and a single symbolic operand which is interesting (in \p BR).
/// We need to use this instead of plain `BR.isInteresting()` because if we
/// are analyzing code like
///   int array[10];
///   int f(int arg) {
///     return array[arg] && array[arg + 10];
///   }
/// then the byte offsets are `arg * 4` and `(arg + 10) * 4`, which are not
/// sub-expressions of each other (but `getSimplifiedOffsets` is smart enough
/// to detect this out of bounds access).
static bool isDeterminedByInterestingSymbol(SVal SV,
                                            PathSensitiveBugReport &BR) {
  SymbolRef Sym = SV.getAsSymbol();
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

/// For a given \p CurRegion that can be represented as a symbolic expression
/// Arr[Idx] (or perhaps Arr[Idx1][Idx2] etc.), return the parent memory block
/// Arr and the distance of Location from the beginning of Arr (expressed in a
/// NonLoc that specifies the number of CharUnits). Returns nullopt when these
/// cannot be determined.
static std::optional<std::pair<const SubRegion *, NonLoc>>
computeOffset(ProgramStateRef State, SValBuilder &SVB,
              const ElementRegion *CurRegion) {
  QualType T = SVB.getArrayIndexType();
  auto EvalBinOp = [&SVB, State, T](BinaryOperatorKind Op, NonLoc L, NonLoc R) {
    // We will use this utility to add and multiply values.
    return SVB.evalBinOpNN(State, Op, L, R, T).getAs<NonLoc>();
  };

  const SubRegion *OwnerRegion = nullptr;
  std::optional<NonLoc> Offset = SVB.makeZeroArrayIndex();

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

static std::optional<int64_t> getConcreteValue(NonLoc SV) {
  if (auto ConcreteVal = SV.getAs<nonloc::ConcreteInt>()) {
    return ConcreteVal->getValue()->tryExtValue();
  }
  return std::nullopt;
}

static std::optional<int64_t> getConcreteValue(std::optional<NonLoc> SV) {
  return SV ? getConcreteValue(*SV) : std::nullopt;
}

static StringRef getAdjective(const bounds::CheckResult &R) {
  return (R.mayUnderflow()
              ? (R.mayOverflow() ? "a negative or overflowing" : "a negative")
              : (R.mayOverflow() ? "an overflowing" : "a valid"));
}

static StringRef getPreposition(const bounds::CheckResult &R) {
  return (R.mayUnderflow() ? (R.mayOverflow() ? "around" : "preceding")
                           : (R.mayOverflow() ? "after the end of" : "within"));
}

static BugDescription describeInvalidAccess(bounds::CheckResult Res,
                                            StringRef RegName, SizeUnit SU) {
  assert(Res.mayBeInvalid());

  std::optional<int64_t> OffsetN = getConcreteValue(Res.getOffset());
  std::optional<int64_t> ExtentN =
      getConcreteValue(Res.getExtentIfMayOverflow());

  if (SU.canExpress(OffsetN) && SU.canExpress(ExtentN)) {
    if (OffsetN)
      *OffsetN /= SU.asCharUnits();
    if (ExtentN)
      *ExtentN /= SU.asCharUnits();
  } else {
    // Fall back to reporting the offsets in bytes.
    SU = SizeUnit::bytes();
  }

  StringRef OffsetOrIndex = SU.isBytes() ? "byte offset" : "index";

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of ";
  if (OffsetN && !ExtentN && !SU.isBytes()) {
    // If the offset is reported as an index, then the report must mention the
    // element type (because it is not always clear from the code). It's more
    // natural to mention the element type later where the extent is described,
    // but if the extent is unknown/irrelevant, then the element type can be
    // inserted into the message at this point.
    Out << SU.asElementName() << " in ";
  }
  Out << RegName << " at ";
  if (OffsetN) {
    if (Res.mayUnderflow() && !Res.mayOverflow())
      Out << "negative ";
    Out << OffsetOrIndex << " " << *OffsetN;
  } else {
    Out << getAdjective(Res) << " " << OffsetOrIndex;
  }
  if (ExtentN) {
    Out << ", while it holds only ";
    if (*ExtentN != 1)
      Out << *ExtentN;
    else
      Out << "a single";

    Out << ' ' << SU.asElementName();

    if (*ExtentN != 1)
      Out << "s";
  }

  return {formatv("Out of bound access to memory {0} {1}", getPreposition(Res),
                  RegName),
          std::string(Buf)};
}

static BugDescription describeTaintBug(bounds::CheckResult Res,
                                       StringRef RegName,
                                       StringRef OffsetName) {
  assert(Res.mayBeInvalid());
  return {formatv("Potential out of bound access to {0} with tainted {1}",
                  RegName, OffsetName),
          formatv("Access of {0} with a tainted {1} that may be{2}{3}{4}",
                  RegName, OffsetName, Res.mayUnderflow() ? " negative" : "",
                  (Res.mayUnderflow() && Res.mayOverflow()) ? " or" : "",
                  Res.mayOverflow() ? " too large" : "")};
}

/// When the access was ambiguous (that is, mayBeInBounds() && mayBeInvalid()),
/// returns the note "assuming in bounds" note that is relevant for the bug
/// report \p BR. When the access wasn't ambiguous or the the assumption is
/// irrelevant for \p BR, this returns the empty string (which signifies "do
/// not emit a note tag" when returned by a note tag callback).
static std::string getAssumptionNote(bounds::CheckResult Res,
                                     PathSensitiveBugReport &BR,
                                     StringRef RegName, SizeUnit SU) {
  bool ShouldReportNonNegative = Res.mayUnderflow();
  if (!isDeterminedByInterestingSymbol(Res.getOffset(), BR)) {
    std::optional<NonLoc> E = Res.getExtentIfMayOverflow();
    if (E && isDeterminedByInterestingSymbol(*E, BR)) {
      // Even if the byte offset isn't interesting (e.g. it's a constant value),
      // the assumption can still be interesting if it provides information
      // about an interesting symbolic upper bound.
      ShouldReportNonNegative = false;
    } else {
      // We don't have anything interesting, don't report the assumption.
      return "";
    }
  }

  std::optional<int64_t> OffsetN = getConcreteValue(Res.getOffset());
  std::optional<int64_t> ExtentN =
      getConcreteValue(Res.getExtentIfMayOverflow());

  if (SU.canExpress(OffsetN) && SU.canExpress(ExtentN)) {
    if (OffsetN)
      *OffsetN /= SU.asCharUnits();
    if (ExtentN)
      *ExtentN /= SU.asCharUnits();
  } else {
    // Fall back to reporting the offsets in bytes.
    SU = SizeUnit::bytes();
  }

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Assuming ";
  if (!SU.isBytes()) {
    Out << "index ";
    if (OffsetN)
      Out << "'" << OffsetN << "' ";
  } else if (Res.mayOverflow()) {
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
  if (Res.mayOverflow()) {
    if (ShouldReportNonNegative)
      Out << " and";
    Out << " less than ";
    if (ExtentN)
      Out << *ExtentN << ", ";
    Out << SU.asExtentDesc() << ' ' << RegName;
  }
  return std::string(Out.str());
}

void ArrayBoundChecker::handleAccessExpr(const Expr *E,
                                         CheckerContext &C) const {
  ASTContext &ACtx = C.getASTContext();
  const ElementRegion *AccessedER =
      dyn_cast_or_null<ElementRegion>(C.getSVal(E).getAsRegion());
  if (!AccessedER)
    return;

  // The header ctype.h (from e.g. glibc) implements the isXXXXX() macros as
  //   #define isXXXXX(arg) (LOOKUP_TABLE[arg] & BITMASK_FOR_XXXXX)
  // and incomplete analysis of these leads to false positives. As even
  // accurate reports would be confusing for the users, just disable reports
  // from these macros:
  if (isFromCtypeMacro(E, ACtx))
    return;

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  const std::optional<std::pair<const SubRegion *, NonLoc>> &RawOffset =
      computeOffset(State, SVB, AccessedER);

  if (!RawOffset)
    return;

  auto [Reg, ByteOffset] = *RawOffset;

  const MemSpaceRegion *Space = Reg->getMemorySpace(State);
  auto Extent = getDynamicExtent(State, Reg, SVB).getAs<NonLoc>();

  // A symbolic region in unknown space represents an unknown pointer that
  // may point into the middle of an array, so we don't look for underflows.
  // Both conditions are significant because we want to check underflows in
  // symbolic regions on the heap (which may be introduced by checkers like
  // MallocChecker that call SValBuilder::getConjuredHeapSymbolVal()) and
  // non-symbolic regions (e.g. a field subregion of a symbolic region) in
  // unknown space.

  bounds::CheckFlags Flags = {
      /*CheckUnderflow=*/!(isa<SymbolicRegion>(Reg) &&
                           isa<UnknownSpaceRegion>(Space)),
      /*OffsetObviouslyNonnegative=*/isOffsetObviouslyNonnegative(E, C)};

  bounds::CheckResult Res = checkBounds(State, SVB, ByteOffset, Extent, Flags);

  if (Res.isCorruptedState()) {
    C.addSink();
    return;
  }

  std::string RegName =
      Reg->getDescriptiveName(/*UseQuotes=*/true, /*AllowFallback=*/true);

  const NoteTag *T = nullptr;
  if (Res.mayBeInvalid()) {
    if (!Res.mayBeInBounds()) {
      if (isa<ArraySubscriptExpr>(E) && isInAddressOf(E, ACtx) && Extent) {
        // Recognize and accept the idiomatic `&array[size]` expression that
        // forms the past-the-end pointer without actually dereferencing it.
        auto [EqualsToThreshold, NotEqualToThreshold] =
            bounds::compareValueToThreshold(State, SVB, ByteOffset, *Extent,
                                            /*CheckEquality=*/true);
        if (EqualsToThreshold && !NotEqualToThreshold) {
          C.addTransition(EqualsToThreshold);
          return;
        }
      }

      SizeUnit SU = SizeUnit::forElementRegion(AccessedER, ACtx);
      BugDescription Desc = describeInvalidAccess(Res, RegName, SU);
      reportOOB(C, State, Desc, ByteOffset, Res.getExtentIfMayOverflow());
      return;
    }

    if (isTainted(State, ByteOffset)) {
      // Diagnostic detail: saying "tainted offset" is always correct, but
      // the common case is that 'idx' is tainted in 'arr[idx]' and then it's
      // nicer to say "tainted index".
      StringRef OffsetName = "offset";
      if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
        if (isTainted(State, ASE->getIdx(), C.getStackFrame()))
          OffsetName = "index";

      BugDescription Desc = describeTaintBug(Res, RegName, OffsetName);
      reportOOB(C, State, Desc, ByteOffset, Res.getExtentIfMayOverflow(),
                /*IsTaintBug=*/true);
      return;
    }

    SizeUnit SU = SizeUnit::forExpr(E, C);
    T = C.getNoteTag(
        [Res, RegName, SU](PathSensitiveBugReport &BR) -> std::string {
          return getAssumptionNote(Res, BR, RegName, SU);
        });
  }

  C.addTransition(Res.getInBoundsState(), T);
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
                                  BugDescription Desc, NonLoc Offset,
                                  std::optional<NonLoc> Extent,
                                  bool IsTaintBug /*=false*/) const {

  ExplodedNode *ErrorNode = C.generateErrorNode(ErrorState);
  if (!ErrorNode)
    return;

  auto BR = std::make_unique<PathSensitiveBugReport>(
      IsTaintBug ? TaintBT : BT, Desc.Short, Desc.Full, ErrorNode);

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

void ento::registerArrayBoundChecker(CheckerManager &mgr) {
  mgr.registerChecker<ArrayBoundChecker>();
}

bool ento::shouldRegisterArrayBoundChecker(const CheckerManager &mgr) {
  return true;
}
