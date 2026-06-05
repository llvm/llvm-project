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

#include "clang/AST/ParentMapContext.h"
#include "clang/StaticAnalyzer/Checkers/BoundsChecking.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include <optional>

using namespace clang;
using namespace ento;
using namespace taint;

namespace {
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

  void reportOOB(CheckerContext &C, ProgramStateRef ErrorState, Messages Msgs,
                 NonLoc Offset, std::optional<NonLoc> Extent,
                 bool IsTaintBug = false) const;

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

void ArrayBoundChecker::handleAccessExpr(const Expr *E,
                                         CheckerContext &C) const {
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

  const MemSpaceRegion *Space = Reg->getMemorySpace(State);
  auto Extent = getDynamicExtent(State, Reg, SVB).getAs<NonLoc>();

  // A symbolic region in unknown space represents an unknown pointer that
  // may point into the middle of an array, so we don't look for underflows.
  // Both conditions are significant because we want to check underflows in
  // symbolic regions on the heap (which may be introduced by checkers like
  // MallocChecker that call SValBuilder::getConjuredHeapSymbolVal()) and
  // non-symbolic regions (e.g. a field subregion of a symbolic region) in
  // unknown space.

  CheckFlags Flags = {
      /*CheckUnderflow=*/!(isa<SymbolicRegion>(Reg) &&
                           isa<UnknownSpaceRegion>(Space)),
      /*OffsetObviouslyNonnegative=*/isOffsetObviouslyNonnegative(E, C),
      /*AcceptPastTheEnd=*/isa<ArraySubscriptExpr>(E) &&
          isInAddressOf(E, C.getASTContext()),
  };

  BoundsCheckResult Res = checkBounds(State, SVB, ByteOffset, Extent, Flags);

  std::string RN = getRegionName(Reg->getMemorySpace(C.getState()), Reg);

  switch (Res.getKind()) {
  case BoundsCheckResult::Kind::Paradox:
    // The current state is paradoxical (due to bad modeling of casts we
    // assumed that an unsigned value is negative), so we should sink the
    // execution path.
    C.addSink();
    return;

  case BoundsCheckResult::Kind::Valid: {
    const NoteTag *Tag = nullptr;
    if (Res.hasAssumption()) {
      SizeUnit SU = SizeUnit::forExpr(E, C);
      Tag = C.getNoteTag(
          [Res, RN, SU](PathSensitiveBugReport &BR) -> std::string {
            return Res.getMessage(BR, RN, SU);
          });
    }

    C.addTransition(Res.getState(), Tag);
    return;
  }
  case BoundsCheckResult::Kind::TaintBug: {
    // Diagnostic detail: saying "tainted offset" is always correct, but
    // the common case is that 'idx' is tainted in 'arr[idx]' and then it's
    // nicer to say "tainted index".
    const char *OffsetName = "offset";
    if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
      if (isTainted(State, ASE->getIdx(), C.getStackFrame()))
        OffsetName = "index";

    Messages Msgs =
        getTaintMsgs(RN, OffsetName,
                     /*AlsoMentionUnderflow=*/Res.assumedNonNegative());
    reportOOB(C, Res.getState(), Msgs, ByteOffset, Extent,
              /*IsTaintBug=*/true);
    return;
  }
  default: {
    SizeUnit SU = SizeUnit::forSVal(Location, C.getASTContext());
    Messages Msgs =
        getNonTaintMsgs(RN, SU, ByteOffset, Extent, *Res.getBadOffsetKind());
    reportOOB(C, Res.getState(), Msgs, ByteOffset, Extent);
    return;
  }
  }
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
