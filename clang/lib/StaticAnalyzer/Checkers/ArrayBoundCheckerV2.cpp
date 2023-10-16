//== ArrayBoundCheckerV2.cpp ------------------------------------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ArrayBoundCheckerV2, which is a path-sensitive check
// which looks for an out-of-bound array element access.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CharUnits.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang;
using namespace ento;
using namespace taint;

namespace {
class ArrayBoundCheckerV2 :
    public Checker<check::Location> {
  BugType BT{this, "Out-of-bound access"};
  BugType TaintBT{this, "Out-of-bound access", categories::TaintedData};

  enum OOB_Kind { OOB_Precedes, OOB_Exceeds, OOB_Taint };

  void reportOOB(CheckerContext &C, ProgramStateRef ErrorState, OOB_Kind Kind,
                 SVal TaintedSVal = UnknownVal()) const;

  static bool isFromCtypeMacro(const Stmt *S, ASTContext &AC);

public:
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;
};
} // anonymous namespace

/// For a given Location that can be represented as a symbolic expression
/// Arr[Idx] (or perhaps Arr[Idx1][Idx2] etc.), return the parent memory block
/// Arr and the distance of Location from the beginning of Arr (expressed in a
/// NonLoc that specifies the number of CharUnits). Returns nullopt when these
/// cannot be determined.
std::optional<std::pair<const SubRegion *, NonLoc>>
computeOffset(ProgramStateRef State, SValBuilder &SVB, SVal Location) {
  QualType T = SVB.getArrayIndexType();
  auto EvalBinOp = [&SVB, State, T](BinaryOperatorKind Op, NonLoc L, NonLoc R) {
    // We will use this utility to add and multiply values.
    return SVB.evalBinOpNN(State, Op, L, R, T).getAs<NonLoc>();
  };

  const auto *Region = dyn_cast_or_null<SubRegion>(Location.getAsRegion());
  // This is initialized to nullopt instead of 0 becasue we want to discard the
  // situations when there are no ElementRegion layers.
  std::optional<NonLoc> Offset = std::nullopt;

  while (const auto *ERegion = dyn_cast_or_null<ElementRegion>(Region)) {
    const auto Index = ERegion->getIndex().getAs<NonLoc>();
    if (!Index)
      return std::nullopt;

    QualType ElemType = ERegion->getElementType();

    // Paranoia: getTypeSizeInChars() doesn't handle incomplete types.
    if (ElemType->isIncompleteType())
      return std::nullopt;

    // Calculate Delta = Index * sizeof(ElemType).
    NonLoc Size = SVB.makeArrayIndex(
        SVB.getContext().getTypeSizeInChars(ElemType).getQuantity());
    auto Delta = EvalBinOp(BO_Mul, *Index, Size);
    if (!Delta)
      return std::nullopt;

    if (!Offset) {
      // Store Delta as the first non-nullopt Offset value.
      Offset = Delta;
    } else {
      // Update the previous offset with Offset += Delta.
      Offset = EvalBinOp(BO_Add, *Offset, *Delta);
      if (!Offset)
        return std::nullopt;
    }

    // Continute the offset calculations with the SuperRegion.
    Region = ERegion->getSuperRegion()->getAs<SubRegion>();
  }

  if (Region && Offset)
    return std::make_pair(Region, *Offset);

  return std::nullopt;
}

// TODO: once the constraint manager is smart enough to handle non simplified
// symbolic expressions remove this function. Note that this can not be used in
// the constraint manager as is, since this does not handle overflows. It is
// safe to assume, however, that memory offsets will not overflow.
// NOTE: callers of this function need to be aware of the effects of overflows
// and signed<->unsigned conversions!
static std::pair<NonLoc, nonloc::ConcreteInt>
getSimplifiedOffsets(NonLoc offset, nonloc::ConcreteInt extent,
                     SValBuilder &svalBuilder) {
  std::optional<nonloc::SymbolVal> SymVal = offset.getAs<nonloc::SymbolVal>();
  if (SymVal && SymVal->isExpression()) {
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SymVal->getSymbol())) {
      llvm::APSInt constant =
          APSIntType(extent.getValue()).convert(SIE->getRHS());
      switch (SIE->getOpcode()) {
      case BO_Mul:
        // The constant should never be 0 here, becasue multiplication by zero
        // is simplified by the engine.
        if ((extent.getValue() % constant) != 0)
          return std::pair<NonLoc, nonloc::ConcreteInt>(offset, extent);
        else
          return getSimplifiedOffsets(
              nonloc::SymbolVal(SIE->getLHS()),
              svalBuilder.makeIntVal(extent.getValue() / constant),
              svalBuilder);
      case BO_Add:
        return getSimplifiedOffsets(
            nonloc::SymbolVal(SIE->getLHS()),
            svalBuilder.makeIntVal(extent.getValue() - constant), svalBuilder);
      default:
        break;
      }
    }
  }

  return std::pair<NonLoc, nonloc::ConcreteInt>(offset, extent);
}

// Evaluate the comparison Value < Threshold with the help of the custom
// simplification algorithm defined for this checker. Return a pair of states,
// where the first one corresponds to "value below threshold" and the second
// corresponds to "value at or above threshold". Returns {nullptr, nullptr} in
// the case when the evaluation fails.
static std::pair<ProgramStateRef, ProgramStateRef>
compareValueToThreshold(ProgramStateRef State, NonLoc Value, NonLoc Threshold,
                        SValBuilder &SVB) {
  if (auto ConcreteThreshold = Threshold.getAs<nonloc::ConcreteInt>()) {
    std::tie(Value, Threshold) = getSimplifiedOffsets(Value, *ConcreteThreshold, SVB);
  }
  if (auto ConcreteThreshold = Threshold.getAs<nonloc::ConcreteInt>()) {
    QualType T = Value.getType(SVB.getContext());
    if (T->isUnsignedIntegerType() && ConcreteThreshold->getValue().isNegative()) {
      // In this case we reduced the bound check to a comparison of the form
      //   (symbol or value with unsigned type) < (negative number)
      // which is always false. We are handling these cases separately because
      // evalBinOpNN can perform a signed->unsigned conversion that turns the
      // negative number into a huge positive value and leads to wildly
      // inaccurate conclusions.
      return {nullptr, State};
    }
  }
  auto BelowThreshold =
      SVB.evalBinOpNN(State, BO_LT, Value, Threshold, SVB.getConditionType()).getAs<NonLoc>();

  if (BelowThreshold)
    return State->assume(*BelowThreshold);

  return {nullptr, nullptr};
}

void ArrayBoundCheckerV2::checkLocation(SVal location, bool isLoad,
                                        const Stmt* LoadS,
                                        CheckerContext &checkerContext) const {

  // NOTE: Instead of using ProgramState::assumeInBound(), we are prototyping
  // some new logic here that reasons directly about memory region extents.
  // Once that logic is more mature, we can bring it back to assumeInBound()
  // for all clients to use.
  //
  // The algorithm we are using here for bounds checking is to see if the
  // memory access is within the extent of the base region.  Since we
  // have some flexibility in defining the base region, we can achieve
  // various levels of conservatism in our buffer overflow checking.

  // The header ctype.h (from e.g. glibc) implements the isXXXXX() macros as
  //   #define isXXXXX(arg) (LOOKUP_TABLE[arg] & BITMASK_FOR_XXXXX)
  // and incomplete analysis of these leads to false positives. As even
  // accurate reports would be confusing for the users, just disable reports
  // from these macros:
  if (isFromCtypeMacro(LoadS, checkerContext.getASTContext()))
    return;

  ProgramStateRef state = checkerContext.getState();
  SValBuilder &svalBuilder = checkerContext.getSValBuilder();

  const std::optional<std::pair<const SubRegion *, NonLoc>> &RawOffset =
      computeOffset(state, svalBuilder, location);

  if (!RawOffset)
    return;

  auto [Reg, ByteOffset] = *RawOffset;

  // CHECK LOWER BOUND
  const MemSpaceRegion *Space = Reg->getMemorySpace();
  if (!(isa<SymbolicRegion>(Reg) && isa<UnknownSpaceRegion>(Space))) {
    // A symbolic region in unknown space represents an unknown pointer that
    // may point into the middle of an array, so we don't look for underflows.
    // Both conditions are significant because we want to check underflows in
    // symbolic regions on the heap (which may be introduced by checkers like
    // MallocChecker that call SValBuilder::getConjuredHeapSymbolVal()) and
    // non-symbolic regions (e.g. a field subregion of a symbolic region) in
    // unknown space.
    auto [state_precedesLowerBound, state_withinLowerBound] =
        compareValueToThreshold(state, ByteOffset,
                                svalBuilder.makeZeroArrayIndex(), svalBuilder);

    if (state_precedesLowerBound && !state_withinLowerBound) {
      // We know that the index definitely precedes the lower bound.
      reportOOB(checkerContext, state_precedesLowerBound, OOB_Precedes);
      return;
    }

    if (state_withinLowerBound)
      state = state_withinLowerBound;
  }

  // CHECK UPPER BOUND
  DefinedOrUnknownSVal Size = getDynamicExtent(state, Reg, svalBuilder);
  if (auto KnownSize = Size.getAs<NonLoc>()) {
    auto [state_withinUpperBound, state_exceedsUpperBound] =
        compareValueToThreshold(state, ByteOffset, *KnownSize, svalBuilder);

    if (state_exceedsUpperBound) {
      if (!state_withinUpperBound) {
        // We know that the index definitely exceeds the upper bound.
        reportOOB(checkerContext, state_exceedsUpperBound, OOB_Exceeds);
        return;
      }
      if (isTainted(state, ByteOffset)) {
        // Both cases are possible, but the index is tainted, so report.
        reportOOB(checkerContext, state_exceedsUpperBound, OOB_Taint,
                  ByteOffset);
        return;
      }
    }

    if (state_withinUpperBound)
      state = state_withinUpperBound;
  }

  checkerContext.addTransition(state);
}

void ArrayBoundCheckerV2::reportOOB(CheckerContext &C,
                                    ProgramStateRef ErrorState, OOB_Kind Kind,
                                    SVal TaintedSVal) const {

  ExplodedNode *ErrorNode = C.generateErrorNode(ErrorState);
  if (!ErrorNode)
    return;

  // FIXME: These diagnostics are preliminary, and they'll be replaced soon by
  // a follow-up commit.

  SmallString<128> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Out of bound memory access ";

  switch (Kind) {
  case OOB_Precedes:
    Out << "(accessed memory precedes memory block)";
    break;
  case OOB_Exceeds:
    Out << "(access exceeds upper limit of memory block)";
    break;
  case OOB_Taint:
    Out << "(index is tainted)";
    break;
  }
  auto BR = std::make_unique<PathSensitiveBugReport>(
      Kind == OOB_Taint ? TaintBT : BT, Out.str(), ErrorNode);
  // Track back the propagation of taintedness, or do nothing if TaintedSVal is
  // the default UnknownVal().
  for (SymbolRef Sym : getTaintedSymbols(ErrorState, TaintedSVal)) {
    BR->markInteresting(Sym);
  }
  C.emitReport(std::move(BR));
}

bool ArrayBoundCheckerV2::isFromCtypeMacro(const Stmt *S, ASTContext &ACtx) {
  SourceLocation Loc = S->getBeginLoc();
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

void ento::registerArrayBoundCheckerV2(CheckerManager &mgr) {
  mgr.registerChecker<ArrayBoundCheckerV2>();
}

bool ento::shouldRegisterArrayBoundCheckerV2(const CheckerManager &mgr) {
  return true;
}
