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
  mutable std::unique_ptr<BuiltinBug> BT;
  mutable std::unique_ptr<BugType> TaintBT;

  enum OOB_Kind { OOB_Precedes, OOB_Excedes };

  void reportOOB(CheckerContext &C, ProgramStateRef errorState,
                 OOB_Kind kind) const;
  void reportTaintOOB(CheckerContext &C, ProgramStateRef errorState,
                      SVal TaintedSVal) const;

  static bool isFromCtypeMacro(const Stmt *S, ASTContext &AC);

public:
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;
};

// FIXME: Eventually replace RegionRawOffset with this class.
class RegionRawOffsetV2 {
private:
  const SubRegion *baseRegion;
  SVal byteOffset;

  RegionRawOffsetV2()
    : baseRegion(nullptr), byteOffset(UnknownVal()) {}

public:
  RegionRawOffsetV2(const SubRegion *base, NonLoc offset)
      : baseRegion(base), byteOffset(offset) { assert(base); }

  NonLoc getByteOffset() const { return byteOffset.castAs<NonLoc>(); }
  const SubRegion *getRegion() const { return baseRegion; }

  static RegionRawOffsetV2 computeOffset(ProgramStateRef state,
                                         SValBuilder &svalBuilder,
                                         SVal location);

  void dump() const;
  void dumpToStream(raw_ostream &os) const;
};
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
        // The constant should never be 0 here, since it the result of scaling
        // based on the size of a type which is never 0.
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
  const RegionRawOffsetV2 &rawOffset =
    RegionRawOffsetV2::computeOffset(state, svalBuilder, location);

  if (!rawOffset.getRegion())
    return;

  NonLoc ByteOffset = rawOffset.getByteOffset();

  // CHECK LOWER BOUND
  const MemSpaceRegion *SR = rawOffset.getRegion()->getMemorySpace();
  if (!llvm::isa<UnknownSpaceRegion>(SR)) {
    // A pointer to UnknownSpaceRegion may point to the middle of
    // an allocated region.

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
  DefinedOrUnknownSVal Size =
      getDynamicExtent(state, rawOffset.getRegion(), svalBuilder);
  if (auto KnownSize = Size.getAs<NonLoc>()) {
    auto [state_withinUpperBound, state_exceedsUpperBound] =
        compareValueToThreshold(state, ByteOffset, *KnownSize, svalBuilder);

    if (state_exceedsUpperBound) {
      if (!state_withinUpperBound) {
        // We know that the index definitely exceeds the upper bound.
        reportOOB(checkerContext, state_exceedsUpperBound, OOB_Excedes);
        return;
      }
      if (isTainted(state, ByteOffset)) {
        // Both cases are possible, but the index is tainted, so report.
        reportTaintOOB(checkerContext, state_exceedsUpperBound, ByteOffset);
        return;
      }
    }

    if (state_withinUpperBound)
      state = state_withinUpperBound;
  }

  checkerContext.addTransition(state);
}

void ArrayBoundCheckerV2::reportTaintOOB(CheckerContext &checkerContext,
                                         ProgramStateRef errorState,
                                         SVal TaintedSVal) const {
  ExplodedNode *errorNode = checkerContext.generateErrorNode(errorState);
  if (!errorNode)
    return;

  if (!TaintBT)
    TaintBT.reset(
        new BugType(this, "Out-of-bound access", categories::TaintedData));

  SmallString<256> buf;
  llvm::raw_svector_ostream os(buf);
  os << "Out of bound memory access (index is tainted)";
  auto BR =
      std::make_unique<PathSensitiveBugReport>(*TaintBT, os.str(), errorNode);

  // Track back the propagation of taintedness.
  for (SymbolRef Sym : getTaintedSymbols(errorState, TaintedSVal)) {
    BR->markInteresting(Sym);
  }

  checkerContext.emitReport(std::move(BR));
}

void ArrayBoundCheckerV2::reportOOB(CheckerContext &checkerContext,
                                    ProgramStateRef errorState,
                                    OOB_Kind kind) const {

  ExplodedNode *errorNode = checkerContext.generateErrorNode(errorState);
  if (!errorNode)
    return;

  if (!BT)
    BT.reset(new BuiltinBug(this, "Out-of-bound access"));

  // FIXME: This diagnostics are preliminary.  We should get far better
  // diagnostics for explaining buffer overruns.

  SmallString<256> buf;
  llvm::raw_svector_ostream os(buf);
  os << "Out of bound memory access ";
  switch (kind) {
  case OOB_Precedes:
    os << "(accessed memory precedes memory block)";
    break;
  case OOB_Excedes:
    os << "(access exceeds upper limit of memory block)";
    break;
  }
  auto BR = std::make_unique<PathSensitiveBugReport>(*BT, os.str(), errorNode);
  checkerContext.emitReport(std::move(BR));
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

#ifndef NDEBUG
LLVM_DUMP_METHOD void RegionRawOffsetV2::dump() const {
  dumpToStream(llvm::errs());
}

void RegionRawOffsetV2::dumpToStream(raw_ostream &os) const {
  os << "raw_offset_v2{" << getRegion() << ',' << getByteOffset() << '}';
}
#endif

// Lazily computes a value to be used by 'computeOffset'.  If 'val'
// is unknown or undefined, we lazily substitute '0'.  Otherwise,
// return 'val'.
static inline SVal getValue(SVal val, SValBuilder &svalBuilder) {
  return val.isUndef() ? svalBuilder.makeZeroArrayIndex() : val;
}

// Scale a base value by a scaling factor, and return the scaled
// value as an SVal.  Used by 'computeOffset'.
static inline SVal scaleValue(ProgramStateRef state,
                              NonLoc baseVal, CharUnits scaling,
                              SValBuilder &sb) {
  return sb.evalBinOpNN(state, BO_Mul, baseVal,
                        sb.makeArrayIndex(scaling.getQuantity()),
                        sb.getArrayIndexType());
}

// Add an SVal to another, treating unknown and undefined values as
// summing to UnknownVal.  Used by 'computeOffset'.
static SVal addValue(ProgramStateRef state, SVal x, SVal y,
                     SValBuilder &svalBuilder) {
  // We treat UnknownVals and UndefinedVals the same here because we
  // only care about computing offsets.
  if (x.isUnknownOrUndef() || y.isUnknownOrUndef())
    return UnknownVal();

  return svalBuilder.evalBinOpNN(state, BO_Add, x.castAs<NonLoc>(),
                                 y.castAs<NonLoc>(),
                                 svalBuilder.getArrayIndexType());
}

/// Compute a raw byte offset from a base region.  Used for array bounds
/// checking.
RegionRawOffsetV2 RegionRawOffsetV2::computeOffset(ProgramStateRef state,
                                                   SValBuilder &svalBuilder,
                                                   SVal location)
{
  const MemRegion *region = location.getAsRegion();
  SVal offset = UndefinedVal();

  while (region) {
    switch (region->getKind()) {
      default: {
        if (const SubRegion *subReg = dyn_cast<SubRegion>(region)) {
          if (auto Offset = getValue(offset, svalBuilder).getAs<NonLoc>())
            return RegionRawOffsetV2(subReg, *Offset);
        }
        return RegionRawOffsetV2();
      }
      case MemRegion::ElementRegionKind: {
        const ElementRegion *elemReg = cast<ElementRegion>(region);
        SVal index = elemReg->getIndex();
        if (!isa<NonLoc>(index))
          return RegionRawOffsetV2();
        QualType elemType = elemReg->getElementType();
        // If the element is an incomplete type, go no further.
        ASTContext &astContext = svalBuilder.getContext();
        if (elemType->isIncompleteType())
          return RegionRawOffsetV2();

        // Update the offset.
        offset = addValue(state,
                          getValue(offset, svalBuilder),
                          scaleValue(state,
                          index.castAs<NonLoc>(),
                          astContext.getTypeSizeInChars(elemType),
                          svalBuilder),
                          svalBuilder);

        if (offset.isUnknownOrUndef())
          return RegionRawOffsetV2();

        region = elemReg->getSuperRegion();
        continue;
      }
    }
  }
  return RegionRawOffsetV2();
}

void ento::registerArrayBoundCheckerV2(CheckerManager &mgr) {
  mgr.registerChecker<ArrayBoundCheckerV2>();
}

bool ento::shouldRegisterArrayBoundCheckerV2(const CheckerManager &mgr) {
  return true;
}
