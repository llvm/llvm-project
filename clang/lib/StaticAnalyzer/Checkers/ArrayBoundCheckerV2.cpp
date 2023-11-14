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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang;
using namespace ento;
using namespace taint;
using llvm::formatv;

namespace {
enum OOB_Kind { OOB_Precedes, OOB_Exceeds, OOB_Taint };

class ArrayBoundCheckerV2 :
    public Checker<check::Location> {
  BugType BT{this, "Out-of-bound access"};
  BugType TaintBT{this, "Out-of-bound access", categories::TaintedData};

  void reportOOB(CheckerContext &C, ProgramStateRef ErrorState, OOB_Kind Kind,
                 NonLoc Offset, std::string RegName, std::string Msg) const;

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

static std::string getRegionName(const SubRegion *Region) {
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

  if (isa<SymbolicRegion>(Region) &&
      isa<HeapSpaceRegion>(Region->getMemorySpace()))
    return "the heap area";

  if (isa<StringRegion>(Region))
    return "the string literal";

  return "the region";
}

static std::optional<int64_t> getConcreteValue(NonLoc SV) {
  if (auto ConcreteVal = SV.getAs<nonloc::ConcreteInt>()) {
    return ConcreteVal->getValue().tryExtValue();
  }
  return std::nullopt;
}

static std::string getShortMsg(OOB_Kind Kind, std::string RegName) {
  static const char *ShortMsgTemplates[] = {
      "Out of bound access to memory preceding {0}",
      "Out of bound access to memory after the end of {0}",
      "Potential out of bound access to {0} with tainted offset"};

  return formatv(ShortMsgTemplates[Kind], RegName);
}

static std::string getPrecedesMsg(std::string RegName, NonLoc Offset) {
  SmallString<128> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of " << RegName << " at negative byte offset";
  if (auto ConcreteIdx = Offset.getAs<nonloc::ConcreteInt>())
    Out << ' ' << ConcreteIdx->getValue();
  return std::string(Buf);
}
static std::string getExceedsMsg(ASTContext &ACtx, std::string RegName,
                                 NonLoc Offset, NonLoc Extent, SVal Location) {
  const auto *EReg = Location.getAsRegion()->getAs<ElementRegion>();
  assert(EReg && "this checker only handles element access");
  QualType ElemType = EReg->getElementType();

  std::optional<int64_t> OffsetN = getConcreteValue(Offset);
  std::optional<int64_t> ExtentN = getConcreteValue(Extent);

  bool UseByteOffsets = true;
  if (int64_t ElemSize = ACtx.getTypeSizeInChars(ElemType).getQuantity()) {
    const bool OffsetHasRemainder = OffsetN && *OffsetN % ElemSize;
    const bool ExtentHasRemainder = ExtentN && *ExtentN % ElemSize;
    if (!OffsetHasRemainder && !ExtentHasRemainder) {
      UseByteOffsets = false;
      if (OffsetN)
        *OffsetN /= ElemSize;
      if (ExtentN)
        *ExtentN /= ElemSize;
    }
  }

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of ";
  if (!ExtentN && !UseByteOffsets)
    Out << "'" << ElemType.getAsString() << "' element in ";
  Out << RegName << " at ";
  if (OffsetN) {
    Out << (UseByteOffsets ? "byte offset " : "index ") << *OffsetN;
  } else {
    Out << "an overflowing " << (UseByteOffsets ? "byte offset" : "index");
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

  return std::string(Buf);
}
static std::string getTaintMsg(std::string RegName) {
  SmallString<128> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << "Access of " << RegName
      << " with a tainted offset that may be too large";
  return std::string(Buf);
}

void ArrayBoundCheckerV2::checkLocation(SVal Location, bool IsLoad,
                                        const Stmt *LoadS,
                                        CheckerContext &C) const {

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
  if (isFromCtypeMacro(LoadS, C.getASTContext()))
    return;

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  const std::optional<std::pair<const SubRegion *, NonLoc>> &RawOffset =
      computeOffset(State, SVB, Location);

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
    auto [PrecedesLowerBound, WithinLowerBound] = compareValueToThreshold(
        State, ByteOffset, SVB.makeZeroArrayIndex(), SVB);

    if (PrecedesLowerBound && !WithinLowerBound) {
      // We know that the index definitely precedes the lower bound.
      std::string RegName = getRegionName(Reg);
      std::string Msg = getPrecedesMsg(RegName, ByteOffset);
      reportOOB(C, PrecedesLowerBound, OOB_Precedes, ByteOffset, RegName, Msg);
      return;
    }

    if (WithinLowerBound)
      State = WithinLowerBound;
  }

  // CHECK UPPER BOUND
  DefinedOrUnknownSVal Size = getDynamicExtent(State, Reg, SVB);
  if (auto KnownSize = Size.getAs<NonLoc>()) {
    auto [WithinUpperBound, ExceedsUpperBound] =
        compareValueToThreshold(State, ByteOffset, *KnownSize, SVB);

    if (ExceedsUpperBound) {
      if (!WithinUpperBound) {
        // We know that the index definitely exceeds the upper bound.
        std::string RegName = getRegionName(Reg);
        std::string Msg = getExceedsMsg(C.getASTContext(), RegName, ByteOffset,
                                        *KnownSize, Location);
        reportOOB(C, ExceedsUpperBound, OOB_Exceeds, ByteOffset, RegName, Msg);
        return;
      }
      if (isTainted(State, ByteOffset)) {
        // Both cases are possible, but the index is tainted, so report.
        std::string RegName = getRegionName(Reg);
        std::string Msg = getTaintMsg(RegName);
        reportOOB(C, ExceedsUpperBound, OOB_Taint, ByteOffset, RegName, Msg);
        return;
      }
    }

    if (WithinUpperBound)
      State = WithinUpperBound;
  }

  C.addTransition(State);
}

void ArrayBoundCheckerV2::reportOOB(CheckerContext &C,
                                    ProgramStateRef ErrorState, OOB_Kind Kind,
                                    NonLoc Offset, std::string RegName,
                                    std::string Msg) const {

  ExplodedNode *ErrorNode = C.generateErrorNode(ErrorState);
  if (!ErrorNode)
    return;

  std::string ShortMsg = getShortMsg(Kind, RegName);

  auto BR = std::make_unique<PathSensitiveBugReport>(
      Kind == OOB_Taint ? TaintBT : BT, ShortMsg, Msg, ErrorNode);

  // Track back the propagation of taintedness.
  if (Kind == OOB_Taint)
    for (SymbolRef Sym : getTaintedSymbols(ErrorState, Offset))
      BR->markInteresting(Sym);

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
