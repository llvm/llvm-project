//== ArrayBoundCheckerV2.cpp ------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ArrayBoundCheckerV2, which is a path-sensitive check
// which looks for an out-of-bound array element access.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/CharUnits.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class ArrayBoundCheckerV2 :
    public Checker<check::Location> {
  mutable std::unique_ptr<BuiltinBug> BT;

  enum OOB_Kind { OOB_Precedes, OOB_Excedes, OOB_Tainted };

  void reportOOB(CheckerContext &C, ProgramStateRef errorState,
                 OOB_Kind kind) const;

public:
  void checkLocation(SVal l, bool isLoad, const Stmt*S,
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
  RegionRawOffsetV2(const SubRegion* base, SVal offset)
    : baseRegion(base), byteOffset(offset) {}

  NonLoc getByteOffset() const { return byteOffset.castAs<NonLoc>(); }
  const SubRegion *getRegion() const { return baseRegion; }

  static RegionRawOffsetV2 computeOffset(ProgramStateRef state,
                                         SValBuilder &svalBuilder,
                                         SVal location);

  void dump() const;
  void dumpToStream(raw_ostream &os) const;
};
}

static SVal computeExtentBegin(SValBuilder &svalBuilder,
                               const MemRegion *region) {
  while (true)
    switch (region->getKind()) {
      default:
        return svalBuilder.makeZeroArrayIndex();
      case MemRegion::SymbolicRegionKind:
        // FIXME: improve this later by tracking symbolic lower bounds
        // for symbolic regions.
        return UnknownVal();
      case MemRegion::ElementRegionKind:
        region = cast<SubRegion>(region)->getSuperRegion();
        continue;
    }
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
  ProgramStateRef state = checkerContext.getState();
  ProgramStateRef originalState = state;

  SValBuilder &svalBuilder = checkerContext.getSValBuilder();
  const RegionRawOffsetV2 &rawOffset =
    RegionRawOffsetV2::computeOffset(state, svalBuilder, location);

  if (!rawOffset.getRegion())
    return;

  // CHECK LOWER BOUND: Is byteOffset < extent begin?
  //  If so, we are doing a load/store
  //  before the first valid offset in the memory region.

  SVal extentBegin = computeExtentBegin(svalBuilder, rawOffset.getRegion());

  if (Optional<NonLoc> NV = extentBegin.getAs<NonLoc>()) {
    SVal lowerBound =
        svalBuilder.evalBinOpNN(state, BO_LT, rawOffset.getByteOffset(), *NV,
                                svalBuilder.getConditionType());

    Optional<NonLoc> lowerBoundToCheck = lowerBound.getAs<NonLoc>();
    if (!lowerBoundToCheck)
      return;

    ProgramStateRef state_precedesLowerBound, state_withinLowerBound;
    std::tie(state_precedesLowerBound, state_withinLowerBound) =
      state->assume(*lowerBoundToCheck);

    // Are we constrained enough to definitely precede the lower bound?
    if (state_precedesLowerBound && !state_withinLowerBound) {
      reportOOB(checkerContext, state_precedesLowerBound, OOB_Precedes);
      return;
    }

    // Otherwise, assume the constraint of the lower bound.
    assert(state_withinLowerBound);
    state = state_withinLowerBound;
  }

  do {
    // CHECK UPPER BOUND: Is byteOffset >= extent(baseRegion)?  If so,
    // we are doing a load/store after the last valid offset.
    DefinedOrUnknownSVal extentVal =
      rawOffset.getRegion()->getExtent(svalBuilder);
    if (!extentVal.getAs<NonLoc>())
      break;

    SVal upperbound
      = svalBuilder.evalBinOpNN(state, BO_GE, rawOffset.getByteOffset(),
                                extentVal.castAs<NonLoc>(),
                                svalBuilder.getConditionType());

    Optional<NonLoc> upperboundToCheck = upperbound.getAs<NonLoc>();
    if (!upperboundToCheck)
      break;

    ProgramStateRef state_exceedsUpperBound, state_withinUpperBound;
    std::tie(state_exceedsUpperBound, state_withinUpperBound) =
      state->assume(*upperboundToCheck);

    // If we are under constrained and the index variables are tainted, report.
    if (state_exceedsUpperBound && state_withinUpperBound) {
      if (state->isTainted(rawOffset.getByteOffset()))
        reportOOB(checkerContext, state_exceedsUpperBound, OOB_Tainted);
        return;
    }

    // If we are constrained enough to definitely exceed the upper bound, report.
    if (state_exceedsUpperBound) {
      assert(!state_withinUpperBound);
      reportOOB(checkerContext, state_exceedsUpperBound, OOB_Excedes);
      return;
    }

    assert(state_withinUpperBound);
    state = state_withinUpperBound;
  }
  while (false);

  if (state != originalState)
    checkerContext.addTransition(state);
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
  case OOB_Tainted:
    os << "(index is tainted)";
    break;
  }

  checkerContext.emitReport(
      llvm::make_unique<BugReport>(*BT, os.str(), errorNode));
}

void RegionRawOffsetV2::dump() const {
  dumpToStream(llvm::errs());
}

void RegionRawOffsetV2::dumpToStream(raw_ostream &os) const {
  os << "raw_offset_v2{" << getRegion() << ',' << getByteOffset() << '}';
}


// Lazily computes a value to be used by 'computeOffset'.  If 'val'
// is unknown or undefined, we lazily substitute '0'.  Otherwise,
// return 'val'.
static inline SVal getValue(SVal val, SValBuilder &svalBuilder) {
  return val.getAs<UndefinedVal>() ? svalBuilder.makeArrayIndex(0) : val;
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
          offset = getValue(offset, svalBuilder);
          if (!offset.isUnknownOrUndef())
            return RegionRawOffsetV2(subReg, offset);
        }
        return RegionRawOffsetV2();
      }
      case MemRegion::ElementRegionKind: {
        const ElementRegion *elemReg = cast<ElementRegion>(region);
        SVal index = elemReg->getIndex();
        if (!index.getAs<NonLoc>())
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
