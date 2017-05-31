//== RangeConstraintManager.cpp - Manage range constraints.------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines RangeConstraintManager, a class that tracks simple
//  equality and inequality constraints on symbolic values of ProgramState.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

/// A Range represents the closed range [from, to].  The caller must
/// guarantee that from <= to.  Note that Range is immutable, so as not
/// to subvert RangeSet's immutability.
namespace {
class Range : public std::pair<const llvm::APSInt*,
                                                const llvm::APSInt*> {
public:
  Range(const llvm::APSInt &from, const llvm::APSInt &to)
    : std::pair<const llvm::APSInt*, const llvm::APSInt*>(&from, &to) {
    assert(from <= to);
  }
  bool Includes(const llvm::APSInt &v) const {
    return *first <= v && v <= *second;
  }
  const llvm::APSInt &From() const {
    return *first;
  }
  const llvm::APSInt &To() const {
    return *second;
  }
  const llvm::APSInt *getConcreteValue() const {
    return &From() == &To() ? &From() : nullptr;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(&From());
    ID.AddPointer(&To());
  }
};


class RangeTrait : public llvm::ImutContainerInfo<Range> {
public:
  // When comparing if one Range is less than another, we should compare
  // the actual APSInt values instead of their pointers.  This keeps the order
  // consistent (instead of comparing by pointer values) and can potentially
  // be used to speed up some of the operations in RangeSet.
  static inline bool isLess(key_type_ref lhs, key_type_ref rhs) {
    return *lhs.first < *rhs.first || (!(*rhs.first < *lhs.first) &&
                                       *lhs.second < *rhs.second);
  }
};

/// RangeSet contains a set of ranges. If the set is empty, then
///  there the value of a symbol is overly constrained and there are no
///  possible values for that symbol.
class RangeSet {
  typedef llvm::ImmutableSet<Range, RangeTrait> PrimRangeSet;
  PrimRangeSet ranges; // no need to make const, since it is an
                       // ImmutableSet - this allows default operator=
                       // to work.
public:
  typedef PrimRangeSet::Factory Factory;
  typedef PrimRangeSet::iterator iterator;

  RangeSet(PrimRangeSet RS) : ranges(RS) {}

  /// Create a new set with all ranges of this set and RS.
  /// Possible intersections are not checked here.
  RangeSet addRange(Factory &F, const RangeSet &RS) {
    PrimRangeSet Ranges(RS.ranges);
    for (const auto &range : ranges)
      Ranges = F.add(Ranges, range);
    return RangeSet(Ranges);
  }

  iterator begin() const { return ranges.begin(); }
  iterator end() const { return ranges.end(); }

  bool isEmpty() const { return ranges.isEmpty(); }

  /// Construct a new RangeSet representing '{ [from, to] }'.
  RangeSet(Factory &F, const llvm::APSInt &from, const llvm::APSInt &to)
    : ranges(F.add(F.getEmptySet(), Range(from, to))) {}

  /// Profile - Generates a hash profile of this RangeSet for use
  ///  by FoldingSet.
  void Profile(llvm::FoldingSetNodeID &ID) const { ranges.Profile(ID); }

  /// getConcreteValue - If a symbol is contrained to equal a specific integer
  ///  constant then this method returns that value.  Otherwise, it returns
  ///  NULL.
  const llvm::APSInt* getConcreteValue() const {
    return ranges.isSingleton() ? ranges.begin()->getConcreteValue() : nullptr;
  }

private:
  void IntersectInRange(BasicValueFactory &BV, Factory &F,
                        const llvm::APSInt &Lower,
                        const llvm::APSInt &Upper,
                        PrimRangeSet &newRanges,
                        PrimRangeSet::iterator &i,
                        PrimRangeSet::iterator &e) const {
    // There are six cases for each range R in the set:
    //   1. R is entirely before the intersection range.
    //   2. R is entirely after the intersection range.
    //   3. R contains the entire intersection range.
    //   4. R starts before the intersection range and ends in the middle.
    //   5. R starts in the middle of the intersection range and ends after it.
    //   6. R is entirely contained in the intersection range.
    // These correspond to each of the conditions below.
    for (/* i = begin(), e = end() */; i != e; ++i) {
      if (i->To() < Lower) {
        continue;
      }
      if (i->From() > Upper) {
        break;
      }

      if (i->Includes(Lower)) {
        if (i->Includes(Upper)) {
          newRanges = F.add(newRanges, Range(BV.getValue(Lower),
                                             BV.getValue(Upper)));
          break;
        } else
          newRanges = F.add(newRanges, Range(BV.getValue(Lower), i->To()));
      } else {
        if (i->Includes(Upper)) {
          newRanges = F.add(newRanges, Range(i->From(), BV.getValue(Upper)));
          break;
        } else
          newRanges = F.add(newRanges, *i);
      }
    }
  }

  const llvm::APSInt &getMinValue() const {
    assert(!isEmpty());
    return ranges.begin()->From();
  }

  bool pin(llvm::APSInt &Lower, llvm::APSInt &Upper) const {
    // This function has nine cases, the cartesian product of range-testing
    // both the upper and lower bounds against the symbol's type.
    // Each case requires a different pinning operation.
    // The function returns false if the described range is entirely outside
    // the range of values for the associated symbol.
    APSIntType Type(getMinValue());
    APSIntType::RangeTestResultKind LowerTest = Type.testInRange(Lower, true);
    APSIntType::RangeTestResultKind UpperTest = Type.testInRange(Upper, true);

    switch (LowerTest) {
    case APSIntType::RTR_Below:
      switch (UpperTest) {
      case APSIntType::RTR_Below:
        // The entire range is outside the symbol's set of possible values.
        // If this is a conventionally-ordered range, the state is infeasible.
        if (Lower <= Upper)
          return false;

        // However, if the range wraps around, it spans all possible values.
        Lower = Type.getMinValue();
        Upper = Type.getMaxValue();
        break;
      case APSIntType::RTR_Within:
        // The range starts below what's possible but ends within it. Pin.
        Lower = Type.getMinValue();
        Type.apply(Upper);
        break;
      case APSIntType::RTR_Above:
        // The range spans all possible values for the symbol. Pin.
        Lower = Type.getMinValue();
        Upper = Type.getMaxValue();
        break;
      }
      break;
    case APSIntType::RTR_Within:
      switch (UpperTest) {
      case APSIntType::RTR_Below:
        // The range wraps around, but all lower values are not possible.
        Type.apply(Lower);
        Upper = Type.getMaxValue();
        break;
      case APSIntType::RTR_Within:
        // The range may or may not wrap around, but both limits are valid.
        Type.apply(Lower);
        Type.apply(Upper);
        break;
      case APSIntType::RTR_Above:
        // The range starts within what's possible but ends above it. Pin.
        Type.apply(Lower);
        Upper = Type.getMaxValue();
        break;
      }
      break;
    case APSIntType::RTR_Above:
      switch (UpperTest) {
      case APSIntType::RTR_Below:
        // The range wraps but is outside the symbol's set of possible values.
        return false;
      case APSIntType::RTR_Within:
        // The range starts above what's possible but ends within it (wrap).
        Lower = Type.getMinValue();
        Type.apply(Upper);
        break;
      case APSIntType::RTR_Above:
        // The entire range is outside the symbol's set of possible values.
        // If this is a conventionally-ordered range, the state is infeasible.
        if (Lower <= Upper)
          return false;

        // However, if the range wraps around, it spans all possible values.
        Lower = Type.getMinValue();
        Upper = Type.getMaxValue();
        break;
      }
      break;
    }

    return true;
  }

public:
  // Returns a set containing the values in the receiving set, intersected with
  // the closed range [Lower, Upper]. Unlike the Range type, this range uses
  // modular arithmetic, corresponding to the common treatment of C integer
  // overflow. Thus, if the Lower bound is greater than the Upper bound, the
  // range is taken to wrap around. This is equivalent to taking the
  // intersection with the two ranges [Min, Upper] and [Lower, Max],
  // or, alternatively, /removing/ all integers between Upper and Lower.
  RangeSet Intersect(BasicValueFactory &BV, Factory &F,
                     llvm::APSInt Lower, llvm::APSInt Upper) const {
    if (!pin(Lower, Upper))
      return F.getEmptySet();

    PrimRangeSet newRanges = F.getEmptySet();

    PrimRangeSet::iterator i = begin(), e = end();
    if (Lower <= Upper)
      IntersectInRange(BV, F, Lower, Upper, newRanges, i, e);
    else {
      // The order of the next two statements is important!
      // IntersectInRange() does not reset the iteration state for i and e.
      // Therefore, the lower range most be handled first.
      IntersectInRange(BV, F, BV.getMinValue(Upper), Upper, newRanges, i, e);
      IntersectInRange(BV, F, Lower, BV.getMaxValue(Lower), newRanges, i, e);
    }

    return newRanges;
  }

  void print(raw_ostream &os) const {
    bool isFirst = true;
    os << "{ ";
    for (iterator i = begin(), e = end(); i != e; ++i) {
      if (isFirst)
        isFirst = false;
      else
        os << ", ";

      os << '[' << i->From().toString(10) << ", " << i->To().toString(10)
         << ']';
    }
    os << " }";
  }

  bool operator==(const RangeSet &other) const {
    return ranges == other.ranges;
  }
};
} // end anonymous namespace

REGISTER_TRAIT_WITH_PROGRAMSTATE(ConstraintRange,
                                 CLANG_ENTO_PROGRAMSTATE_MAP(SymbolRef,
                                                             RangeSet))

namespace {
class RangeConstraintManager : public SimpleConstraintManager{
  RangeSet GetRange(ProgramStateRef state, SymbolRef sym);
public:
  RangeConstraintManager(SubEngine *subengine, SValBuilder &SVB)
    : SimpleConstraintManager(subengine, SVB) {}

  ProgramStateRef assumeSymNE(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymEQ(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymLT(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymGT(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymGE(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymLE(ProgramStateRef state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment) override;

  ProgramStateRef assumeSymbolWithinInclusiveRange(
        ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
        const llvm::APSInt &To, const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymbolOutOfInclusiveRange(
        ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
        const llvm::APSInt &To, const llvm::APSInt &Adjustment) override;

  const llvm::APSInt* getSymVal(ProgramStateRef St,
                                SymbolRef sym) const override;
  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  ProgramStateRef removeDeadBindings(ProgramStateRef St,
                                     SymbolReaper& SymReaper) override;

  void print(ProgramStateRef St, raw_ostream &Out,
             const char* nl, const char *sep) override;

private:
  RangeSet::Factory F;
  RangeSet getSymLTRange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymGTRange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymLERange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymLERange(const RangeSet &RS, const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymGERange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
};

} // end anonymous namespace

std::unique_ptr<ConstraintManager>
ento::CreateRangeConstraintManager(ProgramStateManager &StMgr, SubEngine *Eng) {
  return llvm::make_unique<RangeConstraintManager>(Eng, StMgr.getSValBuilder());
}

const llvm::APSInt* RangeConstraintManager::getSymVal(ProgramStateRef St,
                                                      SymbolRef sym) const {
  const ConstraintRangeTy::data_type *T = St->get<ConstraintRange>(sym);
  return T ? T->getConcreteValue() : nullptr;
}

ConditionTruthVal RangeConstraintManager::checkNull(ProgramStateRef State,
                                                    SymbolRef Sym) {
  const RangeSet *Ranges = State->get<ConstraintRange>(Sym);

  // If we don't have any information about this symbol, it's underconstrained.
  if (!Ranges)
    return ConditionTruthVal();

  // If we have a concrete value, see if it's zero.
  if (const llvm::APSInt *Value = Ranges->getConcreteValue())
    return *Value == 0;

  BasicValueFactory &BV = getBasicVals();
  APSIntType IntType = BV.getAPSIntType(Sym->getType());
  llvm::APSInt Zero = IntType.getZeroValue();

  // Check if zero is in the set of possible values.
  if (Ranges->Intersect(BV, F, Zero, Zero).isEmpty())
    return false;

  // Zero is a possible value, but it is not the /only/ possible value.
  return ConditionTruthVal();
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
ProgramStateRef
RangeConstraintManager::removeDeadBindings(ProgramStateRef state,
                                           SymbolReaper& SymReaper) {

  ConstraintRangeTy CR = state->get<ConstraintRange>();
  ConstraintRangeTy::Factory& CRFactory = state->get_context<ConstraintRange>();

  for (ConstraintRangeTy::iterator I = CR.begin(), E = CR.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym))
      CR = CRFactory.remove(CR, sym);
  }

  return state->set<ConstraintRange>(CR);
}

RangeSet
RangeConstraintManager::GetRange(ProgramStateRef state, SymbolRef sym) {
  if (ConstraintRangeTy::data_type* V = state->get<ConstraintRange>(sym))
    return *V;

  // Lazily generate a new RangeSet representing all possible values for the
  // given symbol type.
  BasicValueFactory &BV = getBasicVals();
  QualType T = sym->getType();

  RangeSet Result(F, BV.getMinValue(T), BV.getMaxValue(T));

  // Special case: references are known to be non-zero.
  if (T->isReferenceType()) {
    APSIntType IntType = BV.getAPSIntType(T);
    Result = Result.Intersect(BV, F, ++IntType.getZeroValue(),
                                     --IntType.getZeroValue());
  }

  return Result;
}

//===------------------------------------------------------------------------===
// assumeSymX methods: public interface for RangeConstraintManager.
//===------------------------------------------------------------------------===/

// The syntax for ranges below is mathematical, using [x, y] for closed ranges
// and (x, y) for open ranges. These ranges are modular, corresponding with
// a common treatment of C integer overflow. This means that these methods
// do not have to worry about overflow; RangeSet::Intersect can handle such a
// "wraparound" range.
// As an example, the range [UINT_MAX-1, 3) contains five values: UINT_MAX-1,
// UINT_MAX, 0, 1, and 2.

ProgramStateRef
RangeConstraintManager::assumeSymNE(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  if (AdjustmentType.testInRange(Int, true) != APSIntType::RTR_Within)
    return St;

  llvm::APSInt Lower = AdjustmentType.convert(Int) - Adjustment;
  llvm::APSInt Upper = Lower;
  --Lower;
  ++Upper;

  // [Int-Adjustment+1, Int-Adjustment-1]
  // Notice that the lower bound is greater than the upper bound.
  RangeSet New = GetRange(St, Sym).Intersect(getBasicVals(), F, Upper, Lower);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

ProgramStateRef
RangeConstraintManager::assumeSymEQ(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  if (AdjustmentType.testInRange(Int, true) != APSIntType::RTR_Within)
    return nullptr;

  // [Int-Adjustment, Int-Adjustment]
  llvm::APSInt AdjInt = AdjustmentType.convert(Int) - Adjustment;
  RangeSet New = GetRange(St, Sym).Intersect(getBasicVals(), F, AdjInt, AdjInt);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

RangeSet RangeConstraintManager::getSymLTRange(ProgramStateRef St,
                                               SymbolRef Sym,
                                               const llvm::APSInt &Int,
                                               const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return F.getEmptySet();
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return GetRange(St, Sym);
  }

  // Special case for Int == Min. This is always false.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (ComparisonVal == Min)
    return F.getEmptySet();

  llvm::APSInt Lower = Min - Adjustment;
  llvm::APSInt Upper = ComparisonVal - Adjustment;
  --Upper;

  return GetRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymLT(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymLTRange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

RangeSet
RangeConstraintManager::getSymGTRange(ProgramStateRef St, SymbolRef Sym,
                                      const llvm::APSInt &Int,
                                      const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return GetRange(St, Sym);
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return F.getEmptySet();
  }

  // Special case for Int == Max. This is always false.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (ComparisonVal == Max)
    return F.getEmptySet();

  llvm::APSInt Lower = ComparisonVal - Adjustment;
  llvm::APSInt Upper = Max - Adjustment;
  ++Lower;

  return GetRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymGT(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGTRange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

RangeSet
RangeConstraintManager::getSymGERange(ProgramStateRef St, SymbolRef Sym,
                                      const llvm::APSInt &Int,
                                      const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return GetRange(St, Sym);
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return F.getEmptySet();
  }

  // Special case for Int == Min. This is always feasible.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (ComparisonVal == Min)
    return GetRange(St, Sym);

  llvm::APSInt Max = AdjustmentType.getMaxValue();
  llvm::APSInt Lower = ComparisonVal - Adjustment;
  llvm::APSInt Upper = Max - Adjustment;

  return GetRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymGE(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGERange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

RangeSet
RangeConstraintManager::getSymLERange(const RangeSet &RS,
                                      const llvm::APSInt &Int,
                                      const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return F.getEmptySet();
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return RS;
  }

  // Special case for Int == Max. This is always feasible.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (ComparisonVal == Max)
    return RS;

  llvm::APSInt Min = AdjustmentType.getMinValue();
  llvm::APSInt Lower = Min - Adjustment;
  llvm::APSInt Upper = ComparisonVal - Adjustment;

  return RS.Intersect(getBasicVals(), F, Lower, Upper);
}

RangeSet
RangeConstraintManager::getSymLERange(ProgramStateRef St, SymbolRef Sym,
                                      const llvm::APSInt &Int,
                                      const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return F.getEmptySet();
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return GetRange(St, Sym);
  }

  // Special case for Int == Max. This is always feasible.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (ComparisonVal == Max)
    return GetRange(St, Sym);

  llvm::APSInt Min = AdjustmentType.getMinValue();
  llvm::APSInt Lower = Min - Adjustment;
  llvm::APSInt Upper = ComparisonVal - Adjustment;

  return GetRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymLE(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymLERange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

ProgramStateRef
RangeConstraintManager::assumeSymbolWithinInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGERange(State, Sym, From, Adjustment);
  if (New.isEmpty())
    return nullptr;
  New = getSymLERange(New, To, Adjustment);
  return New.isEmpty() ? nullptr : State->set<ConstraintRange>(Sym, New);
}

ProgramStateRef
RangeConstraintManager::assumeSymbolOutOfInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet RangeLT = getSymLTRange(State, Sym, From, Adjustment);
  RangeSet RangeGT = getSymGTRange(State, Sym, To, Adjustment);
  RangeSet New(RangeLT.addRange(F, RangeGT));
  return New.isEmpty() ? nullptr : State->set<ConstraintRange>(Sym, New);
}

//===------------------------------------------------------------------------===
// Pretty-printing.
//===------------------------------------------------------------------------===/

void RangeConstraintManager::print(ProgramStateRef St, raw_ostream &Out,
                                   const char* nl, const char *sep) {

  ConstraintRangeTy Ranges = St->get<ConstraintRange>();

  if (Ranges.isEmpty()) {
    Out << nl << sep << "Ranges are empty." << nl;
    return;
  }

  Out << nl << sep << "Ranges of symbol values:";
  for (ConstraintRangeTy::iterator I=Ranges.begin(), E=Ranges.end(); I!=E; ++I){
    Out << nl << ' ' << I.getKey() << " : ";
    I.getData().print(Out);
  }
  Out << nl;
}
