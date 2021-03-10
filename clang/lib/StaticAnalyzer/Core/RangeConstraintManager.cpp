//== RangeConstraintManager.cpp - Manage range constraints.------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines RangeConstraintManager, a class that tracks simple
//  equality and inequality constraints on symbolic values of ProgramState.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/JsonSupport.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValVisitor.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

// This class can be extended with other tables which will help to reason
// about ranges more precisely.
class OperatorRelationsTable {
  static_assert(BO_LT < BO_GT && BO_GT < BO_LE && BO_LE < BO_GE &&
                    BO_GE < BO_EQ && BO_EQ < BO_NE,
                "This class relies on operators order. Rework it otherwise.");

public:
  enum TriStateKind {
    False = 0,
    True,
    Unknown,
  };

private:
  // CmpOpTable holds states which represent the corresponding range for
  // branching an exploded graph. We can reason about the branch if there is
  // a previously known fact of the existence of a comparison expression with
  // operands used in the current expression.
  // E.g. assuming (x < y) is true that means (x != y) is surely true.
  // if (x previous_operation y)  // <    | !=      | >
  //   if (x operation y)         // !=   | >       | <
  //     tristate                 // True | Unknown | False
  //
  // CmpOpTable represents next:
  // __|< |> |<=|>=|==|!=|UnknownX2|
  // < |1 |0 |* |0 |0 |* |1        |
  // > |0 |1 |0 |* |0 |* |1        |
  // <=|1 |0 |1 |* |1 |* |0        |
  // >=|0 |1 |* |1 |1 |* |0        |
  // ==|0 |0 |* |* |1 |0 |1        |
  // !=|1 |1 |* |* |0 |1 |0        |
  //
  // Columns stands for a previous operator.
  // Rows stands for a current operator.
  // Each row has exactly two `Unknown` cases.
  // UnknownX2 means that both `Unknown` previous operators are met in code,
  // and there is a special column for that, for example:
  // if (x >= y)
  //   if (x != y)
  //     if (x <= y)
  //       False only
  static constexpr size_t CmpOpCount = BO_NE - BO_LT + 1;
  const TriStateKind CmpOpTable[CmpOpCount][CmpOpCount + 1] = {
      // <      >      <=     >=     ==     !=    UnknownX2
      {True, False, Unknown, False, False, Unknown, True}, // <
      {False, True, False, Unknown, False, Unknown, True}, // >
      {True, False, True, Unknown, True, Unknown, False},  // <=
      {False, True, Unknown, True, True, Unknown, False},  // >=
      {False, False, Unknown, Unknown, True, False, True}, // ==
      {True, True, Unknown, Unknown, False, True, False},  // !=
  };

  static size_t getIndexFromOp(BinaryOperatorKind OP) {
    return static_cast<size_t>(OP - BO_LT);
  }

public:
  constexpr size_t getCmpOpCount() const { return CmpOpCount; }

  static BinaryOperatorKind getOpFromIndex(size_t Index) {
    return static_cast<BinaryOperatorKind>(Index + BO_LT);
  }

  TriStateKind getCmpOpState(BinaryOperatorKind CurrentOP,
                             BinaryOperatorKind QueriedOP) const {
    return CmpOpTable[getIndexFromOp(CurrentOP)][getIndexFromOp(QueriedOP)];
  }

  TriStateKind getCmpOpStateForUnknownX2(BinaryOperatorKind CurrentOP) const {
    return CmpOpTable[getIndexFromOp(CurrentOP)][CmpOpCount];
  }
};
//===----------------------------------------------------------------------===//
//                           RangeSet implementation
//===----------------------------------------------------------------------===//

void RangeSet::IntersectInRange(BasicValueFactory &BV, Factory &F,
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
        newRanges =
            F.add(newRanges, Range(BV.getValue(Lower), BV.getValue(Upper)));
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

const llvm::APSInt &RangeSet::getMinValue() const {
  assert(!isEmpty());
  return begin()->From();
}

const llvm::APSInt &RangeSet::getMaxValue() const {
  assert(!isEmpty());
  // NOTE: It's a shame that we can't implement 'getMaxValue' without scanning
  //       the whole tree to get to the last element.
  //       llvm::ImmutableSet should support decrement for 'end' iterators
  //       or reverse order iteration.
  auto It = begin();
  for (auto End = end(); std::next(It) != End; ++It) {
  }
  return It->To();
}

bool RangeSet::pin(llvm::APSInt &Lower, llvm::APSInt &Upper) const {
  if (isEmpty()) {
    // This range is already infeasible.
    return false;
  }

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

// Returns a set containing the values in the receiving set, intersected with
// the closed range [Lower, Upper]. Unlike the Range type, this range uses
// modular arithmetic, corresponding to the common treatment of C integer
// overflow. Thus, if the Lower bound is greater than the Upper bound, the
// range is taken to wrap around. This is equivalent to taking the
// intersection with the two ranges [Min, Upper] and [Lower, Max],
// or, alternatively, /removing/ all integers between Upper and Lower.
RangeSet RangeSet::Intersect(BasicValueFactory &BV, Factory &F,
                             llvm::APSInt Lower, llvm::APSInt Upper) const {
  PrimRangeSet newRanges = F.getEmptySet();

  if (isEmpty() || !pin(Lower, Upper))
    return newRanges;

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

// Returns a set containing the values in the receiving set, intersected with
// the range set passed as parameter.
RangeSet RangeSet::Intersect(BasicValueFactory &BV, Factory &F,
                             const RangeSet &Other) const {
  PrimRangeSet newRanges = F.getEmptySet();

  for (iterator i = Other.begin(), e = Other.end(); i != e; ++i) {
    RangeSet newPiece = Intersect(BV, F, i->From(), i->To());
    for (iterator j = newPiece.begin(), ee = newPiece.end(); j != ee; ++j) {
      newRanges = F.add(newRanges, *j);
    }
  }

  return newRanges;
}

// Turn all [A, B] ranges to [-B, -A], when "-" is a C-like unary minus
// operation under the values of the type.
//
// We also handle MIN because applying unary minus to MIN does not change it.
// Example 1:
// char x = -128;        // -128 is a MIN value in a range of 'char'
// char y = -x;          // y: -128
// Example 2:
// unsigned char x = 0;  // 0 is a MIN value in a range of 'unsigned char'
// unsigned char y = -x; // y: 0
//
// And it makes us to separate the range
// like [MIN, N] to [MIN, MIN] U [-N,MAX].
// For instance, whole range is {-128..127} and subrange is [-128,-126],
// thus [-128,-127,-126,.....] negates to [-128,.....,126,127].
//
// Negate restores disrupted ranges on bounds,
// e.g. [MIN, B] => [MIN, MIN] U [-B, MAX] => [MIN, B].
RangeSet RangeSet::Negate(BasicValueFactory &BV, Factory &F) const {
  PrimRangeSet newRanges = F.getEmptySet();

  if (isEmpty())
    return newRanges;

  const llvm::APSInt sampleValue = getMinValue();
  const llvm::APSInt &MIN = BV.getMinValue(sampleValue);
  const llvm::APSInt &MAX = BV.getMaxValue(sampleValue);

  // Handle a special case for MIN value.
  iterator i = begin();
  const llvm::APSInt &from = i->From();
  const llvm::APSInt &to = i->To();
  if (from == MIN) {
    // If [from, to] are [MIN, MAX], then just return the same [MIN, MAX].
    if (to == MAX) {
      newRanges = ranges;
    } else {
      // Add separate range for the lowest value.
      newRanges = F.add(newRanges, Range(MIN, MIN));
      // Skip adding the second range in case when [from, to] are [MIN, MIN].
      if (to != MIN) {
        newRanges = F.add(newRanges, Range(BV.getValue(-to), MAX));
      }
    }
    // Skip the first range in the loop.
    ++i;
  }

  // Negate all other ranges.
  for (iterator e = end(); i != e; ++i) {
    // Negate int values.
    const llvm::APSInt &newFrom = BV.getValue(-i->To());
    const llvm::APSInt &newTo = BV.getValue(-i->From());
    // Add a negated range.
    newRanges = F.add(newRanges, Range(newFrom, newTo));
  }

  if (newRanges.isSingleton())
    return newRanges;

  // Try to find and unite next ranges:
  // [MIN, MIN] & [MIN + 1, N] => [MIN, N].
  iterator iter1 = newRanges.begin();
  iterator iter2 = std::next(iter1);

  if (iter1->To() == MIN && (iter2->From() - 1) == MIN) {
    const llvm::APSInt &to = iter2->To();
    // remove adjacent ranges
    newRanges = F.remove(newRanges, *iter1);
    newRanges = F.remove(newRanges, *newRanges.begin());
    // add united range
    newRanges = F.add(newRanges, Range(MIN, to));
  }

  return newRanges;
}

RangeSet RangeSet::Delete(BasicValueFactory &BV, Factory &F,
                          const llvm::APSInt &Point) const {
  llvm::APSInt Upper = Point;
  llvm::APSInt Lower = Point;

  ++Upper;
  --Lower;

  // Notice that the lower bound is greater than the upper bound.
  return Intersect(BV, F, Upper, Lower);
}

void RangeSet::print(raw_ostream &os) const {
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

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(SymbolSet, SymbolRef)

namespace {
class EquivalenceClass;
} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(ClassMap, SymbolRef, EquivalenceClass)
REGISTER_MAP_WITH_PROGRAMSTATE(ClassMembers, EquivalenceClass, SymbolSet)
REGISTER_MAP_WITH_PROGRAMSTATE(ConstraintRange, EquivalenceClass, RangeSet)

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(ClassSet, EquivalenceClass)
REGISTER_MAP_WITH_PROGRAMSTATE(DisequalityMap, EquivalenceClass, ClassSet)

namespace {
/// This class encapsulates a set of symbols equal to each other.
///
/// The main idea of the approach requiring such classes is in narrowing
/// and sharing constraints between symbols within the class.  Also we can
/// conclude that there is no practical need in storing constraints for
/// every member of the class separately.
///
/// Main terminology:
///
///   * "Equivalence class" is an object of this class, which can be efficiently
///     compared to other classes.  It represents the whole class without
///     storing the actual in it.  The members of the class however can be
///     retrieved from the state.
///
///   * "Class members" are the symbols corresponding to the class.  This means
///     that A == B for every member symbols A and B from the class.  Members of
///     each class are stored in the state.
///
///   * "Trivial class" is a class that has and ever had only one same symbol.
///
///   * "Merge operation" merges two classes into one.  It is the main operation
///     to produce non-trivial classes.
///     If, at some point, we can assume that two symbols from two distinct
///     classes are equal, we can merge these classes.
class EquivalenceClass : public llvm::FoldingSetNode {
public:
  /// Find equivalence class for the given symbol in the given state.
  LLVM_NODISCARD static inline EquivalenceClass find(ProgramStateRef State,
                                                     SymbolRef Sym);

  /// Merge classes for the given symbols and return a new state.
  LLVM_NODISCARD static inline ProgramStateRef
  merge(BasicValueFactory &BV, RangeSet::Factory &F, ProgramStateRef State,
        SymbolRef First, SymbolRef Second);
  // Merge this class with the given class and return a new state.
  LLVM_NODISCARD inline ProgramStateRef merge(BasicValueFactory &BV,
                                              RangeSet::Factory &F,
                                              ProgramStateRef State,
                                              EquivalenceClass Other);

  /// Return a set of class members for the given state.
  LLVM_NODISCARD inline SymbolSet getClassMembers(ProgramStateRef State);
  /// Return true if the current class is trivial in the given state.
  LLVM_NODISCARD inline bool isTrivial(ProgramStateRef State);
  /// Return true if the current class is trivial and its only member is dead.
  LLVM_NODISCARD inline bool isTriviallyDead(ProgramStateRef State,
                                             SymbolReaper &Reaper);

  LLVM_NODISCARD static inline ProgramStateRef
  markDisequal(BasicValueFactory &BV, RangeSet::Factory &F,
               ProgramStateRef State, SymbolRef First, SymbolRef Second);
  LLVM_NODISCARD static inline ProgramStateRef
  markDisequal(BasicValueFactory &BV, RangeSet::Factory &F,
               ProgramStateRef State, EquivalenceClass First,
               EquivalenceClass Second);
  LLVM_NODISCARD inline ProgramStateRef
  markDisequal(BasicValueFactory &BV, RangeSet::Factory &F,
               ProgramStateRef State, EquivalenceClass Other) const;
  LLVM_NODISCARD static inline ClassSet
  getDisequalClasses(ProgramStateRef State, SymbolRef Sym);
  LLVM_NODISCARD inline ClassSet
  getDisequalClasses(ProgramStateRef State) const;
  LLVM_NODISCARD inline ClassSet
  getDisequalClasses(DisequalityMapTy Map, ClassSet::Factory &Factory) const;

  LLVM_NODISCARD static inline Optional<bool>
  areEqual(ProgramStateRef State, SymbolRef First, SymbolRef Second);

  /// Check equivalence data for consistency.
  LLVM_NODISCARD LLVM_ATTRIBUTE_UNUSED static bool
  isClassDataConsistent(ProgramStateRef State);

  LLVM_NODISCARD QualType getType() const {
    return getRepresentativeSymbol()->getType();
  }

  EquivalenceClass() = delete;
  EquivalenceClass(const EquivalenceClass &) = default;
  EquivalenceClass &operator=(const EquivalenceClass &) = delete;
  EquivalenceClass(EquivalenceClass &&) = default;
  EquivalenceClass &operator=(EquivalenceClass &&) = delete;

  bool operator==(const EquivalenceClass &Other) const {
    return ID == Other.ID;
  }
  bool operator<(const EquivalenceClass &Other) const { return ID < Other.ID; }
  bool operator!=(const EquivalenceClass &Other) const {
    return !operator==(Other);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, uintptr_t CID) {
    ID.AddInteger(CID);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const { Profile(ID, this->ID); }

private:
  /* implicit */ EquivalenceClass(SymbolRef Sym)
      : ID(reinterpret_cast<uintptr_t>(Sym)) {}

  /// This function is intended to be used ONLY within the class.
  /// The fact that ID is a pointer to a symbol is an implementation detail
  /// and should stay that way.
  /// In the current implementation, we use it to retrieve the only member
  /// of the trivial class.
  SymbolRef getRepresentativeSymbol() const {
    return reinterpret_cast<SymbolRef>(ID);
  }
  static inline SymbolSet::Factory &getMembersFactory(ProgramStateRef State);

  inline ProgramStateRef mergeImpl(BasicValueFactory &BV, RangeSet::Factory &F,
                                   ProgramStateRef State, SymbolSet Members,
                                   EquivalenceClass Other,
                                   SymbolSet OtherMembers);
  static inline void
  addToDisequalityInfo(DisequalityMapTy &Info, ConstraintRangeTy &Constraints,
                       BasicValueFactory &BV, RangeSet::Factory &F,
                       ProgramStateRef State, EquivalenceClass First,
                       EquivalenceClass Second);

  /// This is a unique identifier of the class.
  uintptr_t ID;
};

//===----------------------------------------------------------------------===//
//                             Constraint functions
//===----------------------------------------------------------------------===//

LLVM_NODISCARD inline const RangeSet *getConstraint(ProgramStateRef State,
                                                    EquivalenceClass Class) {
  return State->get<ConstraintRange>(Class);
}

LLVM_NODISCARD inline const RangeSet *getConstraint(ProgramStateRef State,
                                                    SymbolRef Sym) {
  return getConstraint(State, EquivalenceClass::find(State, Sym));
}

//===----------------------------------------------------------------------===//
//                       Equality/diseqiality abstraction
//===----------------------------------------------------------------------===//

/// A small helper structure representing symbolic equality.
///
/// Equality check can have different forms (like a == b or a - b) and this
/// class encapsulates those away if the only thing the user wants to check -
/// whether it's equality/diseqiality or not and have an easy access to the
/// compared symbols.
struct EqualityInfo {
public:
  SymbolRef Left, Right;
  // true for equality and false for disequality.
  bool IsEquality = true;

  void invert() { IsEquality = !IsEquality; }
  /// Extract equality information from the given symbol and the constants.
  ///
  /// This function assumes the following expression Sym + Adjustment != Int.
  /// It is a default because the most widespread case of the equality check
  /// is (A == B) + 0 != 0.
  static Optional<EqualityInfo> extract(SymbolRef Sym, const llvm::APSInt &Int,
                                        const llvm::APSInt &Adjustment) {
    // As of now, the only equality form supported is Sym + 0 != 0.
    if (!Int.isNullValue() || !Adjustment.isNullValue())
      return llvm::None;

    return extract(Sym);
  }
  /// Extract equality information from the given symbol.
  static Optional<EqualityInfo> extract(SymbolRef Sym) {
    return EqualityExtractor().Visit(Sym);
  }

private:
  class EqualityExtractor
      : public SymExprVisitor<EqualityExtractor, Optional<EqualityInfo>> {
  public:
    Optional<EqualityInfo> VisitSymSymExpr(const SymSymExpr *Sym) const {
      switch (Sym->getOpcode()) {
      case BO_Sub:
        // This case is: A - B != 0 -> disequality check.
        return EqualityInfo{Sym->getLHS(), Sym->getRHS(), false};
      case BO_EQ:
        // This case is: A == B != 0 -> equality check.
        return EqualityInfo{Sym->getLHS(), Sym->getRHS(), true};
      case BO_NE:
        // This case is: A != B != 0 -> diseqiality check.
        return EqualityInfo{Sym->getLHS(), Sym->getRHS(), false};
      default:
        return llvm::None;
      }
    }
  };
};

//===----------------------------------------------------------------------===//
//                            Intersection functions
//===----------------------------------------------------------------------===//

template <class SecondTy, class... RestTy>
LLVM_NODISCARD inline RangeSet intersect(BasicValueFactory &BV,
                                         RangeSet::Factory &F, RangeSet Head,
                                         SecondTy Second, RestTy... Tail);

template <class... RangeTy> struct IntersectionTraits;

template <class... TailTy> struct IntersectionTraits<RangeSet, TailTy...> {
  // Found RangeSet, no need to check any further
  using Type = RangeSet;
};

template <> struct IntersectionTraits<> {
  // We ran out of types, and we didn't find any RangeSet, so the result should
  // be optional.
  using Type = Optional<RangeSet>;
};

template <class OptionalOrPointer, class... TailTy>
struct IntersectionTraits<OptionalOrPointer, TailTy...> {
  // If current type is Optional or a raw pointer, we should keep looking.
  using Type = typename IntersectionTraits<TailTy...>::Type;
};

template <class EndTy>
LLVM_NODISCARD inline EndTy intersect(BasicValueFactory &BV,
                                      RangeSet::Factory &F, EndTy End) {
  // If the list contains only RangeSet or Optional<RangeSet>, simply return
  // that range set.
  return End;
}

LLVM_NODISCARD LLVM_ATTRIBUTE_UNUSED inline Optional<RangeSet>
intersect(BasicValueFactory &BV, RangeSet::Factory &F, const RangeSet *End) {
  // This is an extraneous conversion from a raw pointer into Optional<RangeSet>
  if (End) {
    return *End;
  }
  return llvm::None;
}

template <class... RestTy>
LLVM_NODISCARD inline RangeSet intersect(BasicValueFactory &BV,
                                         RangeSet::Factory &F, RangeSet Head,
                                         RangeSet Second, RestTy... Tail) {
  // Here we call either the <RangeSet,RangeSet,...> or <RangeSet,...> version
  // of the function and can be sure that the result is RangeSet.
  return intersect(BV, F, Head.Intersect(BV, F, Second), Tail...);
}

template <class SecondTy, class... RestTy>
LLVM_NODISCARD inline RangeSet intersect(BasicValueFactory &BV,
                                         RangeSet::Factory &F, RangeSet Head,
                                         SecondTy Second, RestTy... Tail) {
  if (Second) {
    // Here we call the <RangeSet,RangeSet,...> version of the function...
    return intersect(BV, F, Head, *Second, Tail...);
  }
  // ...and here it is either <RangeSet,RangeSet,...> or <RangeSet,...>, which
  // means that the result is definitely RangeSet.
  return intersect(BV, F, Head, Tail...);
}

/// Main generic intersect function.
/// It intersects all of the given range sets.  If some of the given arguments
/// don't hold a range set (nullptr or llvm::None), the function will skip them.
///
/// Available representations for the arguments are:
///   * RangeSet
///   * Optional<RangeSet>
///   * RangeSet *
/// Pointer to a RangeSet is automatically assumed to be nullable and will get
/// checked as well as the optional version.  If this behaviour is undesired,
/// please dereference the pointer in the call.
///
/// Return type depends on the arguments' types.  If we can be sure in compile
/// time that there will be a range set as a result, the returning type is
/// simply RangeSet, in other cases we have to back off to Optional<RangeSet>.
///
/// Please, prefer optional range sets to raw pointers.  If the last argument is
/// a raw pointer and all previous arguments are None, it will cost one
/// additional check to convert RangeSet * into Optional<RangeSet>.
template <class HeadTy, class SecondTy, class... RestTy>
LLVM_NODISCARD inline
    typename IntersectionTraits<HeadTy, SecondTy, RestTy...>::Type
    intersect(BasicValueFactory &BV, RangeSet::Factory &F, HeadTy Head,
              SecondTy Second, RestTy... Tail) {
  if (Head) {
    return intersect(BV, F, *Head, Second, Tail...);
  }
  return intersect(BV, F, Second, Tail...);
}

//===----------------------------------------------------------------------===//
//                           Symbolic reasoning logic
//===----------------------------------------------------------------------===//

/// A little component aggregating all of the reasoning we have about
/// the ranges of symbolic expressions.
///
/// Even when we don't know the exact values of the operands, we still
/// can get a pretty good estimate of the result's range.
class SymbolicRangeInferrer
    : public SymExprVisitor<SymbolicRangeInferrer, RangeSet> {
public:
  template <class SourceType>
  static RangeSet inferRange(BasicValueFactory &BV, RangeSet::Factory &F,
                             ProgramStateRef State, SourceType Origin) {
    SymbolicRangeInferrer Inferrer(BV, F, State);
    return Inferrer.infer(Origin);
  }

  RangeSet VisitSymExpr(SymbolRef Sym) {
    // If we got to this function, the actual type of the symbolic
    // expression is not supported for advanced inference.
    // In this case, we simply backoff to the default "let's simply
    // infer the range from the expression's type".
    return infer(Sym->getType());
  }

  RangeSet VisitSymIntExpr(const SymIntExpr *Sym) {
    return VisitBinaryOperator(Sym);
  }

  RangeSet VisitIntSymExpr(const IntSymExpr *Sym) {
    return VisitBinaryOperator(Sym);
  }

  RangeSet VisitSymSymExpr(const SymSymExpr *Sym) {
    return VisitBinaryOperator(Sym);
  }

private:
  SymbolicRangeInferrer(BasicValueFactory &BV, RangeSet::Factory &F,
                        ProgramStateRef S)
      : ValueFactory(BV), RangeFactory(F), State(S) {}

  /// Infer range information from the given integer constant.
  ///
  /// It's not a real "inference", but is here for operating with
  /// sub-expressions in a more polymorphic manner.
  RangeSet inferAs(const llvm::APSInt &Val, QualType) {
    return {RangeFactory, Val};
  }

  /// Infer range information from symbol in the context of the given type.
  RangeSet inferAs(SymbolRef Sym, QualType DestType) {
    QualType ActualType = Sym->getType();
    // Check that we can reason about the symbol at all.
    if (ActualType->isIntegralOrEnumerationType() ||
        Loc::isLocType(ActualType)) {
      return infer(Sym);
    }
    // Otherwise, let's simply infer from the destination type.
    // We couldn't figure out nothing else about that expression.
    return infer(DestType);
  }

  RangeSet infer(SymbolRef Sym) {
    if (Optional<RangeSet> ConstraintBasedRange = intersect(
            ValueFactory, RangeFactory, getConstraint(State, Sym),
            // If Sym is a difference of symbols A - B, then maybe we have range
            // set stored for B - A.
            //
            // If we have range set stored for both A - B and B - A then
            // calculate the effective range set by intersecting the range set
            // for A - B and the negated range set of B - A.
            getRangeForNegatedSub(Sym), getRangeForEqualities(Sym))) {
      return *ConstraintBasedRange;
    }

    // If Sym is a comparison expression (except <=>),
    // find any other comparisons with the same operands.
    // See function description.
    if (Optional<RangeSet> CmpRangeSet = getRangeForComparisonSymbol(Sym)) {
      return *CmpRangeSet;
    }

    return Visit(Sym);
  }

  RangeSet infer(EquivalenceClass Class) {
    if (const RangeSet *AssociatedConstraint = getConstraint(State, Class))
      return *AssociatedConstraint;

    return infer(Class.getType());
  }

  /// Infer range information solely from the type.
  RangeSet infer(QualType T) {
    // Lazily generate a new RangeSet representing all possible values for the
    // given symbol type.
    RangeSet Result(RangeFactory, ValueFactory.getMinValue(T),
                    ValueFactory.getMaxValue(T));

    // References are known to be non-zero.
    if (T->isReferenceType())
      return assumeNonZero(Result, T);

    return Result;
  }

  template <class BinarySymExprTy>
  RangeSet VisitBinaryOperator(const BinarySymExprTy *Sym) {
    // TODO #1: VisitBinaryOperator implementation might not make a good
    // use of the inferred ranges.  In this case, we might be calculating
    // everything for nothing.  This being said, we should introduce some
    // sort of laziness mechanism here.
    //
    // TODO #2: We didn't go into the nested expressions before, so it
    // might cause us spending much more time doing the inference.
    // This can be a problem for deeply nested expressions that are
    // involved in conditions and get tested continuously.  We definitely
    // need to address this issue and introduce some sort of caching
    // in here.
    QualType ResultType = Sym->getType();
    return VisitBinaryOperator(inferAs(Sym->getLHS(), ResultType),
                               Sym->getOpcode(),
                               inferAs(Sym->getRHS(), ResultType), ResultType);
  }

  RangeSet VisitBinaryOperator(RangeSet LHS, BinaryOperator::Opcode Op,
                               RangeSet RHS, QualType T) {
    switch (Op) {
    case BO_Or:
      return VisitBinaryOperator<BO_Or>(LHS, RHS, T);
    case BO_And:
      return VisitBinaryOperator<BO_And>(LHS, RHS, T);
    case BO_Rem:
      return VisitBinaryOperator<BO_Rem>(LHS, RHS, T);
    default:
      return infer(T);
    }
  }

  //===----------------------------------------------------------------------===//
  //                         Ranges and operators
  //===----------------------------------------------------------------------===//

  /// Return a rough approximation of the given range set.
  ///
  /// For the range set:
  ///   { [x_0, y_0], [x_1, y_1], ... , [x_N, y_N] }
  /// it will return the range [x_0, y_N].
  static Range fillGaps(RangeSet Origin) {
    assert(!Origin.isEmpty());
    return {Origin.getMinValue(), Origin.getMaxValue()};
  }

  /// Try to convert given range into the given type.
  ///
  /// It will return llvm::None only when the trivial conversion is possible.
  llvm::Optional<Range> convert(const Range &Origin, APSIntType To) {
    if (To.testInRange(Origin.From(), false) != APSIntType::RTR_Within ||
        To.testInRange(Origin.To(), false) != APSIntType::RTR_Within) {
      return llvm::None;
    }
    return Range(ValueFactory.Convert(To, Origin.From()),
                 ValueFactory.Convert(To, Origin.To()));
  }

  template <BinaryOperator::Opcode Op>
  RangeSet VisitBinaryOperator(RangeSet LHS, RangeSet RHS, QualType T) {
    // We should propagate information about unfeasbility of one of the
    // operands to the resulting range.
    if (LHS.isEmpty() || RHS.isEmpty()) {
      return RangeFactory.getEmptySet();
    }

    Range CoarseLHS = fillGaps(LHS);
    Range CoarseRHS = fillGaps(RHS);

    APSIntType ResultType = ValueFactory.getAPSIntType(T);

    // We need to convert ranges to the resulting type, so we can compare values
    // and combine them in a meaningful (in terms of the given operation) way.
    auto ConvertedCoarseLHS = convert(CoarseLHS, ResultType);
    auto ConvertedCoarseRHS = convert(CoarseRHS, ResultType);

    // It is hard to reason about ranges when conversion changes
    // borders of the ranges.
    if (!ConvertedCoarseLHS || !ConvertedCoarseRHS) {
      return infer(T);
    }

    return VisitBinaryOperator<Op>(*ConvertedCoarseLHS, *ConvertedCoarseRHS, T);
  }

  template <BinaryOperator::Opcode Op>
  RangeSet VisitBinaryOperator(Range LHS, Range RHS, QualType T) {
    return infer(T);
  }

  /// Return a symmetrical range for the given range and type.
  ///
  /// If T is signed, return the smallest range [-x..x] that covers the original
  /// range, or [-min(T), max(T)] if the aforementioned symmetric range doesn't
  /// exist due to original range covering min(T)).
  ///
  /// If T is unsigned, return the smallest range [0..x] that covers the
  /// original range.
  Range getSymmetricalRange(Range Origin, QualType T) {
    APSIntType RangeType = ValueFactory.getAPSIntType(T);

    if (RangeType.isUnsigned()) {
      return Range(ValueFactory.getMinValue(RangeType), Origin.To());
    }

    if (Origin.From().isMinSignedValue()) {
      // If mini is a minimal signed value, absolute value of it is greater
      // than the maximal signed value.  In order to avoid these
      // complications, we simply return the whole range.
      return {ValueFactory.getMinValue(RangeType),
              ValueFactory.getMaxValue(RangeType)};
    }

    // At this point, we are sure that the type is signed and we can safely
    // use unary - operator.
    //
    // While calculating absolute maximum, we can use the following formula
    // because of these reasons:
    //   * If From >= 0 then To >= From and To >= -From.
    //     AbsMax == To == max(To, -From)
    //   * If To <= 0 then -From >= -To and -From >= From.
    //     AbsMax == -From == max(-From, To)
    //   * Otherwise, From <= 0, To >= 0, and
    //     AbsMax == max(abs(From), abs(To))
    llvm::APSInt AbsMax = std::max(-Origin.From(), Origin.To());

    // Intersection is guaranteed to be non-empty.
    return {ValueFactory.getValue(-AbsMax), ValueFactory.getValue(AbsMax)};
  }

  /// Return a range set subtracting zero from \p Domain.
  RangeSet assumeNonZero(RangeSet Domain, QualType T) {
    APSIntType IntType = ValueFactory.getAPSIntType(T);
    return Domain.Delete(ValueFactory, RangeFactory, IntType.getZeroValue());
  }

  // FIXME: Once SValBuilder supports unary minus, we should use SValBuilder to
  //        obtain the negated symbolic expression instead of constructing the
  //        symbol manually. This will allow us to support finding ranges of not
  //        only negated SymSymExpr-type expressions, but also of other, simpler
  //        expressions which we currently do not know how to negate.
  Optional<RangeSet> getRangeForNegatedSub(SymbolRef Sym) {
    if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(Sym)) {
      if (SSE->getOpcode() == BO_Sub) {
        QualType T = Sym->getType();

        // Do not negate unsigned ranges
        if (!T->isUnsignedIntegerOrEnumerationType() &&
            !T->isSignedIntegerOrEnumerationType())
          return llvm::None;

        SymbolManager &SymMgr = State->getSymbolManager();
        SymbolRef NegatedSym =
            SymMgr.getSymSymExpr(SSE->getRHS(), BO_Sub, SSE->getLHS(), T);

        if (const RangeSet *NegatedRange = getConstraint(State, NegatedSym)) {
          return NegatedRange->Negate(ValueFactory, RangeFactory);
        }
      }
    }
    return llvm::None;
  }

  // Returns ranges only for binary comparison operators (except <=>)
  // when left and right operands are symbolic values.
  // Finds any other comparisons with the same operands.
  // Then do logical calculations and refuse impossible branches.
  // E.g. (x < y) and (x > y) at the same time are impossible.
  // E.g. (x >= y) and (x != y) at the same time makes (x > y) true only.
  // E.g. (x == y) and (y == x) are just reversed but the same.
  // It covers all possible combinations (see CmpOpTable description).
  // Note that `x` and `y` can also stand for subexpressions,
  // not only for actual symbols.
  Optional<RangeSet> getRangeForComparisonSymbol(SymbolRef Sym) {
    const auto *SSE = dyn_cast<SymSymExpr>(Sym);
    if (!SSE)
      return llvm::None;

    BinaryOperatorKind CurrentOP = SSE->getOpcode();

    // We currently do not support <=> (C++20).
    if (!BinaryOperator::isComparisonOp(CurrentOP) || (CurrentOP == BO_Cmp))
      return llvm::None;

    static const OperatorRelationsTable CmpOpTable{};

    const SymExpr *LHS = SSE->getLHS();
    const SymExpr *RHS = SSE->getRHS();
    QualType T = SSE->getType();

    SymbolManager &SymMgr = State->getSymbolManager();

    int UnknownStates = 0;

    // Loop goes through all of the columns exept the last one ('UnknownX2').
    // We treat `UnknownX2` column separately at the end of the loop body.
    for (size_t i = 0; i < CmpOpTable.getCmpOpCount(); ++i) {

      // Let's find an expression e.g. (x < y).
      BinaryOperatorKind QueriedOP = OperatorRelationsTable::getOpFromIndex(i);
      const SymSymExpr *SymSym = SymMgr.getSymSymExpr(LHS, QueriedOP, RHS, T);
      const RangeSet *QueriedRangeSet = getConstraint(State, SymSym);

      // If ranges were not previously found,
      // try to find a reversed expression (y > x).
      if (!QueriedRangeSet) {
        const BinaryOperatorKind ROP =
            BinaryOperator::reverseComparisonOp(QueriedOP);
        SymSym = SymMgr.getSymSymExpr(RHS, ROP, LHS, T);
        QueriedRangeSet = getConstraint(State, SymSym);
      }

      if (!QueriedRangeSet || QueriedRangeSet->isEmpty())
        continue;

      const llvm::APSInt *ConcreteValue = QueriedRangeSet->getConcreteValue();
      const bool isInFalseBranch =
          ConcreteValue ? (*ConcreteValue == 0) : false;

      // If it is a false branch, we shall be guided by opposite operator,
      // because the table is made assuming we are in the true branch.
      // E.g. when (x <= y) is false, then (x > y) is true.
      if (isInFalseBranch)
        QueriedOP = BinaryOperator::negateComparisonOp(QueriedOP);

      OperatorRelationsTable::TriStateKind BranchState =
          CmpOpTable.getCmpOpState(CurrentOP, QueriedOP);

      if (BranchState == OperatorRelationsTable::Unknown) {
        if (++UnknownStates == 2)
          // If we met both Unknown states.
          // if (x <= y)    // assume true
          //   if (x != y)  // assume true
          //     if (x < y) // would be also true
          // Get a state from `UnknownX2` column.
          BranchState = CmpOpTable.getCmpOpStateForUnknownX2(CurrentOP);
        else
          continue;
      }

      return (BranchState == OperatorRelationsTable::True) ? getTrueRange(T)
                                                           : getFalseRange(T);
    }

    return llvm::None;
  }

  Optional<RangeSet> getRangeForEqualities(SymbolRef Sym) {
    Optional<EqualityInfo> Equality = EqualityInfo::extract(Sym);

    if (!Equality)
      return llvm::None;

    if (Optional<bool> AreEqual = EquivalenceClass::areEqual(
            State, Equality->Left, Equality->Right)) {
      if (*AreEqual == Equality->IsEquality) {
        return getTrueRange(Sym->getType());
      }
      return getFalseRange(Sym->getType());
    }

    return llvm::None;
  }

  RangeSet getTrueRange(QualType T) {
    RangeSet TypeRange = infer(T);
    return assumeNonZero(TypeRange, T);
  }

  RangeSet getFalseRange(QualType T) {
    const llvm::APSInt &Zero = ValueFactory.getValue(0, T);
    return RangeSet(RangeFactory, Zero);
  }

  BasicValueFactory &ValueFactory;
  RangeSet::Factory &RangeFactory;
  ProgramStateRef State;
};

//===----------------------------------------------------------------------===//
//               Range-based reasoning about symbolic operations
//===----------------------------------------------------------------------===//

template <>
RangeSet SymbolicRangeInferrer::VisitBinaryOperator<BO_Or>(Range LHS, Range RHS,
                                                           QualType T) {
  APSIntType ResultType = ValueFactory.getAPSIntType(T);
  llvm::APSInt Zero = ResultType.getZeroValue();

  bool IsLHSPositiveOrZero = LHS.From() >= Zero;
  bool IsRHSPositiveOrZero = RHS.From() >= Zero;

  bool IsLHSNegative = LHS.To() < Zero;
  bool IsRHSNegative = RHS.To() < Zero;

  // Check if both ranges have the same sign.
  if ((IsLHSPositiveOrZero && IsRHSPositiveOrZero) ||
      (IsLHSNegative && IsRHSNegative)) {
    // The result is definitely greater or equal than any of the operands.
    const llvm::APSInt &Min = std::max(LHS.From(), RHS.From());

    // We estimate maximal value for positives as the maximal value for the
    // given type.  For negatives, we estimate it with -1 (e.g. 0x11111111).
    //
    // TODO: We basically, limit the resulting range from below, but don't do
    //       anything with the upper bound.
    //
    //       For positive operands, it can be done as follows: for the upper
    //       bound of LHS and RHS we calculate the most significant bit set.
    //       Let's call it the N-th bit.  Then we can estimate the maximal
    //       number to be 2^(N+1)-1, i.e. the number with all the bits up to
    //       the N-th bit set.
    const llvm::APSInt &Max = IsLHSNegative
                                  ? ValueFactory.getValue(--Zero)
                                  : ValueFactory.getMaxValue(ResultType);

    return {RangeFactory, ValueFactory.getValue(Min), Max};
  }

  // Otherwise, let's check if at least one of the operands is negative.
  if (IsLHSNegative || IsRHSNegative) {
    // This means that the result is definitely negative as well.
    return {RangeFactory, ValueFactory.getMinValue(ResultType),
            ValueFactory.getValue(--Zero)};
  }

  RangeSet DefaultRange = infer(T);

  // It is pretty hard to reason about operands with different signs
  // (and especially with possibly different signs).  We simply check if it
  // can be zero.  In order to conclude that the result could not be zero,
  // at least one of the operands should be definitely not zero itself.
  if (!LHS.Includes(Zero) || !RHS.Includes(Zero)) {
    return assumeNonZero(DefaultRange, T);
  }

  // Nothing much else to do here.
  return DefaultRange;
}

template <>
RangeSet SymbolicRangeInferrer::VisitBinaryOperator<BO_And>(Range LHS,
                                                            Range RHS,
                                                            QualType T) {
  APSIntType ResultType = ValueFactory.getAPSIntType(T);
  llvm::APSInt Zero = ResultType.getZeroValue();

  bool IsLHSPositiveOrZero = LHS.From() >= Zero;
  bool IsRHSPositiveOrZero = RHS.From() >= Zero;

  bool IsLHSNegative = LHS.To() < Zero;
  bool IsRHSNegative = RHS.To() < Zero;

  // Check if both ranges have the same sign.
  if ((IsLHSPositiveOrZero && IsRHSPositiveOrZero) ||
      (IsLHSNegative && IsRHSNegative)) {
    // The result is definitely less or equal than any of the operands.
    const llvm::APSInt &Max = std::min(LHS.To(), RHS.To());

    // We conservatively estimate lower bound to be the smallest positive
    // or negative value corresponding to the sign of the operands.
    const llvm::APSInt &Min = IsLHSNegative
                                  ? ValueFactory.getMinValue(ResultType)
                                  : ValueFactory.getValue(Zero);

    return {RangeFactory, Min, Max};
  }

  // Otherwise, let's check if at least one of the operands is positive.
  if (IsLHSPositiveOrZero || IsRHSPositiveOrZero) {
    // This makes result definitely positive.
    //
    // We can also reason about a maximal value by finding the maximal
    // value of the positive operand.
    const llvm::APSInt &Max = IsLHSPositiveOrZero ? LHS.To() : RHS.To();

    // The minimal value on the other hand is much harder to reason about.
    // The only thing we know for sure is that the result is positive.
    return {RangeFactory, ValueFactory.getValue(Zero),
            ValueFactory.getValue(Max)};
  }

  // Nothing much else to do here.
  return infer(T);
}

template <>
RangeSet SymbolicRangeInferrer::VisitBinaryOperator<BO_Rem>(Range LHS,
                                                            Range RHS,
                                                            QualType T) {
  llvm::APSInt Zero = ValueFactory.getAPSIntType(T).getZeroValue();

  Range ConservativeRange = getSymmetricalRange(RHS, T);

  llvm::APSInt Max = ConservativeRange.To();
  llvm::APSInt Min = ConservativeRange.From();

  if (Max == Zero) {
    // It's an undefined behaviour to divide by 0 and it seems like we know
    // for sure that RHS is 0.  Let's say that the resulting range is
    // simply infeasible for that matter.
    return RangeFactory.getEmptySet();
  }

  // At this point, our conservative range is closed.  The result, however,
  // couldn't be greater than the RHS' maximal absolute value.  Because of
  // this reason, we turn the range into open (or half-open in case of
  // unsigned integers).
  //
  // While we operate on integer values, an open interval (a, b) can be easily
  // represented by the closed interval [a + 1, b - 1].  And this is exactly
  // what we do next.
  //
  // If we are dealing with unsigned case, we shouldn't move the lower bound.
  if (Min.isSigned()) {
    ++Min;
  }
  --Max;

  bool IsLHSPositiveOrZero = LHS.From() >= Zero;
  bool IsRHSPositiveOrZero = RHS.From() >= Zero;

  // Remainder operator results with negative operands is implementation
  // defined.  Positive cases are much easier to reason about though.
  if (IsLHSPositiveOrZero && IsRHSPositiveOrZero) {
    // If maximal value of LHS is less than maximal value of RHS,
    // the result won't get greater than LHS.To().
    Max = std::min(LHS.To(), Max);
    // We want to check if it is a situation similar to the following:
    //
    // <------------|---[  LHS  ]--------[  RHS  ]----->
    //  -INF        0                              +INF
    //
    // In this situation, we can conclude that (LHS / RHS) == 0 and
    // (LHS % RHS) == LHS.
    Min = LHS.To() < RHS.From() ? LHS.From() : Zero;
  }

  // Nevertheless, the symmetrical range for RHS is a conservative estimate
  // for any sign of either LHS, or RHS.
  return {RangeFactory, ValueFactory.getValue(Min), ValueFactory.getValue(Max)};
}

//===----------------------------------------------------------------------===//
//                  Constraint manager implementation details
//===----------------------------------------------------------------------===//

class RangeConstraintManager : public RangedConstraintManager {
public:
  RangeConstraintManager(ExprEngine *EE, SValBuilder &SVB)
      : RangedConstraintManager(EE, SVB) {}

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  bool haveEqualConstraints(ProgramStateRef S1,
                            ProgramStateRef S2) const override {
    // NOTE: ClassMembers are as simple as back pointers for ClassMap,
    //       so comparing constraint ranges and class maps should be
    //       sufficient.
    return S1->get<ConstraintRange>() == S2->get<ConstraintRange>() &&
           S1->get<ClassMap>() == S2->get<ClassMap>();
  }

  bool canReasonAbout(SVal X) const override;

  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  const llvm::APSInt *getSymVal(ProgramStateRef State,
                                SymbolRef Sym) const override;

  ProgramStateRef removeDeadBindings(ProgramStateRef State,
                                     SymbolReaper &SymReaper) override;

  void printJson(raw_ostream &Out, ProgramStateRef State, const char *NL = "\n",
                 unsigned int Space = 0, bool IsDot = false) const override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from RangedConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSymNE(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymEQ(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymLT(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymGT(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymLE(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymGE(ProgramStateRef State, SymbolRef Sym,
                              const llvm::APSInt &V,
                              const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymWithinInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) override;

  ProgramStateRef assumeSymOutsideInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) override;

private:
  RangeSet::Factory F;

  RangeSet getRange(ProgramStateRef State, SymbolRef Sym);
  RangeSet getRange(ProgramStateRef State, EquivalenceClass Class);

  RangeSet getSymLTRange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymGTRange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymLERange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymLERange(llvm::function_ref<RangeSet()> RS,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);
  RangeSet getSymGERange(ProgramStateRef St, SymbolRef Sym,
                         const llvm::APSInt &Int,
                         const llvm::APSInt &Adjustment);

  //===------------------------------------------------------------------===//
  // Equality tracking implementation
  //===------------------------------------------------------------------===//

  ProgramStateRef trackEQ(RangeSet NewConstraint, ProgramStateRef State,
                          SymbolRef Sym, const llvm::APSInt &Int,
                          const llvm::APSInt &Adjustment) {
    return track<true>(NewConstraint, State, Sym, Int, Adjustment);
  }

  ProgramStateRef trackNE(RangeSet NewConstraint, ProgramStateRef State,
                          SymbolRef Sym, const llvm::APSInt &Int,
                          const llvm::APSInt &Adjustment) {
    return track<false>(NewConstraint, State, Sym, Int, Adjustment);
  }

  template <bool EQ>
  ProgramStateRef track(RangeSet NewConstraint, ProgramStateRef State,
                        SymbolRef Sym, const llvm::APSInt &Int,
                        const llvm::APSInt &Adjustment) {
    if (NewConstraint.isEmpty())
      // This is an infeasible assumption.
      return nullptr;

    ProgramStateRef NewState = setConstraint(State, Sym, NewConstraint);
    if (auto Equality = EqualityInfo::extract(Sym, Int, Adjustment)) {
      // If the original assumption is not Sym + Adjustment !=/</> Int,
      // we should invert IsEquality flag.
      Equality->IsEquality = Equality->IsEquality != EQ;
      return track(NewState, *Equality);
    }

    return NewState;
  }

  ProgramStateRef track(ProgramStateRef State, EqualityInfo ToTrack) {
    if (ToTrack.IsEquality) {
      return trackEquality(State, ToTrack.Left, ToTrack.Right);
    }
    return trackDisequality(State, ToTrack.Left, ToTrack.Right);
  }

  ProgramStateRef trackDisequality(ProgramStateRef State, SymbolRef LHS,
                                   SymbolRef RHS) {
    return EquivalenceClass::markDisequal(getBasicVals(), F, State, LHS, RHS);
  }

  ProgramStateRef trackEquality(ProgramStateRef State, SymbolRef LHS,
                                SymbolRef RHS) {
    return EquivalenceClass::merge(getBasicVals(), F, State, LHS, RHS);
  }

  LLVM_NODISCARD LLVM_ATTRIBUTE_UNUSED static bool
  areFeasible(ConstraintRangeTy Constraints) {
    return llvm::none_of(
        Constraints,
        [](const std::pair<EquivalenceClass, RangeSet> &ClassConstraint) {
          return ClassConstraint.second.isEmpty();
        });
  }

  LLVM_NODISCARD ProgramStateRef setConstraint(ProgramStateRef State,
                                               EquivalenceClass Class,
                                               RangeSet Constraint) {
    ConstraintRangeTy Constraints = State->get<ConstraintRange>();
    ConstraintRangeTy::Factory &CF = State->get_context<ConstraintRange>();

    assert(!Constraint.isEmpty() && "New constraint should not be empty");

    // Add new constraint.
    Constraints = CF.add(Constraints, Class, Constraint);

    // There is a chance that we might need to update constraints for the
    // classes that are known to be disequal to Class.
    //
    // In order for this to be even possible, the new constraint should
    // be simply a constant because we can't reason about range disequalities.
    if (const llvm::APSInt *Point = Constraint.getConcreteValue())
      for (EquivalenceClass DisequalClass : Class.getDisequalClasses(State)) {
        RangeSet UpdatedConstraint =
            getRange(State, DisequalClass).Delete(getBasicVals(), F, *Point);

        // If we end up with at least one of the disequal classes to be
        // constrainted with an empty range-set, the state is infeasible.
        if (UpdatedConstraint.isEmpty())
          return nullptr;

        Constraints = CF.add(Constraints, DisequalClass, UpdatedConstraint);
      }

    assert(areFeasible(Constraints) && "Constraint manager shouldn't produce "
                                       "a state with infeasible constraints");

    return State->set<ConstraintRange>(Constraints);
  }

  LLVM_NODISCARD inline ProgramStateRef
  setConstraint(ProgramStateRef State, SymbolRef Sym, RangeSet Constraint) {
    return setConstraint(State, EquivalenceClass::find(State, Sym), Constraint);
  }
};

} // end anonymous namespace

std::unique_ptr<ConstraintManager>
ento::CreateRangeConstraintManager(ProgramStateManager &StMgr,
                                   ExprEngine *Eng) {
  return std::make_unique<RangeConstraintManager>(Eng, StMgr.getSValBuilder());
}

ConstraintMap ento::getConstraintMap(ProgramStateRef State) {
  ConstraintMap::Factory &F = State->get_context<ConstraintMap>();
  ConstraintMap Result = F.getEmptyMap();

  ConstraintRangeTy Constraints = State->get<ConstraintRange>();
  for (std::pair<EquivalenceClass, RangeSet> ClassConstraint : Constraints) {
    EquivalenceClass Class = ClassConstraint.first;
    SymbolSet ClassMembers = Class.getClassMembers(State);
    assert(!ClassMembers.isEmpty() &&
           "Class must always have at least one member!");

    SymbolRef Representative = *ClassMembers.begin();
    Result = F.add(Result, Representative, ClassConstraint.second);
  }

  return Result;
}

//===----------------------------------------------------------------------===//
//                     EqualityClass implementation details
//===----------------------------------------------------------------------===//

inline EquivalenceClass EquivalenceClass::find(ProgramStateRef State,
                                               SymbolRef Sym) {
  // We store far from all Symbol -> Class mappings
  if (const EquivalenceClass *NontrivialClass = State->get<ClassMap>(Sym))
    return *NontrivialClass;

  // This is a trivial class of Sym.
  return Sym;
}

inline ProgramStateRef EquivalenceClass::merge(BasicValueFactory &BV,
                                               RangeSet::Factory &F,
                                               ProgramStateRef State,
                                               SymbolRef First,
                                               SymbolRef Second) {
  EquivalenceClass FirstClass = find(State, First);
  EquivalenceClass SecondClass = find(State, Second);

  return FirstClass.merge(BV, F, State, SecondClass);
}

inline ProgramStateRef EquivalenceClass::merge(BasicValueFactory &BV,
                                               RangeSet::Factory &F,
                                               ProgramStateRef State,
                                               EquivalenceClass Other) {
  // It is already the same class.
  if (*this == Other)
    return State;

  // FIXME: As of now, we support only equivalence classes of the same type.
  //        This limitation is connected to the lack of explicit casts in
  //        our symbolic expression model.
  //
  //        That means that for `int x` and `char y` we don't distinguish
  //        between these two very different cases:
  //          * `x == y`
  //          * `(char)x == y`
  //
  //        The moment we introduce symbolic casts, this restriction can be
  //        lifted.
  if (getType() != Other.getType())
    return State;

  SymbolSet Members = getClassMembers(State);
  SymbolSet OtherMembers = Other.getClassMembers(State);

  // We estimate the size of the class by the height of tree containing
  // its members.  Merging is not a trivial operation, so it's easier to
  // merge the smaller class into the bigger one.
  if (Members.getHeight() >= OtherMembers.getHeight()) {
    return mergeImpl(BV, F, State, Members, Other, OtherMembers);
  } else {
    return Other.mergeImpl(BV, F, State, OtherMembers, *this, Members);
  }
}

inline ProgramStateRef
EquivalenceClass::mergeImpl(BasicValueFactory &ValueFactory,
                            RangeSet::Factory &RangeFactory,
                            ProgramStateRef State, SymbolSet MyMembers,
                            EquivalenceClass Other, SymbolSet OtherMembers) {
  // Essentially what we try to recreate here is some kind of union-find
  // data structure.  It does have certain limitations due to persistence
  // and the need to remove elements from classes.
  //
  // In this setting, EquialityClass object is the representative of the class
  // or the parent element.  ClassMap is a mapping of class members to their
  // parent. Unlike the union-find structure, they all point directly to the
  // class representative because we don't have an opportunity to actually do
  // path compression when dealing with immutability.  This means that we
  // compress paths every time we do merges.  It also means that we lose
  // the main amortized complexity benefit from the original data structure.
  ConstraintRangeTy Constraints = State->get<ConstraintRange>();
  ConstraintRangeTy::Factory &CRF = State->get_context<ConstraintRange>();

  // 1. If the merged classes have any constraints associated with them, we
  //    need to transfer them to the class we have left.
  //
  // Intersection here makes perfect sense because both of these constraints
  // must hold for the whole new class.
  if (Optional<RangeSet> NewClassConstraint =
          intersect(ValueFactory, RangeFactory, getConstraint(State, *this),
                    getConstraint(State, Other))) {
    // NOTE: Essentially, NewClassConstraint should NEVER be infeasible because
    //       range inferrer shouldn't generate ranges incompatible with
    //       equivalence classes. However, at the moment, due to imperfections
    //       in the solver, it is possible and the merge function can also
    //       return infeasible states aka null states.
    if (NewClassConstraint->isEmpty())
      // Infeasible state
      return nullptr;

    // No need in tracking constraints of a now-dissolved class.
    Constraints = CRF.remove(Constraints, Other);
    // Assign new constraints for this class.
    Constraints = CRF.add(Constraints, *this, *NewClassConstraint);

    State = State->set<ConstraintRange>(Constraints);
  }

  // 2. Get ALL equivalence-related maps
  ClassMapTy Classes = State->get<ClassMap>();
  ClassMapTy::Factory &CMF = State->get_context<ClassMap>();

  ClassMembersTy Members = State->get<ClassMembers>();
  ClassMembersTy::Factory &MF = State->get_context<ClassMembers>();

  DisequalityMapTy DisequalityInfo = State->get<DisequalityMap>();
  DisequalityMapTy::Factory &DF = State->get_context<DisequalityMap>();

  ClassSet::Factory &CF = State->get_context<ClassSet>();
  SymbolSet::Factory &F = getMembersFactory(State);

  // 2. Merge members of the Other class into the current class.
  SymbolSet NewClassMembers = MyMembers;
  for (SymbolRef Sym : OtherMembers) {
    NewClassMembers = F.add(NewClassMembers, Sym);
    // *this is now the class for all these new symbols.
    Classes = CMF.add(Classes, Sym, *this);
  }

  // 3. Adjust member mapping.
  //
  // No need in tracking members of a now-dissolved class.
  Members = MF.remove(Members, Other);
  // Now only the current class is mapped to all the symbols.
  Members = MF.add(Members, *this, NewClassMembers);

  // 4. Update disequality relations
  ClassSet DisequalToOther = Other.getDisequalClasses(DisequalityInfo, CF);
  if (!DisequalToOther.isEmpty()) {
    ClassSet DisequalToThis = getDisequalClasses(DisequalityInfo, CF);
    DisequalityInfo = DF.remove(DisequalityInfo, Other);

    for (EquivalenceClass DisequalClass : DisequalToOther) {
      DisequalToThis = CF.add(DisequalToThis, DisequalClass);

      // Disequality is a symmetric relation meaning that if
      // DisequalToOther not null then the set for DisequalClass is not
      // empty and has at least Other.
      ClassSet OriginalSetLinkedToOther =
          *DisequalityInfo.lookup(DisequalClass);

      // Other will be eliminated and we should replace it with the bigger
      // united class.
      ClassSet NewSet = CF.remove(OriginalSetLinkedToOther, Other);
      NewSet = CF.add(NewSet, *this);

      DisequalityInfo = DF.add(DisequalityInfo, DisequalClass, NewSet);
    }

    DisequalityInfo = DF.add(DisequalityInfo, *this, DisequalToThis);
    State = State->set<DisequalityMap>(DisequalityInfo);
  }

  // 5. Update the state
  State = State->set<ClassMap>(Classes);
  State = State->set<ClassMembers>(Members);

  return State;
}

inline SymbolSet::Factory &
EquivalenceClass::getMembersFactory(ProgramStateRef State) {
  return State->get_context<SymbolSet>();
}

SymbolSet EquivalenceClass::getClassMembers(ProgramStateRef State) {
  if (const SymbolSet *Members = State->get<ClassMembers>(*this))
    return *Members;

  // This class is trivial, so we need to construct a set
  // with just that one symbol from the class.
  SymbolSet::Factory &F = getMembersFactory(State);
  return F.add(F.getEmptySet(), getRepresentativeSymbol());
}

bool EquivalenceClass::isTrivial(ProgramStateRef State) {
  return State->get<ClassMembers>(*this) == nullptr;
}

bool EquivalenceClass::isTriviallyDead(ProgramStateRef State,
                                       SymbolReaper &Reaper) {
  return isTrivial(State) && Reaper.isDead(getRepresentativeSymbol());
}

inline ProgramStateRef EquivalenceClass::markDisequal(BasicValueFactory &VF,
                                                      RangeSet::Factory &RF,
                                                      ProgramStateRef State,
                                                      SymbolRef First,
                                                      SymbolRef Second) {
  return markDisequal(VF, RF, State, find(State, First), find(State, Second));
}

inline ProgramStateRef EquivalenceClass::markDisequal(BasicValueFactory &VF,
                                                      RangeSet::Factory &RF,
                                                      ProgramStateRef State,
                                                      EquivalenceClass First,
                                                      EquivalenceClass Second) {
  return First.markDisequal(VF, RF, State, Second);
}

inline ProgramStateRef
EquivalenceClass::markDisequal(BasicValueFactory &VF, RangeSet::Factory &RF,
                               ProgramStateRef State,
                               EquivalenceClass Other) const {
  // If we know that two classes are equal, we can only produce an infeasible
  // state.
  if (*this == Other) {
    return nullptr;
  }

  DisequalityMapTy DisequalityInfo = State->get<DisequalityMap>();
  ConstraintRangeTy Constraints = State->get<ConstraintRange>();

  // Disequality is a symmetric relation, so if we mark A as disequal to B,
  // we should also mark B as disequalt to A.
  addToDisequalityInfo(DisequalityInfo, Constraints, VF, RF, State, *this,
                       Other);
  addToDisequalityInfo(DisequalityInfo, Constraints, VF, RF, State, Other,
                       *this);

  State = State->set<DisequalityMap>(DisequalityInfo);
  State = State->set<ConstraintRange>(Constraints);

  return State;
}

inline void EquivalenceClass::addToDisequalityInfo(
    DisequalityMapTy &Info, ConstraintRangeTy &Constraints,
    BasicValueFactory &VF, RangeSet::Factory &RF, ProgramStateRef State,
    EquivalenceClass First, EquivalenceClass Second) {

  // 1. Get all of the required factories.
  DisequalityMapTy::Factory &F = State->get_context<DisequalityMap>();
  ClassSet::Factory &CF = State->get_context<ClassSet>();
  ConstraintRangeTy::Factory &CRF = State->get_context<ConstraintRange>();

  // 2. Add Second to the set of classes disequal to First.
  const ClassSet *CurrentSet = Info.lookup(First);
  ClassSet NewSet = CurrentSet ? *CurrentSet : CF.getEmptySet();
  NewSet = CF.add(NewSet, Second);

  Info = F.add(Info, First, NewSet);

  // 3. If Second is known to be a constant, we can delete this point
  //    from the constraint asociated with First.
  //
  //    So, if Second == 10, it means that First != 10.
  //    At the same time, the same logic does not apply to ranges.
  if (const RangeSet *SecondConstraint = Constraints.lookup(Second))
    if (const llvm::APSInt *Point = SecondConstraint->getConcreteValue()) {

      RangeSet FirstConstraint = SymbolicRangeInferrer::inferRange(
          VF, RF, State, First.getRepresentativeSymbol());

      FirstConstraint = FirstConstraint.Delete(VF, RF, *Point);
      Constraints = CRF.add(Constraints, First, FirstConstraint);
    }
}

inline Optional<bool> EquivalenceClass::areEqual(ProgramStateRef State,
                                                 SymbolRef FirstSym,
                                                 SymbolRef SecondSym) {
  EquivalenceClass First = find(State, FirstSym);
  EquivalenceClass Second = find(State, SecondSym);

  // The same equivalence class => symbols are equal.
  if (First == Second)
    return true;

  // Let's check if we know anything about these two classes being not equal to
  // each other.
  ClassSet DisequalToFirst = First.getDisequalClasses(State);
  if (DisequalToFirst.contains(Second))
    return false;

  // It is not clear.
  return llvm::None;
}

inline ClassSet EquivalenceClass::getDisequalClasses(ProgramStateRef State,
                                                     SymbolRef Sym) {
  return find(State, Sym).getDisequalClasses(State);
}

inline ClassSet
EquivalenceClass::getDisequalClasses(ProgramStateRef State) const {
  return getDisequalClasses(State->get<DisequalityMap>(),
                            State->get_context<ClassSet>());
}

inline ClassSet
EquivalenceClass::getDisequalClasses(DisequalityMapTy Map,
                                     ClassSet::Factory &Factory) const {
  if (const ClassSet *DisequalClasses = Map.lookup(*this))
    return *DisequalClasses;

  return Factory.getEmptySet();
}

bool EquivalenceClass::isClassDataConsistent(ProgramStateRef State) {
  ClassMembersTy Members = State->get<ClassMembers>();

  for (std::pair<EquivalenceClass, SymbolSet> ClassMembersPair : Members) {
    for (SymbolRef Member : ClassMembersPair.second) {
      // Every member of the class should have a mapping back to the class.
      if (find(State, Member) == ClassMembersPair.first) {
        continue;
      }

      return false;
    }
  }

  DisequalityMapTy Disequalities = State->get<DisequalityMap>();
  for (std::pair<EquivalenceClass, ClassSet> DisequalityInfo : Disequalities) {
    EquivalenceClass Class = DisequalityInfo.first;
    ClassSet DisequalClasses = DisequalityInfo.second;

    // There is no use in keeping empty sets in the map.
    if (DisequalClasses.isEmpty())
      return false;

    // Disequality is symmetrical, i.e. for every Class A and B that A != B,
    // B != A should also be true.
    for (EquivalenceClass DisequalClass : DisequalClasses) {
      const ClassSet *DisequalToDisequalClasses =
          Disequalities.lookup(DisequalClass);

      // It should be a set of at least one element: Class
      if (!DisequalToDisequalClasses ||
          !DisequalToDisequalClasses->contains(Class))
        return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//                    RangeConstraintManager implementation
//===----------------------------------------------------------------------===//

bool RangeConstraintManager::canReasonAbout(SVal X) const {
  Optional<nonloc::SymbolVal> SymVal = X.getAs<nonloc::SymbolVal>();
  if (SymVal && SymVal->isExpression()) {
    const SymExpr *SE = SymVal->getSymbol();

    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SE)) {
      switch (SIE->getOpcode()) {
      // We don't reason yet about bitwise-constraints on symbolic values.
      case BO_And:
      case BO_Or:
      case BO_Xor:
        return false;
      // We don't reason yet about these arithmetic constraints on
      // symbolic values.
      case BO_Mul:
      case BO_Div:
      case BO_Rem:
      case BO_Shl:
      case BO_Shr:
        return false;
      // All other cases.
      default:
        return true;
      }
    }

    if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(SE)) {
      // FIXME: Handle <=> here.
      if (BinaryOperator::isEqualityOp(SSE->getOpcode()) ||
          BinaryOperator::isRelationalOp(SSE->getOpcode())) {
        // We handle Loc <> Loc comparisons, but not (yet) NonLoc <> NonLoc.
        // We've recently started producing Loc <> NonLoc comparisons (that
        // result from casts of one of the operands between eg. intptr_t and
        // void *), but we can't reason about them yet.
        if (Loc::isLocType(SSE->getLHS()->getType())) {
          return Loc::isLocType(SSE->getRHS()->getType());
        }
      }
    }

    return false;
  }

  return true;
}

ConditionTruthVal RangeConstraintManager::checkNull(ProgramStateRef State,
                                                    SymbolRef Sym) {
  const RangeSet *Ranges = getConstraint(State, Sym);

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

const llvm::APSInt *RangeConstraintManager::getSymVal(ProgramStateRef St,
                                                      SymbolRef Sym) const {
  const RangeSet *T = getConstraint(St, Sym);
  return T ? T->getConcreteValue() : nullptr;
}

//===----------------------------------------------------------------------===//
//                Remove dead symbols from existing constraints
//===----------------------------------------------------------------------===//

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
ProgramStateRef
RangeConstraintManager::removeDeadBindings(ProgramStateRef State,
                                           SymbolReaper &SymReaper) {
  ClassMembersTy ClassMembersMap = State->get<ClassMembers>();
  ClassMembersTy NewClassMembersMap = ClassMembersMap;
  ClassMembersTy::Factory &EMFactory = State->get_context<ClassMembers>();
  SymbolSet::Factory &SetFactory = State->get_context<SymbolSet>();

  ConstraintRangeTy Constraints = State->get<ConstraintRange>();
  ConstraintRangeTy NewConstraints = Constraints;
  ConstraintRangeTy::Factory &ConstraintFactory =
      State->get_context<ConstraintRange>();

  ClassMapTy Map = State->get<ClassMap>();
  ClassMapTy NewMap = Map;
  ClassMapTy::Factory &ClassFactory = State->get_context<ClassMap>();

  DisequalityMapTy Disequalities = State->get<DisequalityMap>();
  DisequalityMapTy::Factory &DisequalityFactory =
      State->get_context<DisequalityMap>();
  ClassSet::Factory &ClassSetFactory = State->get_context<ClassSet>();

  bool ClassMapChanged = false;
  bool MembersMapChanged = false;
  bool ConstraintMapChanged = false;
  bool DisequalitiesChanged = false;

  auto removeDeadClass = [&](EquivalenceClass Class) {
    // Remove associated constraint ranges.
    Constraints = ConstraintFactory.remove(Constraints, Class);
    ConstraintMapChanged = true;

    // Update disequality information to not hold any information on the
    // removed class.
    ClassSet DisequalClasses =
        Class.getDisequalClasses(Disequalities, ClassSetFactory);
    if (!DisequalClasses.isEmpty()) {
      for (EquivalenceClass DisequalClass : DisequalClasses) {
        ClassSet DisequalToDisequalSet =
            DisequalClass.getDisequalClasses(Disequalities, ClassSetFactory);
        // DisequalToDisequalSet is guaranteed to be non-empty for consistent
        // disequality info.
        assert(!DisequalToDisequalSet.isEmpty());
        ClassSet NewSet = ClassSetFactory.remove(DisequalToDisequalSet, Class);

        // No need in keeping an empty set.
        if (NewSet.isEmpty()) {
          Disequalities =
              DisequalityFactory.remove(Disequalities, DisequalClass);
        } else {
          Disequalities =
              DisequalityFactory.add(Disequalities, DisequalClass, NewSet);
        }
      }
      // Remove the data for the class
      Disequalities = DisequalityFactory.remove(Disequalities, Class);
      DisequalitiesChanged = true;
    }
  };

  // 1. Let's see if dead symbols are trivial and have associated constraints.
  for (std::pair<EquivalenceClass, RangeSet> ClassConstraintPair :
       Constraints) {
    EquivalenceClass Class = ClassConstraintPair.first;
    if (Class.isTriviallyDead(State, SymReaper)) {
      // If this class is trivial, we can remove its constraints right away.
      removeDeadClass(Class);
    }
  }

  // 2. We don't need to track classes for dead symbols.
  for (std::pair<SymbolRef, EquivalenceClass> SymbolClassPair : Map) {
    SymbolRef Sym = SymbolClassPair.first;

    if (SymReaper.isDead(Sym)) {
      ClassMapChanged = true;
      NewMap = ClassFactory.remove(NewMap, Sym);
    }
  }

  // 3. Remove dead members from classes and remove dead non-trivial classes
  //    and their constraints.
  for (std::pair<EquivalenceClass, SymbolSet> ClassMembersPair :
       ClassMembersMap) {
    EquivalenceClass Class = ClassMembersPair.first;
    SymbolSet LiveMembers = ClassMembersPair.second;
    bool MembersChanged = false;

    for (SymbolRef Member : ClassMembersPair.second) {
      if (SymReaper.isDead(Member)) {
        MembersChanged = true;
        LiveMembers = SetFactory.remove(LiveMembers, Member);
      }
    }

    // Check if the class changed.
    if (!MembersChanged)
      continue;

    MembersMapChanged = true;

    if (LiveMembers.isEmpty()) {
      // The class is dead now, we need to wipe it out of the members map...
      NewClassMembersMap = EMFactory.remove(NewClassMembersMap, Class);

      // ...and remove all of its constraints.
      removeDeadClass(Class);
    } else {
      // We need to change the members associated with the class.
      NewClassMembersMap =
          EMFactory.add(NewClassMembersMap, Class, LiveMembers);
    }
  }

  // 4. Update the state with new maps.
  //
  // Here we try to be humble and update a map only if it really changed.
  if (ClassMapChanged)
    State = State->set<ClassMap>(NewMap);

  if (MembersMapChanged)
    State = State->set<ClassMembers>(NewClassMembersMap);

  if (ConstraintMapChanged)
    State = State->set<ConstraintRange>(Constraints);

  if (DisequalitiesChanged)
    State = State->set<DisequalityMap>(Disequalities);

  assert(EquivalenceClass::isClassDataConsistent(State));

  return State;
}

RangeSet RangeConstraintManager::getRange(ProgramStateRef State,
                                          SymbolRef Sym) {
  return SymbolicRangeInferrer::inferRange(getBasicVals(), F, State, Sym);
}

RangeSet RangeConstraintManager::getRange(ProgramStateRef State,
                                          EquivalenceClass Class) {
  return SymbolicRangeInferrer::inferRange(getBasicVals(), F, State, Class);
}

//===------------------------------------------------------------------------===
// assumeSymX methods: protected interface for RangeConstraintManager.
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

  llvm::APSInt Point = AdjustmentType.convert(Int) - Adjustment;

  RangeSet New = getRange(St, Sym).Delete(getBasicVals(), F, Point);

  return trackNE(New, St, Sym, Int, Adjustment);
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
  RangeSet New = getRange(St, Sym).Intersect(getBasicVals(), F, AdjInt, AdjInt);

  return trackEQ(New, St, Sym, Int, Adjustment);
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
    return getRange(St, Sym);
  }

  // Special case for Int == Min. This is always false.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (ComparisonVal == Min)
    return F.getEmptySet();

  llvm::APSInt Lower = Min - Adjustment;
  llvm::APSInt Upper = ComparisonVal - Adjustment;
  --Upper;

  return getRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymLT(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymLTRange(St, Sym, Int, Adjustment);
  return trackNE(New, St, Sym, Int, Adjustment);
}

RangeSet RangeConstraintManager::getSymGTRange(ProgramStateRef St,
                                               SymbolRef Sym,
                                               const llvm::APSInt &Int,
                                               const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return getRange(St, Sym);
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

  return getRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymGT(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGTRange(St, Sym, Int, Adjustment);
  return trackNE(New, St, Sym, Int, Adjustment);
}

RangeSet RangeConstraintManager::getSymGERange(ProgramStateRef St,
                                               SymbolRef Sym,
                                               const llvm::APSInt &Int,
                                               const llvm::APSInt &Adjustment) {
  // Before we do any real work, see if the value can even show up.
  APSIntType AdjustmentType(Adjustment);
  switch (AdjustmentType.testInRange(Int, true)) {
  case APSIntType::RTR_Below:
    return getRange(St, Sym);
  case APSIntType::RTR_Within:
    break;
  case APSIntType::RTR_Above:
    return F.getEmptySet();
  }

  // Special case for Int == Min. This is always feasible.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Min = AdjustmentType.getMinValue();
  if (ComparisonVal == Min)
    return getRange(St, Sym);

  llvm::APSInt Max = AdjustmentType.getMaxValue();
  llvm::APSInt Lower = ComparisonVal - Adjustment;
  llvm::APSInt Upper = Max - Adjustment;

  return getRange(St, Sym).Intersect(getBasicVals(), F, Lower, Upper);
}

ProgramStateRef
RangeConstraintManager::assumeSymGE(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGERange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : setConstraint(St, Sym, New);
}

RangeSet
RangeConstraintManager::getSymLERange(llvm::function_ref<RangeSet()> RS,
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
    return RS();
  }

  // Special case for Int == Max. This is always feasible.
  llvm::APSInt ComparisonVal = AdjustmentType.convert(Int);
  llvm::APSInt Max = AdjustmentType.getMaxValue();
  if (ComparisonVal == Max)
    return RS();

  llvm::APSInt Min = AdjustmentType.getMinValue();
  llvm::APSInt Lower = Min - Adjustment;
  llvm::APSInt Upper = ComparisonVal - Adjustment;

  return RS().Intersect(getBasicVals(), F, Lower, Upper);
}

RangeSet RangeConstraintManager::getSymLERange(ProgramStateRef St,
                                               SymbolRef Sym,
                                               const llvm::APSInt &Int,
                                               const llvm::APSInt &Adjustment) {
  return getSymLERange([&] { return getRange(St, Sym); }, Int, Adjustment);
}

ProgramStateRef
RangeConstraintManager::assumeSymLE(ProgramStateRef St, SymbolRef Sym,
                                    const llvm::APSInt &Int,
                                    const llvm::APSInt &Adjustment) {
  RangeSet New = getSymLERange(St, Sym, Int, Adjustment);
  return New.isEmpty() ? nullptr : setConstraint(St, Sym, New);
}

ProgramStateRef RangeConstraintManager::assumeSymWithinInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGERange(State, Sym, From, Adjustment);
  if (New.isEmpty())
    return nullptr;
  RangeSet Out = getSymLERange([&] { return New; }, To, Adjustment);
  return Out.isEmpty() ? nullptr : setConstraint(State, Sym, Out);
}

ProgramStateRef RangeConstraintManager::assumeSymOutsideInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet RangeLT = getSymLTRange(State, Sym, From, Adjustment);
  RangeSet RangeGT = getSymGTRange(State, Sym, To, Adjustment);
  RangeSet New(RangeLT.addRange(F, RangeGT));
  return New.isEmpty() ? nullptr : setConstraint(State, Sym, New);
}

//===----------------------------------------------------------------------===//
// Pretty-printing.
//===----------------------------------------------------------------------===//

void RangeConstraintManager::printJson(raw_ostream &Out, ProgramStateRef State,
                                       const char *NL, unsigned int Space,
                                       bool IsDot) const {
  ConstraintRangeTy Constraints = State->get<ConstraintRange>();

  Indent(Out, Space, IsDot) << "\"constraints\": ";
  if (Constraints.isEmpty()) {
    Out << "null," << NL;
    return;
  }

  ++Space;
  Out << '[' << NL;
  bool First = true;
  for (std::pair<EquivalenceClass, RangeSet> P : Constraints) {
    SymbolSet ClassMembers = P.first.getClassMembers(State);

    // We can print the same constraint for every class member.
    for (SymbolRef ClassMember : ClassMembers) {
      if (First) {
        First = false;
      } else {
        Out << ',';
        Out << NL;
      }
      Indent(Out, Space, IsDot)
          << "{ \"symbol\": \"" << ClassMember << "\", \"range\": \"";
      P.second.print(Out);
      Out << "\" }";
    }
  }
  Out << NL;

  --Space;
  Indent(Out, Space, IsDot) << "]," << NL;
}
