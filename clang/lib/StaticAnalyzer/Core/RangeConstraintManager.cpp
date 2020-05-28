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
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

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
  return ranges.begin()->From();
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

namespace {

/// A little component aggregating all of the reasoning we have about
/// the ranges of symbolic expressions.
///
/// Even when we don't know the exact values of the operands, we still
/// can get a pretty good estimate of the result's range.
class SymbolicRangeInferrer
    : public SymExprVisitor<SymbolicRangeInferrer, RangeSet> {
public:
  static RangeSet inferRange(BasicValueFactory &BV, RangeSet::Factory &F,
                             ProgramStateRef State, SymbolRef Sym) {
    SymbolicRangeInferrer Inferrer(BV, F, State);
    return Inferrer.infer(Sym);
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
    const RangeSet *AssociatedRange = State->get<ConstraintRange>(Sym);

    // If Sym is a difference of symbols A - B, then maybe we have range set
    // stored for B - A.
    const RangeSet *RangeAssociatedWithNegatedSym =
        getRangeForMinusSymbol(State, Sym);

    // If we have range set stored for both A - B and B - A then calculate the
    // effective range set by intersecting the range set for A - B and the
    // negated range set of B - A.
    if (AssociatedRange && RangeAssociatedWithNegatedSym)
      return AssociatedRange->Intersect(
          ValueFactory, RangeFactory,
          RangeAssociatedWithNegatedSym->Negate(ValueFactory, RangeFactory));

    if (AssociatedRange)
      return *AssociatedRange;

    if (RangeAssociatedWithNegatedSym)
      return RangeAssociatedWithNegatedSym->Negate(ValueFactory, RangeFactory);

    return Visit(Sym);
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
      return VisitOrOperator(LHS, RHS, T);
    case BO_And:
      return VisitAndOperator(LHS, RHS, T);
    default:
      return infer(T);
    }
  }

  RangeSet VisitOrOperator(RangeSet LHS, RangeSet RHS, QualType T) {
    // TODO: generalize for the ranged RHS.
    if (const llvm::APSInt *RHSConstant = RHS.getConcreteValue()) {
      // For unsigned types, the output is greater-or-equal than RHS.
      if (T->isUnsignedIntegerType()) {
        return LHS.Intersect(ValueFactory, RangeFactory, *RHSConstant,
                             ValueFactory.getMaxValue(T));
      }

      // Bitwise-or with a non-zero constant is always non-zero.
      const llvm::APSInt &Zero = ValueFactory.getAPSIntType(T).getZeroValue();
      if (*RHSConstant != Zero) {
        return assumeNonZero(LHS, T);
      }
    }
    return infer(T);
  }

  RangeSet VisitAndOperator(RangeSet LHS, RangeSet RHS, QualType T) {
    // TODO: generalize for the ranged RHS.
    if (const llvm::APSInt *RHSConstant = RHS.getConcreteValue()) {
      const llvm::APSInt &Zero = ValueFactory.getAPSIntType(T).getZeroValue();

      // For unsigned types, or positive RHS,
      // bitwise-and output is always smaller-or-equal than RHS (assuming two's
      // complement representation of signed types).
      if (T->isUnsignedIntegerType() || *RHSConstant >= Zero) {
        return LHS.Intersect(ValueFactory, RangeFactory,
                             ValueFactory.getMinValue(T), *RHSConstant);
      }
    }
    return infer(T);
  }

  /// Return a range set subtracting zero from \p Domain.
  RangeSet assumeNonZero(RangeSet Domain, QualType T) {
    APSIntType IntType = ValueFactory.getAPSIntType(T);
    return Domain.Intersect(ValueFactory, RangeFactory,
                            ++IntType.getZeroValue(), --IntType.getZeroValue());
  }

  // FIXME: Once SValBuilder supports unary minus, we should use SValBuilder to
  //        obtain the negated symbolic expression instead of constructing the
  //        symbol manually. This will allow us to support finding ranges of not
  //        only negated SymSymExpr-type expressions, but also of other, simpler
  //        expressions which we currently do not know how to negate.
  const RangeSet *getRangeForMinusSymbol(ProgramStateRef State, SymbolRef Sym) {
    if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(Sym)) {
      if (SSE->getOpcode() == BO_Sub) {
        QualType T = Sym->getType();
        SymbolManager &SymMgr = State->getSymbolManager();
        SymbolRef negSym =
            SymMgr.getSymSymExpr(SSE->getRHS(), BO_Sub, SSE->getLHS(), T);

        if (const RangeSet *negV = State->get<ConstraintRange>(negSym)) {
          // Unsigned range set cannot be negated, unless it is [0, 0].
          if (T->isUnsignedIntegerOrEnumerationType() ||
              T->isSignedIntegerOrEnumerationType())
            return negV;
        }
      }
    }
    return nullptr;
  }

  BasicValueFactory &ValueFactory;
  RangeSet::Factory &RangeFactory;
  ProgramStateRef State;
};

class RangeConstraintManager : public RangedConstraintManager {
public:
  RangeConstraintManager(ExprEngine *EE, SValBuilder &SVB)
      : RangedConstraintManager(EE, SVB) {}

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  bool haveEqualConstraints(ProgramStateRef S1,
                            ProgramStateRef S2) const override {
    return S1->get<ConstraintRange>() == S2->get<ConstraintRange>();
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
  const RangeSet *getRangeForMinusSymbol(ProgramStateRef State, SymbolRef Sym);

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
};

} // end anonymous namespace

std::unique_ptr<ConstraintManager>
ento::CreateRangeConstraintManager(ProgramStateManager &StMgr,
                                   ExprEngine *Eng) {
  return std::make_unique<RangeConstraintManager>(Eng, StMgr.getSValBuilder());
}

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

const llvm::APSInt *RangeConstraintManager::getSymVal(ProgramStateRef St,
                                                      SymbolRef Sym) const {
  const ConstraintRangeTy::data_type *T = St->get<ConstraintRange>(Sym);
  return T ? T->getConcreteValue() : nullptr;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
ProgramStateRef
RangeConstraintManager::removeDeadBindings(ProgramStateRef State,
                                           SymbolReaper &SymReaper) {
  bool Changed = false;
  ConstraintRangeTy CR = State->get<ConstraintRange>();
  ConstraintRangeTy::Factory &CRFactory = State->get_context<ConstraintRange>();

  for (ConstraintRangeTy::iterator I = CR.begin(), E = CR.end(); I != E; ++I) {
    SymbolRef Sym = I.getKey();
    if (SymReaper.isDead(Sym)) {
      Changed = true;
      CR = CRFactory.remove(CR, Sym);
    }
  }

  return Changed ? State->set<ConstraintRange>(CR) : State;
}

RangeSet RangeConstraintManager::getRange(ProgramStateRef State,
                                          SymbolRef Sym) {
  return SymbolicRangeInferrer::inferRange(getBasicVals(), F, State, Sym);
}

// FIXME: Once SValBuilder supports unary minus, we should use SValBuilder to
//        obtain the negated symbolic expression instead of constructing the
//        symbol manually. This will allow us to support finding ranges of not
//        only negated SymSymExpr-type expressions, but also of other, simpler
//        expressions which we currently do not know how to negate.
const RangeSet*
RangeConstraintManager::getRangeForMinusSymbol(ProgramStateRef State,
                                               SymbolRef Sym) {
  if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(Sym)) {
    if (SSE->getOpcode() == BO_Sub) {
      QualType T = Sym->getType();
      SymbolManager &SymMgr = State->getSymbolManager();
      SymbolRef negSym = SymMgr.getSymSymExpr(SSE->getRHS(), BO_Sub,
                                              SSE->getLHS(), T);
      if (const RangeSet *negV = State->get<ConstraintRange>(negSym)) {
        if (T->isUnsignedIntegerOrEnumerationType() ||
            T->isSignedIntegerOrEnumerationType())
          return negV;
      }
    }
  }
  return nullptr;
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

  llvm::APSInt Lower = AdjustmentType.convert(Int) - Adjustment;
  llvm::APSInt Upper = Lower;
  --Lower;
  ++Upper;

  // [Int-Adjustment+1, Int-Adjustment-1]
  // Notice that the lower bound is greater than the upper bound.
  RangeSet New = getRange(St, Sym).Intersect(getBasicVals(), F, Upper, Lower);
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
  RangeSet New = getRange(St, Sym).Intersect(getBasicVals(), F, AdjInt, AdjInt);
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
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
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
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
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
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

RangeSet RangeConstraintManager::getSymLERange(
      llvm::function_ref<RangeSet()> RS,
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
  return New.isEmpty() ? nullptr : St->set<ConstraintRange>(Sym, New);
}

ProgramStateRef RangeConstraintManager::assumeSymWithinInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet New = getSymGERange(State, Sym, From, Adjustment);
  if (New.isEmpty())
    return nullptr;
  RangeSet Out = getSymLERange([&] { return New; }, To, Adjustment);
  return Out.isEmpty() ? nullptr : State->set<ConstraintRange>(Sym, Out);
}

ProgramStateRef RangeConstraintManager::assumeSymOutsideInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, const llvm::APSInt &Adjustment) {
  RangeSet RangeLT = getSymLTRange(State, Sym, From, Adjustment);
  RangeSet RangeGT = getSymGTRange(State, Sym, To, Adjustment);
  RangeSet New(RangeLT.addRange(F, RangeGT));
  return New.isEmpty() ? nullptr : State->set<ConstraintRange>(Sym, New);
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
  for (ConstraintRangeTy::iterator I = Constraints.begin();
       I != Constraints.end(); ++I) {
    Indent(Out, Space, IsDot)
        << "{ \"symbol\": \"" << I.getKey() << "\", \"range\": \"";
    I.getData().print(Out);
    Out << "\" }";

    if (std::next(I) != Constraints.end())
      Out << ',';
    Out << NL;
  }

  --Space;
  Indent(Out, Space, IsDot) << "]," << NL;
}
