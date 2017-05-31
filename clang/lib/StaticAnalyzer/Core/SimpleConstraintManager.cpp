//== SimpleConstraintManager.cpp --------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SimpleConstraintManager, a class that holds code shared
//  between BasicConstraintManager and RangeConstraintManager.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

namespace clang {

namespace ento {

SimpleConstraintManager::~SimpleConstraintManager() {}

bool SimpleConstraintManager::canReasonAbout(SVal X) const {
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
      if (BinaryOperator::isComparisonOp(SSE->getOpcode())) {
        // We handle Loc <> Loc comparisons, but not (yet) NonLoc <> NonLoc.
        if (Loc::isLocType(SSE->getLHS()->getType())) {
          assert(Loc::isLocType(SSE->getRHS()->getType()));
          return true;
        }
      }
    }

    return false;
  }

  return true;
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               DefinedSVal Cond,
                                               bool Assumption) {
  // If we have a Loc value, cast it to a bool NonLoc first.
  if (Optional<Loc> LV = Cond.getAs<Loc>()) {
    SValBuilder &SVB = state->getStateManager().getSValBuilder();
    QualType T;
    const MemRegion *MR = LV->getAsRegion();
    if (const TypedRegion *TR = dyn_cast_or_null<TypedRegion>(MR))
      T = TR->getLocationType();
    else
      T = SVB.getContext().VoidPtrTy;

    Cond = SVB.evalCast(*LV, SVB.getContext().BoolTy, T).castAs<DefinedSVal>();
  }

  return assume(state, Cond.castAs<NonLoc>(), Assumption);
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               NonLoc cond,
                                               bool assumption) {
  state = assumeAux(state, cond, assumption);
  if (NotifyAssumeClients && SU)
    return SU->processAssume(state, cond, assumption);
  return state;
}


ProgramStateRef
SimpleConstraintManager::assumeAuxForSymbol(ProgramStateRef State,
                                            SymbolRef Sym, bool Assumption) {
  BasicValueFactory &BVF = getBasicVals();
  QualType T = Sym->getType();

  // None of the constraint solvers currently support non-integer types.
  if (!T->isIntegralOrEnumerationType())
    return State;

  const llvm::APSInt &zero = BVF.getValue(0, T);
  if (Assumption)
    return assumeSymNE(State, Sym, zero, zero);
  else
    return assumeSymEQ(State, Sym, zero, zero);
}

ProgramStateRef SimpleConstraintManager::assumeAux(ProgramStateRef state,
                                                  NonLoc Cond,
                                                  bool Assumption) {

  // We cannot reason about SymSymExprs, and can only reason about some
  // SymIntExprs.
  if (!canReasonAbout(Cond)) {
    // Just add the constraint to the expression without trying to simplify.
    SymbolRef sym = Cond.getAsSymExpr();
    return assumeAuxForSymbol(state, sym, Assumption);
  }

  switch (Cond.getSubKind()) {
  default:
    llvm_unreachable("'Assume' not implemented for this NonLoc");

  case nonloc::SymbolValKind: {
    nonloc::SymbolVal SV = Cond.castAs<nonloc::SymbolVal>();
    SymbolRef sym = SV.getSymbol();
    assert(sym);

    // Handle SymbolData.
    if (!SV.isExpression()) {
      return assumeAuxForSymbol(state, sym, Assumption);

    // Handle symbolic expression.
    } else if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(sym)) {
      // We can only simplify expressions whose RHS is an integer.

      BinaryOperator::Opcode op = SE->getOpcode();
      if (BinaryOperator::isComparisonOp(op)) {
        if (!Assumption)
          op = BinaryOperator::negateComparisonOp(op);

        return assumeSymRel(state, SE->getLHS(), op, SE->getRHS());
      }

    } else if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(sym)) {
      // Translate "a != b" to "(b - a) != 0".
      // We invert the order of the operands as a heuristic for how loop
      // conditions are usually written ("begin != end") as compared to length
      // calculations ("end - begin"). The more correct thing to do would be to
      // canonicalize "a - b" and "b - a", which would allow us to treat
      // "a != b" and "b != a" the same.
      SymbolManager &SymMgr = getSymbolManager();
      BinaryOperator::Opcode Op = SSE->getOpcode();
      assert(BinaryOperator::isComparisonOp(Op));

      // For now, we only support comparing pointers.
      assert(Loc::isLocType(SSE->getLHS()->getType()));
      assert(Loc::isLocType(SSE->getRHS()->getType()));
      QualType DiffTy = SymMgr.getContext().getPointerDiffType();
      SymbolRef Subtraction = SymMgr.getSymSymExpr(SSE->getRHS(), BO_Sub,
                                                   SSE->getLHS(), DiffTy);

      const llvm::APSInt &Zero = getBasicVals().getValue(0, DiffTy);
      Op = BinaryOperator::reverseComparisonOp(Op);
      if (!Assumption)
        Op = BinaryOperator::negateComparisonOp(Op);
      return assumeSymRel(state, Subtraction, Op, Zero);
    }

    // If we get here, there's nothing else we can do but treat the symbol as
    // opaque.
    return assumeAuxForSymbol(state, sym, Assumption);
  }

  case nonloc::ConcreteIntKind: {
    bool b = Cond.castAs<nonloc::ConcreteInt>().getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : nullptr;
  }

  case nonloc::LocAsIntegerKind:
    return assume(state, Cond.castAs<nonloc::LocAsInteger>().getLoc(),
                  Assumption);
  } // end switch
}

ProgramStateRef SimpleConstraintManager::assumeWithinInclusiveRange(
    ProgramStateRef State, NonLoc Value, const llvm::APSInt &From,
    const llvm::APSInt &To, bool InRange) {

  assert(From.isUnsigned() == To.isUnsigned() &&
         From.getBitWidth() == To.getBitWidth() &&
         "Values should have same types!");

  if (!canReasonAbout(Value)) {
    // Just add the constraint to the expression without trying to simplify.
    SymbolRef Sym = Value.getAsSymExpr();
    assert(Sym);
    return assumeSymWithinInclusiveRange(State, Sym, From, To, InRange);
  }

  switch (Value.getSubKind()) {
  default:
    llvm_unreachable("'assumeWithinInclusiveRange' is not implemented"
                     "for this NonLoc");

  case nonloc::LocAsIntegerKind:
  case nonloc::SymbolValKind: {
    if (SymbolRef Sym = Value.getAsSymbol())
      return assumeSymWithinInclusiveRange(State, Sym, From, To, InRange);
    return State;
  } // end switch

  case nonloc::ConcreteIntKind: {
    const llvm::APSInt &IntVal = Value.castAs<nonloc::ConcreteInt>().getValue();
    bool IsInRange = IntVal >= From && IntVal <= To;
    bool isFeasible = (IsInRange == InRange);
    return isFeasible ? State : nullptr;
  }
  } // end switch
}

static void computeAdjustment(SymbolRef &Sym, llvm::APSInt &Adjustment) {
  // Is it a "($sym+constant1)" expression?
  if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(Sym)) {
    BinaryOperator::Opcode Op = SE->getOpcode();
    if (Op == BO_Add || Op == BO_Sub) {
      Sym = SE->getLHS();
      Adjustment = APSIntType(Adjustment).convert(SE->getRHS());

      // Don't forget to negate the adjustment if it's being subtracted.
      // This should happen /after/ promotion, in case the value being
      // subtracted is, say, CHAR_MIN, and the promoted type is 'int'.
      if (Op == BO_Sub)
        Adjustment = -Adjustment;
    }
  }
}

ProgramStateRef SimpleConstraintManager::assumeSymRel(ProgramStateRef state,
                                                     const SymExpr *LHS,
                                                     BinaryOperator::Opcode op,
                                                     const llvm::APSInt& Int) {
  assert(BinaryOperator::isComparisonOp(op) &&
         "Non-comparison ops should be rewritten as comparisons to zero.");

  // Get the type used for calculating wraparound.
  BasicValueFactory &BVF = getBasicVals();
  APSIntType WraparoundType = BVF.getAPSIntType(LHS->getType());

  // We only handle simple comparisons of the form "$sym == constant"
  // or "($sym+constant1) == constant2".
  // The adjustment is "constant1" in the above expression. It's used to
  // "slide" the solution range around for modular arithmetic. For example,
  // x < 4 has the solution [0, 3]. x+2 < 4 has the solution [0-2, 3-2], which
  // in modular arithmetic is [0, 1] U [UINT_MAX-1, UINT_MAX]. It's up to
  // the subclasses of SimpleConstraintManager to handle the adjustment.
  SymbolRef Sym = LHS;
  llvm::APSInt Adjustment = WraparoundType.getZeroValue();
  computeAdjustment(Sym, Adjustment);

  // Convert the right-hand side integer as necessary.
  APSIntType ComparisonType = std::max(WraparoundType, APSIntType(Int));
  llvm::APSInt ConvertedInt = ComparisonType.convert(Int);

  // Prefer unsigned comparisons.
  if (ComparisonType.getBitWidth() == WraparoundType.getBitWidth() &&
      ComparisonType.isUnsigned() && !WraparoundType.isUnsigned())
    Adjustment.setIsSigned(false);

  switch (op) {
  default:
    llvm_unreachable("invalid operation not caught by assertion above");

  case BO_EQ:
    return assumeSymEQ(state, Sym, ConvertedInt, Adjustment);

  case BO_NE:
    return assumeSymNE(state, Sym, ConvertedInt, Adjustment);

  case BO_GT:
    return assumeSymGT(state, Sym, ConvertedInt, Adjustment);

  case BO_GE:
    return assumeSymGE(state, Sym, ConvertedInt, Adjustment);

  case BO_LT:
    return assumeSymLT(state, Sym, ConvertedInt, Adjustment);

  case BO_LE:
    return assumeSymLE(state, Sym, ConvertedInt, Adjustment);
  } // end switch
}

ProgramStateRef
SimpleConstraintManager::assumeSymWithinInclusiveRange(ProgramStateRef State,
                                                       SymbolRef Sym,
                                                       const llvm::APSInt &From,
                                                       const llvm::APSInt &To,
                                                       bool InRange) {
  // Get the type used for calculating wraparound.
  BasicValueFactory &BVF = getBasicVals();
  APSIntType WraparoundType = BVF.getAPSIntType(Sym->getType());

  llvm::APSInt Adjustment = WraparoundType.getZeroValue();
  SymbolRef AdjustedSym = Sym;
  computeAdjustment(AdjustedSym, Adjustment);

  // Convert the right-hand side integer as necessary.
  APSIntType ComparisonType = std::max(WraparoundType, APSIntType(From));
  llvm::APSInt ConvertedFrom = ComparisonType.convert(From);
  llvm::APSInt ConvertedTo = ComparisonType.convert(To);

  // Prefer unsigned comparisons.
  if (ComparisonType.getBitWidth() == WraparoundType.getBitWidth() &&
      ComparisonType.isUnsigned() && !WraparoundType.isUnsigned())
    Adjustment.setIsSigned(false);

  if (InRange)
    return assumeSymbolWithinInclusiveRange(State, AdjustedSym, ConvertedFrom,
                                            ConvertedTo, Adjustment);
  return assumeSymbolOutOfInclusiveRange(State, AdjustedSym, ConvertedFrom,
                                         ConvertedTo, Adjustment);
}

} // end of namespace ento

} // end of namespace clang
