//== SMTConstraintManager.cpp -----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConstraintManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

using namespace clang;
using namespace ento;

ProgramStateRef SMTConstraintManager::assumeSym(ProgramStateRef State,
                                                SymbolRef Sym,
                                                bool Assumption) {
  ASTContext &Ctx = getBasicVals().getContext();

  QualType RetTy;
  bool hasComparison;

  SMTExprRef Exp = Solver->getExpr(Ctx, Sym, &RetTy, &hasComparison);

  // Create zero comparison for implicit boolean cast, with reversed assumption
  if (!hasComparison && !RetTy->isBooleanType())
    return assumeExpr(State, Sym,
                      Solver->getZeroExpr(Ctx, Exp, RetTy, !Assumption));

  return assumeExpr(State, Sym, Assumption ? Exp : Solver->mkNot(Exp));
}

ProgramStateRef SMTConstraintManager::assumeSymInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, bool InRange) {
  ASTContext &Ctx = getBasicVals().getContext();
  return assumeExpr(State, Sym,
                    Solver->getRangeExpr(Ctx, Sym, From, To, InRange));
}

ProgramStateRef
SMTConstraintManager::assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                           bool Assumption) {
  // Skip anything that is unsupported
  return State;
}

ConditionTruthVal SMTConstraintManager::checkNull(ProgramStateRef State,
                                                  SymbolRef Sym) {
  ASTContext &Ctx = getBasicVals().getContext();

  QualType RetTy;
  // The expression may be casted, so we cannot call getZ3DataExpr() directly
  SMTExprRef VarExp = Solver->getExpr(Ctx, Sym, &RetTy);
  SMTExprRef Exp = Solver->getZeroExpr(Ctx, VarExp, RetTy, /*Assumption=*/true);

  // Negate the constraint
  SMTExprRef NotExp =
      Solver->getZeroExpr(Ctx, VarExp, RetTy, /*Assumption=*/false);

  Solver->reset();
  addStateConstraints(State);

  Solver->push();
  Solver->addConstraint(Exp);
  ConditionTruthVal isSat = Solver->check();

  Solver->pop();
  Solver->addConstraint(NotExp);
  ConditionTruthVal isNotSat = Solver->check();

  // Zero is the only possible solution
  if (isSat.isConstrainedTrue() && isNotSat.isConstrainedFalse())
    return true;

  // Zero is not a solution
  if (isSat.isConstrainedFalse() && isNotSat.isConstrainedTrue())
    return false;

  // Zero may be a solution
  return ConditionTruthVal();
}

const llvm::APSInt *SMTConstraintManager::getSymVal(ProgramStateRef State,
                                                    SymbolRef Sym) const {
  BasicValueFactory &BVF = getBasicVals();
  ASTContext &Ctx = BVF.getContext();

  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    QualType Ty = Sym->getType();
    assert(!Ty->isRealFloatingType());
    llvm::APSInt Value(Ctx.getTypeSize(Ty),
                       !Ty->isSignedIntegerOrEnumerationType());

    SMTExprRef Exp =
        Solver->fromData(SD->getSymbolID(), Ty, Ctx.getTypeSize(Ty));

    Solver->reset();
    addStateConstraints(State);

    // Constraints are unsatisfiable
    ConditionTruthVal isSat = Solver->check();
    if (!isSat.isConstrainedTrue())
      return nullptr;

    // Model does not assign interpretation
    if (!Solver->getInterpretation(Exp, Value))
      return nullptr;

    // A value has been obtained, check if it is the only value
    SMTExprRef NotExp = Solver->fromBinOp(
        Exp, BO_NE,
        Ty->isBooleanType() ? Solver->fromBoolean(Value.getBoolValue())
                            : Solver->fromAPSInt(Value),
        false);

    Solver->addConstraint(NotExp);

    ConditionTruthVal isNotSat = Solver->check();
    if (isNotSat.isConstrainedTrue())
      return nullptr;

    // This is the only solution, store it
    return &BVF.getValue(Value);
  }

  if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    SymbolRef CastSym = SC->getOperand();
    QualType CastTy = SC->getType();
    // Skip the void type
    if (CastTy->isVoidType())
      return nullptr;

    const llvm::APSInt *Value;
    if (!(Value = getSymVal(State, CastSym)))
      return nullptr;
    return &BVF.Convert(SC->getType(), *Value);
  }

  if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    const llvm::APSInt *LHS, *RHS;
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
      LHS = getSymVal(State, SIE->getLHS());
      RHS = &SIE->getRHS();
    } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
      LHS = &ISE->getLHS();
      RHS = getSymVal(State, ISE->getRHS());
    } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
      // Early termination to avoid expensive call
      LHS = getSymVal(State, SSM->getLHS());
      RHS = LHS ? getSymVal(State, SSM->getRHS()) : nullptr;
    } else {
      llvm_unreachable("Unsupported binary expression to get symbol value!");
    }

    if (!LHS || !RHS)
      return nullptr;

    llvm::APSInt ConvertedLHS, ConvertedRHS;
    QualType LTy, RTy;
    std::tie(ConvertedLHS, LTy) = Solver->fixAPSInt(Ctx, *LHS);
    std::tie(ConvertedRHS, RTy) = Solver->fixAPSInt(Ctx, *RHS);
    Solver->doIntTypeConversion<llvm::APSInt, &SMTSolver::castAPSInt>(
        Ctx, ConvertedLHS, LTy, ConvertedRHS, RTy);
    return BVF.evalAPSInt(BSE->getOpcode(), ConvertedLHS, ConvertedRHS);
  }

  llvm_unreachable("Unsupported expression to get symbol value!");
}

ConditionTruthVal
SMTConstraintManager::checkModel(ProgramStateRef State,
                                 const SMTExprRef &Exp) const {
  Solver->reset();
  Solver->addConstraint(Exp);
  addStateConstraints(State);
  return Solver->check();
}
