//===-- SemaConcept.cpp - Semantic Analysis for Constraints and Concepts --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ constraints and concepts.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/Template.h"
#include "clang/AST/ExprCXX.h"
using namespace clang;
using namespace sema;

bool Sema::CheckConstraintExpression(Expr *ConstraintExpression) {
  // C++2a [temp.constr.atomic]p1
  // ..E shall be a constant expression of type bool.

  ConstraintExpression = ConstraintExpression->IgnoreParenImpCasts();

  if (auto *BinOp = dyn_cast<BinaryOperator>(ConstraintExpression)) {
    if (BinOp->getOpcode() == BO_LAnd || BinOp->getOpcode() == BO_LOr)
      return CheckConstraintExpression(BinOp->getLHS()) &&
             CheckConstraintExpression(BinOp->getRHS());
  } else if (auto *C = dyn_cast<ExprWithCleanups>(ConstraintExpression))
    return CheckConstraintExpression(C->getSubExpr());

  // An atomic constraint!
  if (ConstraintExpression->isTypeDependent())
    return true;

  QualType Type = ConstraintExpression->getType();
  if (!Context.hasSameUnqualifiedType(Type, Context.BoolTy)) {
    Diag(ConstraintExpression->getExprLoc(),
         diag::err_non_bool_atomic_constraint) << Type
        << ConstraintExpression->getSourceRange();
    return false;
  }
  return true;
}

bool
Sema::CalculateConstraintSatisfaction(ConceptDecl *NamedConcept,
                                      MultiLevelTemplateArgumentList &MLTAL,
                                      Expr *ConstraintExpr,
                                      bool &IsSatisfied) {
  ConstraintExpr = ConstraintExpr->IgnoreParenImpCasts();

  if (auto *BO = dyn_cast<BinaryOperator>(ConstraintExpr)) {
    if (BO->getOpcode() == BO_LAnd) {
      if (CalculateConstraintSatisfaction(NamedConcept, MLTAL, BO->getLHS(),
                                          IsSatisfied))
        return true;
      if (!IsSatisfied)
        return false;
      return CalculateConstraintSatisfaction(NamedConcept, MLTAL, BO->getRHS(),
                                             IsSatisfied);
    } else if (BO->getOpcode() == BO_LOr) {
      if (CalculateConstraintSatisfaction(NamedConcept, MLTAL, BO->getLHS(),
                                          IsSatisfied))
        return true;
      if (IsSatisfied)
        return false;
      return CalculateConstraintSatisfaction(NamedConcept, MLTAL, BO->getRHS(),
                                             IsSatisfied);
    }
  }
  else if (auto *C = dyn_cast<ExprWithCleanups>(ConstraintExpr))
    return CalculateConstraintSatisfaction(NamedConcept, MLTAL, C->getSubExpr(),
                                           IsSatisfied);

  EnterExpressionEvaluationContext ConstantEvaluated(
      *this, Sema::ExpressionEvaluationContext::ConstantEvaluated);

  // Atomic constraint - substitute arguments and check satisfaction.
  ExprResult E;
  {
    TemplateDeductionInfo Info(ConstraintExpr->getBeginLoc());
    InstantiatingTemplate Inst(*this, ConstraintExpr->getBeginLoc(),
                               InstantiatingTemplate::ConstraintSubstitution{},
                               NamedConcept, Info,
                               ConstraintExpr->getSourceRange());
    if (Inst.isInvalid())
      return true;
    // We do not want error diagnostics escaping here.
    Sema::SFINAETrap Trap(*this);

    E = SubstExpr(ConstraintExpr, MLTAL);
    if (E.isInvalid() || Trap.hasErrorOccurred()) {
      // C++2a [temp.constr.atomic]p1
      //   ...If substitution results in an invalid type or expression, the
      //   constraint is not satisfied.
      IsSatisfied = false;
      return false;
    }
  }

  if (!CheckConstraintExpression(E.get()))
    return true;

  SmallVector<PartialDiagnosticAt, 2> EvaluationDiags;
  Expr::EvalResult EvalResult;
  EvalResult.Diag = &EvaluationDiags;
  if (!E.get()->EvaluateAsRValue(EvalResult, Context)) {
    // C++2a [temp.constr.atomic]p1
    //   ...E shall be a constant expression of type bool.
    Diag(E.get()->getBeginLoc(),
         diag::err_non_constant_constraint_expression)
        << E.get()->getSourceRange();
    for (const PartialDiagnosticAt &PDiag : EvaluationDiags)
      Diag(PDiag.first, PDiag.second);
    return true;
  }

  IsSatisfied = EvalResult.Val.getInt().getBoolValue();

  return false;
}