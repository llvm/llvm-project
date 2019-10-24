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
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/Template.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
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

template <typename AtomicEvaluator>
static bool
calculateConstraintSatisfaction(Sema &S, const Expr *ConstraintExpr,
                                ConstraintSatisfaction &Satisfaction,
                                AtomicEvaluator &&Evaluator) {
  ConstraintExpr = ConstraintExpr->IgnoreParenImpCasts();

  if (auto *BO = dyn_cast<BinaryOperator>(ConstraintExpr)) {
    if (BO->getOpcode() == BO_LAnd || BO->getOpcode() == BO_LOr) {
      if (calculateConstraintSatisfaction(S, BO->getLHS(), Satisfaction,
                                          Evaluator))
        return true;

      bool IsLHSSatisfied = Satisfaction.IsSatisfied;

      if (BO->getOpcode() == BO_LOr && IsLHSSatisfied)
        // [temp.constr.op] p3
        //    A disjunction is a constraint taking two operands. To determine if
        //    a disjunction is satisfied, the satisfaction of the first operand
        //    is checked. If that is satisfied, the disjunction is satisfied.
        //    Otherwise, the disjunction is satisfied if and only if the second
        //    operand is satisfied.
        return false;

      if (BO->getOpcode() == BO_LAnd && !IsLHSSatisfied)
        // [temp.constr.op] p2
        //    A conjunction is a constraint taking two operands. To determine if
        //    a conjunction is satisfied, the satisfaction of the first operand
        //    is checked. If that is not satisfied, the conjunction is not
        //    satisfied. Otherwise, the conjunction is satisfied if and only if
        //    the second operand is satisfied.
        return false;

      return calculateConstraintSatisfaction(S, BO->getRHS(), Satisfaction,
          std::forward<AtomicEvaluator>(Evaluator));
    }
  }
  else if (auto *C = dyn_cast<ExprWithCleanups>(ConstraintExpr))
    return calculateConstraintSatisfaction(S, C->getSubExpr(), Satisfaction,
        std::forward<AtomicEvaluator>(Evaluator));

  // An atomic constraint expression
  ExprResult SubstitutedAtomicExpr = Evaluator(ConstraintExpr);

  if (SubstitutedAtomicExpr.isInvalid())
    return true;

  if (!SubstitutedAtomicExpr.isUsable())
    // Evaluator has decided satisfaction without yielding an expression.
    return false;

  EnterExpressionEvaluationContext ConstantEvaluated(
      S, Sema::ExpressionEvaluationContext::ConstantEvaluated);
  SmallVector<PartialDiagnosticAt, 2> EvaluationDiags;
  Expr::EvalResult EvalResult;
  EvalResult.Diag = &EvaluationDiags;
  if (!SubstitutedAtomicExpr.get()->EvaluateAsRValue(EvalResult, S.Context)) {
      // C++2a [temp.constr.atomic]p1
      //   ...E shall be a constant expression of type bool.
    S.Diag(SubstitutedAtomicExpr.get()->getBeginLoc(),
           diag::err_non_constant_constraint_expression)
        << SubstitutedAtomicExpr.get()->getSourceRange();
    for (const PartialDiagnosticAt &PDiag : EvaluationDiags)
      S.Diag(PDiag.first, PDiag.second);
    return true;
  }

  Satisfaction.IsSatisfied = EvalResult.Val.getInt().getBoolValue();
  if (!Satisfaction.IsSatisfied)
    Satisfaction.Details.emplace_back(ConstraintExpr,
                                      SubstitutedAtomicExpr.get());

  return false;
}

template <typename TemplateDeclT>
static bool calculateConstraintSatisfaction(
    Sema &S, TemplateDeclT *Template, ArrayRef<TemplateArgument> TemplateArgs,
    SourceLocation TemplateNameLoc, MultiLevelTemplateArgumentList &MLTAL,
    const Expr *ConstraintExpr, ConstraintSatisfaction &Satisfaction) {
  return calculateConstraintSatisfaction(
      S, ConstraintExpr, Satisfaction, [&](const Expr *AtomicExpr) {
        EnterExpressionEvaluationContext ConstantEvaluated(
            S, Sema::ExpressionEvaluationContext::ConstantEvaluated);

        // Atomic constraint - substitute arguments and check satisfaction.
        ExprResult SubstitutedExpression;
        {
          TemplateDeductionInfo Info(TemplateNameLoc);
          Sema::InstantiatingTemplate Inst(S, AtomicExpr->getBeginLoc(),
              Sema::InstantiatingTemplate::ConstraintSubstitution{}, Template,
              Info, AtomicExpr->getSourceRange());
          if (Inst.isInvalid())
            return ExprError();
          // We do not want error diagnostics escaping here.
          Sema::SFINAETrap Trap(S);
          SubstitutedExpression = S.SubstExpr(const_cast<Expr *>(AtomicExpr),
                                              MLTAL);
          if (SubstitutedExpression.isInvalid() || Trap.hasErrorOccurred()) {
            // C++2a [temp.constr.atomic]p1
            //   ...If substitution results in an invalid type or expression, the
            //   constraint is not satisfied.
            if (!Trap.hasErrorOccurred())
              // A non-SFINAE error has occured as a result of this
              // substitution.
              return ExprError();

            PartialDiagnosticAt SubstDiag{SourceLocation(),
                                          PartialDiagnostic::NullDiagnostic()};
            Info.takeSFINAEDiagnostic(SubstDiag);
            // FIXME: Concepts: This is an unfortunate consequence of there
            //  being no serialization code for PartialDiagnostics and the fact
            //  that serializing them would likely take a lot more storage than
            //  just storing them as strings. We would still like, in the
            //  future, to serialize the proper PartialDiagnostic as serializing
            //  it as a string defeats the purpose of the diagnostic mechanism.
            SmallString<128> DiagString;
            DiagString = ": ";
            SubstDiag.second.EmitToString(S.getDiagnostics(), DiagString);
            Satisfaction.Details.emplace_back(
                AtomicExpr,
                new (S.Context) ConstraintSatisfaction::SubstitutionDiagnostic{
                        SubstDiag.first,
                        std::string(DiagString.begin(), DiagString.end())});
            Satisfaction.IsSatisfied = false;
            return ExprEmpty();
          }
        }

        if (!S.CheckConstraintExpression(SubstitutedExpression.get()))
          return ExprError();

        return SubstitutedExpression;
      });
}

template<typename TemplateDeclT>
static bool CheckConstraintSatisfaction(Sema &S, TemplateDeclT *Template,
                                        ArrayRef<const Expr *> ConstraintExprs,
                                        ArrayRef<TemplateArgument> TemplateArgs,
                                        SourceRange TemplateIDRange,
                                        ConstraintSatisfaction &Satisfaction) {
  if (ConstraintExprs.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  for (auto& Arg : TemplateArgs)
    if (Arg.isInstantiationDependent()) {
      // No need to check satisfaction for dependent constraint expressions.
      Satisfaction.IsSatisfied = true;
      return false;
    }

  Sema::InstantiatingTemplate Inst(S, TemplateIDRange.getBegin(),
      Sema::InstantiatingTemplate::ConstraintsCheck{}, Template, TemplateArgs,
      TemplateIDRange);
  if (Inst.isInvalid())
    return true;

  MultiLevelTemplateArgumentList MLTAL;
  MLTAL.addOuterTemplateArguments(TemplateArgs);

  for (const Expr *ConstraintExpr : ConstraintExprs) {
    if (calculateConstraintSatisfaction(S, Template, TemplateArgs,
                                        TemplateIDRange.getBegin(), MLTAL,
                                        ConstraintExpr, Satisfaction))
      return true;
    if (!Satisfaction.IsSatisfied)
      // [temp.constr.op] p2
      //   [...] To determine if a conjunction is satisfied, the satisfaction
      //   of the first operand is checked. If that is not satisfied, the
      //   conjunction is not satisfied. [...]
      return false;
  }
  return false;
}

bool Sema::CheckConstraintSatisfaction(TemplateDecl *Template,
                                       ArrayRef<const Expr *> ConstraintExprs,
                                       ArrayRef<TemplateArgument> TemplateArgs,
                                       SourceRange TemplateIDRange,
                                       ConstraintSatisfaction &Satisfaction) {
  return ::CheckConstraintSatisfaction(*this, Template, ConstraintExprs,
                                       TemplateArgs, TemplateIDRange,
                                       Satisfaction);
}

bool
Sema::CheckConstraintSatisfaction(ClassTemplatePartialSpecializationDecl* Part,
                                  ArrayRef<const Expr *> ConstraintExprs,
                                  ArrayRef<TemplateArgument> TemplateArgs,
                                  SourceRange TemplateIDRange,
                                  ConstraintSatisfaction &Satisfaction) {
  return ::CheckConstraintSatisfaction(*this, Part, ConstraintExprs,
                                       TemplateArgs, TemplateIDRange,
                                       Satisfaction);
}

bool
Sema::CheckConstraintSatisfaction(VarTemplatePartialSpecializationDecl* Partial,
                                  ArrayRef<const Expr *> ConstraintExprs,
                                  ArrayRef<TemplateArgument> TemplateArgs,
                                  SourceRange TemplateIDRange,
                                  ConstraintSatisfaction &Satisfaction) {
  return ::CheckConstraintSatisfaction(*this, Partial, ConstraintExprs,
                                       TemplateArgs, TemplateIDRange,
                                       Satisfaction);
}

bool Sema::CheckConstraintSatisfaction(const Expr *ConstraintExpr,
                                       ConstraintSatisfaction &Satisfaction) {
  return calculateConstraintSatisfaction(
      *this, ConstraintExpr, Satisfaction,
      [](const Expr *AtomicExpr) -> ExprResult {
        return ExprResult(const_cast<Expr *>(AtomicExpr));
      });
}

bool Sema::EnsureTemplateArgumentListConstraints(
    TemplateDecl *TD, ArrayRef<TemplateArgument> TemplateArgs,
    SourceRange TemplateIDRange) {
  ConstraintSatisfaction Satisfaction;
  llvm::SmallVector<const Expr *, 3> AssociatedConstraints;
  TD->getAssociatedConstraints(AssociatedConstraints);
  if (CheckConstraintSatisfaction(TD, AssociatedConstraints, TemplateArgs,
                                  TemplateIDRange, Satisfaction))
    return true;

  if (!Satisfaction.IsSatisfied) {
    SmallString<128> TemplateArgString;
    TemplateArgString = " ";
    TemplateArgString += getTemplateArgumentBindingsText(
        TD->getTemplateParameters(), TemplateArgs.data(), TemplateArgs.size());

    Diag(TemplateIDRange.getBegin(),
         diag::err_template_arg_list_constraints_not_satisfied)
        << (int)getTemplateNameKindForDiagnostics(TemplateName(TD)) << TD
        << TemplateArgString << TemplateIDRange;
    DiagnoseUnsatisfiedConstraint(Satisfaction);
    return true;
  }
  return false;
}

static void diagnoseWellFormedUnsatisfiedConstraintExpr(Sema &S,
                                                        Expr *SubstExpr,
                                                        bool First = true) {
  SubstExpr = SubstExpr->IgnoreParenImpCasts();
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(SubstExpr)) {
    switch (BO->getOpcode()) {
    // These two cases will in practice only be reached when using fold
    // expressions with || and &&, since otherwise the || and && will have been
    // broken down into atomic constraints during satisfaction checking.
    case BO_LOr:
      // Or evaluated to false - meaning both RHS and LHS evaluated to false.
      diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getLHS(), First);
      diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getRHS(),
                                                  /*First=*/false);
      return;
    case BO_LAnd:
      bool LHSSatisfied;
      BO->getLHS()->EvaluateAsBooleanCondition(LHSSatisfied, S.Context);
      if (LHSSatisfied) {
        // LHS is true, so RHS must be false.
        diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getRHS(), First);
        return;
      }
      // LHS is false
      diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getLHS(), First);

      // RHS might also be false
      bool RHSSatisfied;
      BO->getRHS()->EvaluateAsBooleanCondition(RHSSatisfied, S.Context);
      if (!RHSSatisfied)
        diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getRHS(),
                                                    /*First=*/false);
      return;
    case BO_GE:
    case BO_LE:
    case BO_GT:
    case BO_LT:
    case BO_EQ:
    case BO_NE:
      if (BO->getLHS()->getType()->isIntegerType() &&
          BO->getRHS()->getType()->isIntegerType()) {
        Expr::EvalResult SimplifiedLHS;
        Expr::EvalResult SimplifiedRHS;
        BO->getLHS()->EvaluateAsInt(SimplifiedLHS, S.Context);
        BO->getRHS()->EvaluateAsInt(SimplifiedRHS, S.Context);
        if (!SimplifiedLHS.Diag && ! SimplifiedRHS.Diag) {
          S.Diag(SubstExpr->getBeginLoc(),
                 diag::note_atomic_constraint_evaluated_to_false_elaborated)
              << (int)First << SubstExpr
              << SimplifiedLHS.Val.getInt().toString(10)
              << BinaryOperator::getOpcodeStr(BO->getOpcode())
              << SimplifiedRHS.Val.getInt().toString(10);
          return;
        }
      }
      break;

    default:
      break;
    }
  } else if (auto *CSE = dyn_cast<ConceptSpecializationExpr>(SubstExpr)) {
    if (CSE->getTemplateArgsAsWritten()->NumTemplateArgs == 1) {
      S.Diag(
          CSE->getSourceRange().getBegin(),
          diag::
          note_single_arg_concept_specialization_constraint_evaluated_to_false)
          << (int)First
          << CSE->getTemplateArgsAsWritten()->arguments()[0].getArgument()
          << CSE->getNamedConcept();
    } else {
      S.Diag(SubstExpr->getSourceRange().getBegin(),
             diag::note_concept_specialization_constraint_evaluated_to_false)
          << (int)First << CSE;
    }
    S.DiagnoseUnsatisfiedConstraint(CSE->getSatisfaction());
    return;
  }

  S.Diag(SubstExpr->getSourceRange().getBegin(),
         diag::note_atomic_constraint_evaluated_to_false)
      << (int)First << SubstExpr;
}

template<typename SubstitutionDiagnostic>
static void diagnoseUnsatisfiedConstraintExpr(
    Sema &S, const Expr *E,
    const llvm::PointerUnion<Expr *, SubstitutionDiagnostic *> &Record,
    bool First = true) {
  if (auto *Diag = Record.template dyn_cast<SubstitutionDiagnostic *>()){
    S.Diag(Diag->first, diag::note_substituted_constraint_expr_is_ill_formed)
        << Diag->second;
    return;
  }

  diagnoseWellFormedUnsatisfiedConstraintExpr(S,
      Record.template get<Expr *>(), First);
}

void Sema::DiagnoseUnsatisfiedConstraint(
    const ConstraintSatisfaction& Satisfaction) {
  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");
  bool First = true;
  for (auto &Pair : Satisfaction.Details) {
    diagnoseUnsatisfiedConstraintExpr(*this, Pair.first, Pair.second, First);
    First = false;
  }
}

void Sema::DiagnoseUnsatisfiedConstraint(
    const ASTConstraintSatisfaction &Satisfaction) {
  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");
  bool First = true;
  for (auto &Pair : Satisfaction) {
    diagnoseUnsatisfiedConstraintExpr(*this, Pair.first, Pair.second, First);
    First = false;
  }
}