//===-- SemaConcept.cpp - Semantic Analysis for Constraints and Concepts --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ constraints and concepts.
//
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/Sema/SemaConcept.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

namespace {
class LogicalBinOp {
  SourceLocation Loc;
  OverloadedOperatorKind Op = OO_None;
  const Expr *LHS = nullptr;
  const Expr *RHS = nullptr;

public:
  LogicalBinOp(const Expr *E) {
    if (auto *BO = dyn_cast<BinaryOperator>(E)) {
      Op = BinaryOperator::getOverloadedOperator(BO->getOpcode());
      LHS = BO->getLHS();
      RHS = BO->getRHS();
      Loc = BO->getExprLoc();
    } else if (auto *OO = dyn_cast<CXXOperatorCallExpr>(E)) {
      // If OO is not || or && it might not have exactly 2 arguments.
      if (OO->getNumArgs() == 2) {
        Op = OO->getOperator();
        LHS = OO->getArg(0);
        RHS = OO->getArg(1);
        Loc = OO->getOperatorLoc();
      }
    }
  }

  bool isAnd() const { return Op == OO_AmpAmp; }
  bool isOr() const { return Op == OO_PipePipe; }
  explicit operator bool() const { return isAnd() || isOr(); }

  const Expr *getLHS() const { return LHS; }
  const Expr *getRHS() const { return RHS; }

  ExprResult recreateBinOp(Sema &SemaRef, ExprResult LHS) const {
    return recreateBinOp(SemaRef, LHS, const_cast<Expr *>(getRHS()));
  }

  ExprResult recreateBinOp(Sema &SemaRef, ExprResult LHS,
                           ExprResult RHS) const {
    assert((isAnd() || isOr()) && "Not the right kind of op?");
    assert((!LHS.isInvalid() && !RHS.isInvalid()) && "not good expressions?");

    if (!LHS.isUsable() || !RHS.isUsable())
      return ExprEmpty();

    // We should just be able to 'normalize' these to the builtin Binary
    // Operator, since that is how they are evaluated in constriant checks.
    return BinaryOperator::Create(SemaRef.Context, LHS.get(), RHS.get(),
                                  BinaryOperator::getOverloadedOpcode(Op),
                                  SemaRef.Context.BoolTy, VK_PRValue,
                                  OK_Ordinary, Loc, FPOptionsOverride{});
  }
};
}

bool Sema::CheckConstraintExpression(const Expr *ConstraintExpression,
                                     Token NextToken, bool *PossibleNonPrimary,
                                     bool IsTrailingRequiresClause) {
  // C++2a [temp.constr.atomic]p1
  // ..E shall be a constant expression of type bool.

  ConstraintExpression = ConstraintExpression->IgnoreParenImpCasts();

  if (LogicalBinOp BO = ConstraintExpression) {
    return CheckConstraintExpression(BO.getLHS(), NextToken,
                                     PossibleNonPrimary) &&
           CheckConstraintExpression(BO.getRHS(), NextToken,
                                     PossibleNonPrimary);
  } else if (auto *C = dyn_cast<ExprWithCleanups>(ConstraintExpression))
    return CheckConstraintExpression(C->getSubExpr(), NextToken,
                                     PossibleNonPrimary);

  QualType Type = ConstraintExpression->getType();

  auto CheckForNonPrimary = [&] {
    if (PossibleNonPrimary)
      *PossibleNonPrimary =
          // We have the following case:
          // template<typename> requires func(0) struct S { };
          // The user probably isn't aware of the parentheses required around
          // the function call, and we're only going to parse 'func' as the
          // primary-expression, and complain that it is of non-bool type.
          (NextToken.is(tok::l_paren) &&
           (IsTrailingRequiresClause ||
            (Type->isDependentType() &&
             isa<UnresolvedLookupExpr>(ConstraintExpression)) ||
            Type->isFunctionType() ||
            Type->isSpecificBuiltinType(BuiltinType::Overload))) ||
          // We have the following case:
          // template<typename T> requires size_<T> == 0 struct S { };
          // The user probably isn't aware of the parentheses required around
          // the binary operator, and we're only going to parse 'func' as the
          // first operand, and complain that it is of non-bool type.
          getBinOpPrecedence(NextToken.getKind(),
                             /*GreaterThanIsOperator=*/true,
                             getLangOpts().CPlusPlus11) > prec::LogicalAnd;
  };

  // An atomic constraint!
  if (ConstraintExpression->isTypeDependent()) {
    CheckForNonPrimary();
    return true;
  }

  if (!Context.hasSameUnqualifiedType(Type, Context.BoolTy)) {
    Diag(ConstraintExpression->getExprLoc(),
         diag::err_non_bool_atomic_constraint) << Type
        << ConstraintExpression->getSourceRange();
    CheckForNonPrimary();
    return false;
  }

  if (PossibleNonPrimary)
      *PossibleNonPrimary = false;
  return true;
}

namespace {
struct SatisfactionStackRAII {
  Sema &SemaRef;
  SatisfactionStackRAII(Sema &SemaRef, llvm::FoldingSetNodeID FSNID)
      : SemaRef(SemaRef) {
      SemaRef.PushSatisfactionStackEntry(FSNID);
  }
  ~SatisfactionStackRAII() { SemaRef.PopSatisfactionStackEntry(); }
};
} // namespace

template <typename AtomicEvaluator>
static ExprResult
calculateConstraintSatisfaction(Sema &S, const Expr *ConstraintExpr,
                                ConstraintSatisfaction &Satisfaction,
                                AtomicEvaluator &&Evaluator) {
  ConstraintExpr = ConstraintExpr->IgnoreParenImpCasts();

  if (LogicalBinOp BO = ConstraintExpr) {
    ExprResult LHSRes = calculateConstraintSatisfaction(
        S, BO.getLHS(), Satisfaction, Evaluator);

    if (LHSRes.isInvalid())
      return ExprError();

    bool IsLHSSatisfied = Satisfaction.IsSatisfied;

    if (BO.isOr() && IsLHSSatisfied)
      // [temp.constr.op] p3
      //    A disjunction is a constraint taking two operands. To determine if
      //    a disjunction is satisfied, the satisfaction of the first operand
      //    is checked. If that is satisfied, the disjunction is satisfied.
      //    Otherwise, the disjunction is satisfied if and only if the second
      //    operand is satisfied.
      // LHS is instantiated while RHS is not. Skip creating invalid BinaryOp.
      return LHSRes;

    if (BO.isAnd() && !IsLHSSatisfied)
      // [temp.constr.op] p2
      //    A conjunction is a constraint taking two operands. To determine if
      //    a conjunction is satisfied, the satisfaction of the first operand
      //    is checked. If that is not satisfied, the conjunction is not
      //    satisfied. Otherwise, the conjunction is satisfied if and only if
      //    the second operand is satisfied.
      // LHS is instantiated while RHS is not. Skip creating invalid BinaryOp.
      return LHSRes;

    ExprResult RHSRes = calculateConstraintSatisfaction(
        S, BO.getRHS(), Satisfaction, std::forward<AtomicEvaluator>(Evaluator));
    if (RHSRes.isInvalid())
      return ExprError();

    return BO.recreateBinOp(S, LHSRes, RHSRes);
  }

  if (auto *C = dyn_cast<ExprWithCleanups>(ConstraintExpr)) {
    // These aren't evaluated, so we don't care about cleanups, so we can just
    // evaluate these as if the cleanups didn't exist.
    return calculateConstraintSatisfaction(
        S, C->getSubExpr(), Satisfaction,
        std::forward<AtomicEvaluator>(Evaluator));
  }

  // An atomic constraint expression
  ExprResult SubstitutedAtomicExpr = Evaluator(ConstraintExpr);

  if (SubstitutedAtomicExpr.isInvalid())
    return ExprError();

  if (!SubstitutedAtomicExpr.isUsable())
    // Evaluator has decided satisfaction without yielding an expression.
    return ExprEmpty();

  // We don't have the ability to evaluate this, since it contains a
  // RecoveryExpr, so we want to fail overload resolution.  Otherwise,
  // we'd potentially pick up a different overload, and cause confusing
  // diagnostics. SO, add a failure detail that will cause us to make this
  // overload set not viable.
  if (SubstitutedAtomicExpr.get()->containsErrors()) {
    Satisfaction.IsSatisfied = false;
    Satisfaction.ContainsErrors = true;

    PartialDiagnostic Msg = S.PDiag(diag::note_constraint_references_error);
    SmallString<128> DiagString;
    DiagString = ": ";
    Msg.EmitToString(S.getDiagnostics(), DiagString);
    unsigned MessageSize = DiagString.size();
    char *Mem = new (S.Context) char[MessageSize];
    memcpy(Mem, DiagString.c_str(), MessageSize);
    Satisfaction.Details.emplace_back(
        ConstraintExpr,
        new (S.Context) ConstraintSatisfaction::SubstitutionDiagnostic{
            SubstitutedAtomicExpr.get()->getBeginLoc(),
            StringRef(Mem, MessageSize)});
    return SubstitutedAtomicExpr;
  }

  EnterExpressionEvaluationContext ConstantEvaluated(
      S, Sema::ExpressionEvaluationContext::ConstantEvaluated);
  SmallVector<PartialDiagnosticAt, 2> EvaluationDiags;
  Expr::EvalResult EvalResult;
  EvalResult.Diag = &EvaluationDiags;
  if (!SubstitutedAtomicExpr.get()->EvaluateAsConstantExpr(EvalResult,
                                                           S.Context) ||
      !EvaluationDiags.empty()) {
    // C++2a [temp.constr.atomic]p1
    //   ...E shall be a constant expression of type bool.
    S.Diag(SubstitutedAtomicExpr.get()->getBeginLoc(),
           diag::err_non_constant_constraint_expression)
        << SubstitutedAtomicExpr.get()->getSourceRange();
    for (const PartialDiagnosticAt &PDiag : EvaluationDiags)
      S.Diag(PDiag.first, PDiag.second);
    return ExprError();
  }

  assert(EvalResult.Val.isInt() &&
         "evaluating bool expression didn't produce int");
  Satisfaction.IsSatisfied = EvalResult.Val.getInt().getBoolValue();
  if (!Satisfaction.IsSatisfied)
    Satisfaction.Details.emplace_back(ConstraintExpr,
                                      SubstitutedAtomicExpr.get());

  return SubstitutedAtomicExpr;
}

static bool
DiagRecursiveConstraintEval(Sema &S, llvm::FoldingSetNodeID &ID, const Expr *E,
                            const MultiLevelTemplateArgumentList &MLTAL) {
  E->Profile(ID, S.Context, /*Canonical=*/true);
  for (const auto &List : MLTAL)
    for (const auto &TemplateArg : List.Args)
      TemplateArg.Profile(ID, S.Context);

  // Note that we have to do this with our own collection, because there are
  // times where a constraint-expression check can cause us to need to evaluate
  // other constriants that are unrelated, such as when evaluating a recovery
  // expression, or when trying to determine the constexpr-ness of special
  // members. Otherwise we could just use the
  // Sema::InstantiatingTemplate::isAlreadyBeingInstantiated function.
  if (S.SatisfactionStackContains(ID)) {
    S.Diag(E->getExprLoc(), diag::err_constraint_depends_on_self)
        << const_cast<Expr *>(E) << E->getSourceRange();
    return true;
  }

  return false;
}

static ExprResult calculateConstraintSatisfaction(
    Sema &S, const NamedDecl *Template, SourceLocation TemplateNameLoc,
    const MultiLevelTemplateArgumentList &MLTAL, const Expr *ConstraintExpr,
    ConstraintSatisfaction &Satisfaction) {
  return calculateConstraintSatisfaction(
      S, ConstraintExpr, Satisfaction, [&](const Expr *AtomicExpr) {
        EnterExpressionEvaluationContext ConstantEvaluated(
            S, Sema::ExpressionEvaluationContext::ConstantEvaluated,
            Sema::ReuseLambdaContextDecl);

        // Atomic constraint - substitute arguments and check satisfaction.
        ExprResult SubstitutedExpression;
        {
          TemplateDeductionInfo Info(TemplateNameLoc);
          Sema::InstantiatingTemplate Inst(S, AtomicExpr->getBeginLoc(),
              Sema::InstantiatingTemplate::ConstraintSubstitution{},
              const_cast<NamedDecl *>(Template), Info,
              AtomicExpr->getSourceRange());
          if (Inst.isInvalid())
            return ExprError();

          llvm::FoldingSetNodeID ID;
          if (DiagRecursiveConstraintEval(S, ID, AtomicExpr, MLTAL)) {
            Satisfaction.IsSatisfied = false;
            Satisfaction.ContainsErrors = true;
            return ExprEmpty();
          }

          SatisfactionStackRAII StackRAII(S, ID);

          // We do not want error diagnostics escaping here.
          Sema::SFINAETrap Trap(S);
          SubstitutedExpression =
              S.SubstConstraintExpr(const_cast<Expr *>(AtomicExpr), MLTAL);
          // Substitution might have stripped off a contextual conversion to
          // bool if this is the operand of an '&&' or '||'. For example, we
          // might lose an lvalue-to-rvalue conversion here. If so, put it back
          // before we try to evaluate.
          if (SubstitutedExpression.isUsable() &&
              !SubstitutedExpression.isInvalid())
            SubstitutedExpression =
                S.PerformContextuallyConvertToBool(SubstitutedExpression.get());
          if (SubstitutedExpression.isInvalid() || Trap.hasErrorOccurred()) {
            // C++2a [temp.constr.atomic]p1
            //   ...If substitution results in an invalid type or expression, the
            //   constraint is not satisfied.
            if (!Trap.hasErrorOccurred())
              // A non-SFINAE error has occurred as a result of this
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
            unsigned MessageSize = DiagString.size();
            char *Mem = new (S.Context) char[MessageSize];
            memcpy(Mem, DiagString.c_str(), MessageSize);
            Satisfaction.Details.emplace_back(
                AtomicExpr,
                new (S.Context) ConstraintSatisfaction::SubstitutionDiagnostic{
                        SubstDiag.first, StringRef(Mem, MessageSize)});
            Satisfaction.IsSatisfied = false;
            return ExprEmpty();
          }
        }

        if (!S.CheckConstraintExpression(SubstitutedExpression.get()))
          return ExprError();

        return SubstitutedExpression;
      });
}

static bool CheckConstraintSatisfaction(
    Sema &S, const NamedDecl *Template, ArrayRef<const Expr *> ConstraintExprs,
    llvm::SmallVectorImpl<Expr *> &Converted,
    const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange, ConstraintSatisfaction &Satisfaction) {
  if (ConstraintExprs.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  if (TemplateArgsLists.isAnyArgInstantiationDependent()) {
    // No need to check satisfaction for dependent constraint expressions.
    Satisfaction.IsSatisfied = true;
    return false;
  }

  ArrayRef<TemplateArgument> TemplateArgs =
      TemplateArgsLists.getNumSubstitutedLevels() > 0
          ? TemplateArgsLists.getOutermost()
          : ArrayRef<TemplateArgument> {};
  Sema::InstantiatingTemplate Inst(S, TemplateIDRange.getBegin(),
      Sema::InstantiatingTemplate::ConstraintsCheck{},
      const_cast<NamedDecl *>(Template), TemplateArgs, TemplateIDRange);
  if (Inst.isInvalid())
    return true;

  for (const Expr *ConstraintExpr : ConstraintExprs) {
    ExprResult Res = calculateConstraintSatisfaction(
        S, Template, TemplateIDRange.getBegin(), TemplateArgsLists,
        ConstraintExpr, Satisfaction);
    if (Res.isInvalid())
      return true;

    Converted.push_back(Res.get());
    if (!Satisfaction.IsSatisfied) {
      // Backfill the 'converted' list with nulls so we can keep the Converted
      // and unconverted lists in sync.
      Converted.append(ConstraintExprs.size() - Converted.size(), nullptr);
      // [temp.constr.op] p2
      // [...] To determine if a conjunction is satisfied, the satisfaction
      // of the first operand is checked. If that is not satisfied, the
      // conjunction is not satisfied. [...]
      return false;
    }
  }
  return false;
}

bool Sema::CheckConstraintSatisfaction(
    const NamedDecl *Template, ArrayRef<const Expr *> ConstraintExprs,
    llvm::SmallVectorImpl<Expr *> &ConvertedConstraints,
    const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange, ConstraintSatisfaction &OutSatisfaction) {
  if (ConstraintExprs.empty()) {
    OutSatisfaction.IsSatisfied = true;
    return false;
  }
  if (!Template) {
    return ::CheckConstraintSatisfaction(
        *this, nullptr, ConstraintExprs, ConvertedConstraints,
        TemplateArgsLists, TemplateIDRange, OutSatisfaction);
  }

  // A list of the template argument list flattened in a predictible manner for
  // the purposes of caching. The ConstraintSatisfaction type is in AST so it
  // has no access to the MultiLevelTemplateArgumentList, so this has to happen
  // here.
  llvm::SmallVector<TemplateArgument, 4> FlattenedArgs;
  for (auto List : TemplateArgsLists)
    FlattenedArgs.insert(FlattenedArgs.end(), List.Args.begin(),
                         List.Args.end());

  llvm::FoldingSetNodeID ID;
  ConstraintSatisfaction::Profile(ID, Context, Template, FlattenedArgs);
  void *InsertPos;
  if (auto *Cached = SatisfactionCache.FindNodeOrInsertPos(ID, InsertPos)) {
    OutSatisfaction = *Cached;
    return false;
  }

  auto Satisfaction =
      std::make_unique<ConstraintSatisfaction>(Template, FlattenedArgs);
  if (::CheckConstraintSatisfaction(*this, Template, ConstraintExprs,
                                    ConvertedConstraints, TemplateArgsLists,
                                    TemplateIDRange, *Satisfaction)) {
    OutSatisfaction = *Satisfaction;
    return true;
  }

  if (auto *Cached = SatisfactionCache.FindNodeOrInsertPos(ID, InsertPos)) {
    // The evaluation of this constraint resulted in us trying to re-evaluate it
    // recursively. This isn't really possible, except we try to form a
    // RecoveryExpr as a part of the evaluation.  If this is the case, just
    // return the 'cached' version (which will have the same result), and save
    // ourselves the extra-insert. If it ever becomes possible to legitimately
    // recursively check a constraint, we should skip checking the 'inner' one
    // above, and replace the cached version with this one, as it would be more
    // specific.
    OutSatisfaction = *Cached;
    return false;
  }

  // Else we can simply add this satisfaction to the list.
  OutSatisfaction = *Satisfaction;
  // We cannot use InsertPos here because CheckConstraintSatisfaction might have
  // invalidated it.
  // Note that entries of SatisfactionCache are deleted in Sema's destructor.
  SatisfactionCache.InsertNode(Satisfaction.release());
  return false;
}

bool Sema::CheckConstraintSatisfaction(const Expr *ConstraintExpr,
                                       ConstraintSatisfaction &Satisfaction) {
  return calculateConstraintSatisfaction(
             *this, ConstraintExpr, Satisfaction,
             [this](const Expr *AtomicExpr) -> ExprResult {
               // We only do this to immitate lvalue-to-rvalue conversion.
               return PerformContextuallyConvertToBool(
                   const_cast<Expr *>(AtomicExpr));
             })
      .isInvalid();
}

bool Sema::SetupConstraintScope(
    FunctionDecl *FD, llvm::Optional<ArrayRef<TemplateArgument>> TemplateArgs,
    MultiLevelTemplateArgumentList MLTAL, LocalInstantiationScope &Scope) {
  if (FD->isTemplateInstantiation() && FD->getPrimaryTemplate()) {
    FunctionTemplateDecl *PrimaryTemplate = FD->getPrimaryTemplate();
    InstantiatingTemplate Inst(
        *this, FD->getPointOfInstantiation(),
        Sema::InstantiatingTemplate::ConstraintsCheck{}, PrimaryTemplate,
        TemplateArgs ? *TemplateArgs : ArrayRef<TemplateArgument>{},
        SourceRange());
    if (Inst.isInvalid())
      return true;

    // addInstantiatedParametersToScope creates a map of 'uninstantiated' to
    // 'instantiated' parameters and adds it to the context. For the case where
    // this function is a template being instantiated NOW, we also need to add
    // the list of current template arguments to the list so that they also can
    // be picked out of the map.
    if (auto *SpecArgs = FD->getTemplateSpecializationArgs()) {
      MultiLevelTemplateArgumentList JustTemplArgs(FD, SpecArgs->asArray(),
                                                   /*Final=*/false);
      if (addInstantiatedParametersToScope(
              FD, PrimaryTemplate->getTemplatedDecl(), Scope, JustTemplArgs))
        return true;
    }

    // If this is a member function, make sure we get the parameters that
    // reference the original primary template.
    if (const auto *FromMemTempl =
            PrimaryTemplate->getInstantiatedFromMemberTemplate()) {
      if (addInstantiatedParametersToScope(FD, FromMemTempl->getTemplatedDecl(),
                                           Scope, MLTAL))
        return true;
    }

    return false;
  }

  if (FD->getTemplatedKind() == FunctionDecl::TK_MemberSpecialization ||
      FD->getTemplatedKind() == FunctionDecl::TK_DependentNonTemplate) {
    FunctionDecl *InstantiatedFrom =
        FD->getTemplatedKind() == FunctionDecl::TK_MemberSpecialization
            ? FD->getInstantiatedFromMemberFunction()
            : FD->getInstantiatedFromDecl();

    InstantiatingTemplate Inst(
        *this, FD->getPointOfInstantiation(),
        Sema::InstantiatingTemplate::ConstraintsCheck{}, InstantiatedFrom,
        TemplateArgs ? *TemplateArgs : ArrayRef<TemplateArgument>{},
        SourceRange());
    if (Inst.isInvalid())
      return true;

    // Case where this was not a template, but instantiated as a
    // child-function.
    if (addInstantiatedParametersToScope(FD, InstantiatedFrom, Scope, MLTAL))
      return true;
  }

  return false;
}

// This function collects all of the template arguments for the purposes of
// constraint-instantiation and checking.
llvm::Optional<MultiLevelTemplateArgumentList>
Sema::SetupConstraintCheckingTemplateArgumentsAndScope(
    FunctionDecl *FD, llvm::Optional<ArrayRef<TemplateArgument>> TemplateArgs,
    LocalInstantiationScope &Scope) {
  MultiLevelTemplateArgumentList MLTAL;

  // Collect the list of template arguments relative to the 'primary' template.
  // We need the entire list, since the constraint is completely uninstantiated
  // at this point.
  MLTAL =
      getTemplateInstantiationArgs(FD, /*Final=*/false, /*Innermost=*/nullptr,
                                   /*RelativeToPrimary=*/true,
                                   /*Pattern=*/nullptr,
                                   /*ForConstraintInstantiation=*/true);
  if (SetupConstraintScope(FD, TemplateArgs, MLTAL, Scope))
    return std::nullopt;

  return MLTAL;
}

bool Sema::CheckFunctionConstraints(const FunctionDecl *FD,
                                    ConstraintSatisfaction &Satisfaction,
                                    SourceLocation UsageLoc,
                                    bool ForOverloadResolution) {
  // Don't check constraints if the function is dependent. Also don't check if
  // this is a function template specialization, as the call to
  // CheckinstantiatedFunctionTemplateConstraints after this will check it
  // better.
  if (FD->isDependentContext() ||
      FD->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  DeclContext *CtxToSave = const_cast<FunctionDecl *>(FD);

  while (isLambdaCallOperator(CtxToSave) || FD->isTransparentContext()) {
    if (isLambdaCallOperator(CtxToSave))
      CtxToSave = CtxToSave->getParent()->getParent();
    else
      CtxToSave = CtxToSave->getNonTransparentContext();
  }

  ContextRAII SavedContext{*this, CtxToSave};
  LocalInstantiationScope Scope(*this, !ForOverloadResolution ||
                                           isLambdaCallOperator(FD));
  llvm::Optional<MultiLevelTemplateArgumentList> MLTAL =
      SetupConstraintCheckingTemplateArgumentsAndScope(
          const_cast<FunctionDecl *>(FD), {}, Scope);

  if (!MLTAL)
    return true;

  Qualifiers ThisQuals;
  CXXRecordDecl *Record = nullptr;
  if (auto *Method = dyn_cast<CXXMethodDecl>(FD)) {
    ThisQuals = Method->getMethodQualifiers();
    Record = const_cast<CXXRecordDecl *>(Method->getParent());
  }
  CXXThisScopeRAII ThisScope(*this, Record, ThisQuals, Record != nullptr);
  // We substitute with empty arguments in order to rebuild the atomic
  // constraint in a constant-evaluated context.
  // FIXME: Should this be a dedicated TreeTransform?
  const Expr *RC = FD->getTrailingRequiresClause();
  llvm::SmallVector<Expr *, 1> Converted;

  if (CheckConstraintSatisfaction(
          FD, {RC}, Converted, *MLTAL,
          SourceRange(UsageLoc.isValid() ? UsageLoc : FD->getLocation()),
          Satisfaction))
    return true;

  // FIXME: we need to do this for the function constraints for
  // comparison of constraints to work, but do we also need to do it for
  // CheckInstantiatedFunctionConstraints?  That one is more difficult, but we
  // seem to always just pick up the constraints from the primary template.
  assert(Converted.size() <= 1 && "Got more expressions converted?");
  if (!Converted.empty() && Converted[0] != nullptr)
    const_cast<FunctionDecl *>(FD)->setTrailingRequiresClause(Converted[0]);
  return false;
}


// Figure out the to-translation-unit depth for this function declaration for
// the purpose of seeing if they differ by constraints. This isn't the same as
// getTemplateDepth, because it includes already instantiated parents.
static unsigned
CalculateTemplateDepthForConstraints(Sema &S, const NamedDecl *ND,
                                     bool SkipForSpecialization = false) {
  MultiLevelTemplateArgumentList MLTAL = S.getTemplateInstantiationArgs(
      ND, /*Final=*/false, /*Innermost=*/nullptr, /*RelativeToPrimary=*/true,
      /*Pattern=*/nullptr,
      /*ForConstraintInstantiation=*/true, SkipForSpecialization);
  return MLTAL.getNumSubstitutedLevels();
}

namespace {
  class AdjustConstraintDepth : public TreeTransform<AdjustConstraintDepth> {
  unsigned TemplateDepth = 0;
  public:
  using inherited = TreeTransform<AdjustConstraintDepth>;
  AdjustConstraintDepth(Sema &SemaRef, unsigned TemplateDepth)
      : inherited(SemaRef), TemplateDepth(TemplateDepth) {}

  using inherited::TransformTemplateTypeParmType;
  QualType TransformTemplateTypeParmType(TypeLocBuilder &TLB,
                                         TemplateTypeParmTypeLoc TL, bool) {
    const TemplateTypeParmType *T = TL.getTypePtr();

    TemplateTypeParmDecl *NewTTPDecl = nullptr;
    if (TemplateTypeParmDecl *OldTTPDecl = T->getDecl())
      NewTTPDecl = cast_or_null<TemplateTypeParmDecl>(
          TransformDecl(TL.getNameLoc(), OldTTPDecl));

    QualType Result = getSema().Context.getTemplateTypeParmType(
        T->getDepth() + TemplateDepth, T->getIndex(), T->isParameterPack(),
        NewTTPDecl);
    TemplateTypeParmTypeLoc NewTL = TLB.push<TemplateTypeParmTypeLoc>(Result);
    NewTL.setNameLoc(TL.getNameLoc());
    return Result;
  }
  };
} // namespace

bool Sema::AreConstraintExpressionsEqual(const NamedDecl *Old,
                                         const Expr *OldConstr,
                                         const NamedDecl *New,
                                         const Expr *NewConstr) {
  if (Old && New && Old != New) {
    unsigned Depth1 = CalculateTemplateDepthForConstraints(
        *this, Old);
    unsigned Depth2 = CalculateTemplateDepthForConstraints(
        *this, New);

    // Adjust the 'shallowest' verison of this to increase the depth to match
    // the 'other'.
    if (Depth2 > Depth1) {
      OldConstr = AdjustConstraintDepth(*this, Depth2 - Depth1)
                      .TransformExpr(const_cast<Expr *>(OldConstr))
                      .get();
    } else if (Depth1 > Depth2) {
      NewConstr = AdjustConstraintDepth(*this, Depth1 - Depth2)
                      .TransformExpr(const_cast<Expr *>(NewConstr))
                      .get();
    }
  }

  llvm::FoldingSetNodeID ID1, ID2;
  OldConstr->Profile(ID1, Context, /*Canonical=*/true);
  NewConstr->Profile(ID2, Context, /*Canonical=*/true);
  return ID1 == ID2;
}

bool Sema::FriendConstraintsDependOnEnclosingTemplate(const FunctionDecl *FD) {
  assert(FD->getFriendObjectKind() && "Must be a friend!");

  // The logic for non-templates is handled in ASTContext::isSameEntity, so we
  // don't have to bother checking 'DependsOnEnclosingTemplate' for a
  // non-function-template.
  assert(FD->getDescribedFunctionTemplate() &&
         "Non-function templates don't need to be checked");

  SmallVector<const Expr *, 3> ACs;
  FD->getDescribedFunctionTemplate()->getAssociatedConstraints(ACs);

  unsigned OldTemplateDepth = CalculateTemplateDepthForConstraints(*this, FD);
  for (const Expr *Constraint : ACs)
    if (ConstraintExpressionDependsOnEnclosingTemplate(FD, OldTemplateDepth,
                                                       Constraint))
      return true;

  return false;
}

bool Sema::EnsureTemplateArgumentListConstraints(
    TemplateDecl *TD, const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange) {
  ConstraintSatisfaction Satisfaction;
  llvm::SmallVector<const Expr *, 3> AssociatedConstraints;
  TD->getAssociatedConstraints(AssociatedConstraints);
  if (CheckConstraintSatisfaction(TD, AssociatedConstraints, TemplateArgsLists,
                                  TemplateIDRange, Satisfaction))
    return true;

  if (!Satisfaction.IsSatisfied) {
    SmallString<128> TemplateArgString;
    TemplateArgString = " ";
    TemplateArgString += getTemplateArgumentBindingsText(
        TD->getTemplateParameters(), TemplateArgsLists.getInnermost().data(),
        TemplateArgsLists.getInnermost().size());

    Diag(TemplateIDRange.getBegin(),
         diag::err_template_arg_list_constraints_not_satisfied)
        << (int)getTemplateNameKindForDiagnostics(TemplateName(TD)) << TD
        << TemplateArgString << TemplateIDRange;
    DiagnoseUnsatisfiedConstraint(Satisfaction);
    return true;
  }
  return false;
}

bool Sema::CheckInstantiatedFunctionTemplateConstraints(
    SourceLocation PointOfInstantiation, FunctionDecl *Decl,
    ArrayRef<TemplateArgument> TemplateArgs,
    ConstraintSatisfaction &Satisfaction) {
  // In most cases we're not going to have constraints, so check for that first.
  FunctionTemplateDecl *Template = Decl->getPrimaryTemplate();
  // Note - code synthesis context for the constraints check is created
  // inside CheckConstraintsSatisfaction.
  SmallVector<const Expr *, 3> TemplateAC;
  Template->getAssociatedConstraints(TemplateAC);
  if (TemplateAC.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  Sema::ContextRAII savedContext(*this, Decl);
  LocalInstantiationScope Scope(*this);

  Optional<MultiLevelTemplateArgumentList> MLTAL =
      SetupConstraintCheckingTemplateArgumentsAndScope(Decl, TemplateArgs,
                                                       Scope);

  if (!MLTAL)
    return true;

  Qualifiers ThisQuals;
  CXXRecordDecl *Record = nullptr;
  if (auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
    ThisQuals = Method->getMethodQualifiers();
    Record = Method->getParent();
  }
  CXXThisScopeRAII ThisScope(*this, Record, ThisQuals, Record != nullptr);
  FunctionScopeRAII FuncScope(*this);
  if (isLambdaCallOperator(Decl))
    PushLambdaScope();
  else
    FuncScope.disable();

  llvm::SmallVector<Expr *, 1> Converted;
  return CheckConstraintSatisfaction(Template, TemplateAC, Converted, *MLTAL,
                                     PointOfInstantiation, Satisfaction);
}

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::ExprRequirement *Req,
                                           bool First) {
  assert(!Req->isSatisfied()
         && "Diagnose() can only be used on an unsatisfied requirement");
  switch (Req->getSatisfactionStatus()) {
    case concepts::ExprRequirement::SS_Dependent:
      llvm_unreachable("Diagnosing a dependent requirement");
      break;
    case concepts::ExprRequirement::SS_ExprSubstitutionFailure: {
      auto *SubstDiag = Req->getExprSubstitutionDiagnostic();
      if (!SubstDiag->DiagMessage.empty())
        S.Diag(SubstDiag->DiagLoc,
               diag::note_expr_requirement_expr_substitution_error)
               << (int)First << SubstDiag->SubstitutedEntity
               << SubstDiag->DiagMessage;
      else
        S.Diag(SubstDiag->DiagLoc,
               diag::note_expr_requirement_expr_unknown_substitution_error)
            << (int)First << SubstDiag->SubstitutedEntity;
      break;
    }
    case concepts::ExprRequirement::SS_NoexceptNotMet:
      S.Diag(Req->getNoexceptLoc(),
             diag::note_expr_requirement_noexcept_not_met)
          << (int)First << Req->getExpr();
      break;
    case concepts::ExprRequirement::SS_TypeRequirementSubstitutionFailure: {
      auto *SubstDiag =
          Req->getReturnTypeRequirement().getSubstitutionDiagnostic();
      if (!SubstDiag->DiagMessage.empty())
        S.Diag(SubstDiag->DiagLoc,
               diag::note_expr_requirement_type_requirement_substitution_error)
            << (int)First << SubstDiag->SubstitutedEntity
            << SubstDiag->DiagMessage;
      else
        S.Diag(SubstDiag->DiagLoc,
               diag::note_expr_requirement_type_requirement_unknown_substitution_error)
            << (int)First << SubstDiag->SubstitutedEntity;
      break;
    }
    case concepts::ExprRequirement::SS_ConstraintsNotSatisfied: {
      ConceptSpecializationExpr *ConstraintExpr =
          Req->getReturnTypeRequirementSubstitutedConstraintExpr();
      if (ConstraintExpr->getTemplateArgsAsWritten()->NumTemplateArgs == 1) {
        // A simple case - expr type is the type being constrained and the concept
        // was not provided arguments.
        Expr *e = Req->getExpr();
        S.Diag(e->getBeginLoc(),
               diag::note_expr_requirement_constraints_not_satisfied_simple)
            << (int)First << S.Context.getReferenceQualifiedType(e)
            << ConstraintExpr->getNamedConcept();
      } else {
        S.Diag(ConstraintExpr->getBeginLoc(),
               diag::note_expr_requirement_constraints_not_satisfied)
            << (int)First << ConstraintExpr;
      }
      S.DiagnoseUnsatisfiedConstraint(ConstraintExpr->getSatisfaction());
      break;
    }
    case concepts::ExprRequirement::SS_Satisfied:
      llvm_unreachable("We checked this above");
  }
}

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::TypeRequirement *Req,
                                           bool First) {
  assert(!Req->isSatisfied()
         && "Diagnose() can only be used on an unsatisfied requirement");
  switch (Req->getSatisfactionStatus()) {
  case concepts::TypeRequirement::SS_Dependent:
    llvm_unreachable("Diagnosing a dependent requirement");
    return;
  case concepts::TypeRequirement::SS_SubstitutionFailure: {
    auto *SubstDiag = Req->getSubstitutionDiagnostic();
    if (!SubstDiag->DiagMessage.empty())
      S.Diag(SubstDiag->DiagLoc,
             diag::note_type_requirement_substitution_error) << (int)First
          << SubstDiag->SubstitutedEntity << SubstDiag->DiagMessage;
    else
      S.Diag(SubstDiag->DiagLoc,
             diag::note_type_requirement_unknown_substitution_error)
          << (int)First << SubstDiag->SubstitutedEntity;
    return;
  }
  default:
    llvm_unreachable("Unknown satisfaction status");
    return;
  }
}
static void diagnoseWellFormedUnsatisfiedConstraintExpr(Sema &S,
                                                        Expr *SubstExpr,
                                                        bool First = true);

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::NestedRequirement *Req,
                                           bool First) {
  using SubstitutionDiagnostic = std::pair<SourceLocation, StringRef>;
  for (auto &Pair : Req->getConstraintSatisfaction()) {
    if (auto *SubstDiag = Pair.second.dyn_cast<SubstitutionDiagnostic *>())
      S.Diag(SubstDiag->first, diag::note_nested_requirement_substitution_error)
          << (int)First << Req->getInvalidConstraintEntity() << SubstDiag->second;
    else
      diagnoseWellFormedUnsatisfiedConstraintExpr(
          S, Pair.second.dyn_cast<Expr *>(), First);
    First = false;
  }
}

static void diagnoseWellFormedUnsatisfiedConstraintExpr(Sema &S,
                                                        Expr *SubstExpr,
                                                        bool First) {
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
    case BO_LAnd: {
      bool LHSSatisfied =
          BO->getLHS()->EvaluateKnownConstInt(S.Context).getBoolValue();
      if (LHSSatisfied) {
        // LHS is true, so RHS must be false.
        diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getRHS(), First);
        return;
      }
      // LHS is false
      diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getLHS(), First);

      // RHS might also be false
      bool RHSSatisfied =
          BO->getRHS()->EvaluateKnownConstInt(S.Context).getBoolValue();
      if (!RHSSatisfied)
        diagnoseWellFormedUnsatisfiedConstraintExpr(S, BO->getRHS(),
                                                    /*First=*/false);
      return;
    }
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
        BO->getLHS()->EvaluateAsInt(SimplifiedLHS, S.Context,
                                    Expr::SE_NoSideEffects,
                                    /*InConstantContext=*/true);
        BO->getRHS()->EvaluateAsInt(SimplifiedRHS, S.Context,
                                    Expr::SE_NoSideEffects,
                                    /*InConstantContext=*/true);
        if (!SimplifiedLHS.Diag && ! SimplifiedRHS.Diag) {
          S.Diag(SubstExpr->getBeginLoc(),
                 diag::note_atomic_constraint_evaluated_to_false_elaborated)
              << (int)First << SubstExpr
              << toString(SimplifiedLHS.Val.getInt(), 10)
              << BinaryOperator::getOpcodeStr(BO->getOpcode())
              << toString(SimplifiedRHS.Val.getInt(), 10);
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
  } else if (auto *RE = dyn_cast<RequiresExpr>(SubstExpr)) {
    // FIXME: RequiresExpr should store dependent diagnostics.
    for (concepts::Requirement *Req : RE->getRequirements())
      if (!Req->isDependent() && !Req->isSatisfied()) {
        if (auto *E = dyn_cast<concepts::ExprRequirement>(Req))
          diagnoseUnsatisfiedRequirement(S, E, First);
        else if (auto *T = dyn_cast<concepts::TypeRequirement>(Req))
          diagnoseUnsatisfiedRequirement(S, T, First);
        else
          diagnoseUnsatisfiedRequirement(
              S, cast<concepts::NestedRequirement>(Req), First);
        break;
      }
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

void
Sema::DiagnoseUnsatisfiedConstraint(const ConstraintSatisfaction& Satisfaction,
                                    bool First) {
  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");
  for (auto &Pair : Satisfaction.Details) {
    diagnoseUnsatisfiedConstraintExpr(*this, Pair.first, Pair.second, First);
    First = false;
  }
}

void Sema::DiagnoseUnsatisfiedConstraint(
    const ASTConstraintSatisfaction &Satisfaction,
    bool First) {
  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");
  for (auto &Pair : Satisfaction) {
    diagnoseUnsatisfiedConstraintExpr(*this, Pair.first, Pair.second, First);
    First = false;
  }
}

const NormalizedConstraint *
Sema::getNormalizedAssociatedConstraints(
    NamedDecl *ConstrainedDecl, ArrayRef<const Expr *> AssociatedConstraints) {
  auto CacheEntry = NormalizationCache.find(ConstrainedDecl);
  if (CacheEntry == NormalizationCache.end()) {
    auto Normalized =
        NormalizedConstraint::fromConstraintExprs(*this, ConstrainedDecl,
                                                  AssociatedConstraints);
    CacheEntry =
        NormalizationCache
            .try_emplace(ConstrainedDecl,
                         Normalized
                             ? new (Context) NormalizedConstraint(
                                 std::move(*Normalized))
                             : nullptr)
            .first;
  }
  return CacheEntry->second;
}

static bool
substituteParameterMappings(Sema &S, NormalizedConstraint &N,
                            ConceptDecl *Concept,
                            const MultiLevelTemplateArgumentList &MLTAL,
                            const ASTTemplateArgumentListInfo *ArgsAsWritten) {
  if (!N.isAtomic()) {
    if (substituteParameterMappings(S, N.getLHS(), Concept, MLTAL,
                                    ArgsAsWritten))
      return true;
    return substituteParameterMappings(S, N.getRHS(), Concept, MLTAL,
                                       ArgsAsWritten);
  }
  TemplateParameterList *TemplateParams = Concept->getTemplateParameters();

  AtomicConstraint &Atomic = *N.getAtomicConstraint();
  TemplateArgumentListInfo SubstArgs;
  if (!Atomic.ParameterMapping) {
    llvm::SmallBitVector OccurringIndices(TemplateParams->size());
    S.MarkUsedTemplateParameters(Atomic.ConstraintExpr, /*OnlyDeduced=*/false,
                                 /*Depth=*/0, OccurringIndices);
    TemplateArgumentLoc *TempArgs =
        new (S.Context) TemplateArgumentLoc[OccurringIndices.count()];
    for (unsigned I = 0, J = 0, C = TemplateParams->size(); I != C; ++I)
      if (OccurringIndices[I])
        new (&(TempArgs)[J++])
            TemplateArgumentLoc(S.getIdentityTemplateArgumentLoc(
                TemplateParams->begin()[I],
                // Here we assume we do not support things like
                // template<typename A, typename B>
                // concept C = ...;
                //
                // template<typename... Ts> requires C<Ts...>
                // struct S { };
                // The above currently yields a diagnostic.
                // We still might have default arguments for concept parameters.
                ArgsAsWritten->NumTemplateArgs > I
                    ? ArgsAsWritten->arguments()[I].getLocation()
                    : SourceLocation()));
    Atomic.ParameterMapping.emplace(TempArgs,  OccurringIndices.count());
  }
  Sema::InstantiatingTemplate Inst(
      S, ArgsAsWritten->arguments().front().getSourceRange().getBegin(),
      Sema::InstantiatingTemplate::ParameterMappingSubstitution{}, Concept,
      SourceRange(ArgsAsWritten->arguments()[0].getSourceRange().getBegin(),
                  ArgsAsWritten->arguments().back().getSourceRange().getEnd()));
  if (S.SubstTemplateArguments(*Atomic.ParameterMapping, MLTAL, SubstArgs))
    return true;

  TemplateArgumentLoc *TempArgs =
      new (S.Context) TemplateArgumentLoc[SubstArgs.size()];
  std::copy(SubstArgs.arguments().begin(), SubstArgs.arguments().end(),
            TempArgs);
  Atomic.ParameterMapping.emplace(TempArgs, SubstArgs.size());
  return false;
}

static bool substituteParameterMappings(Sema &S, NormalizedConstraint &N,
                                        const ConceptSpecializationExpr *CSE) {
  TemplateArgumentList TAL{TemplateArgumentList::OnStack,
                           CSE->getTemplateArguments()};
  MultiLevelTemplateArgumentList MLTAL = S.getTemplateInstantiationArgs(
      CSE->getNamedConcept(), /*Final=*/true, &TAL,
      /*RelativeToPrimary=*/true,
      /*Pattern=*/nullptr,
      /*ForConstraintInstantiation=*/true);

  return substituteParameterMappings(S, N, CSE->getNamedConcept(), MLTAL,
                                     CSE->getTemplateArgsAsWritten());
}

Optional<NormalizedConstraint>
NormalizedConstraint::fromConstraintExprs(Sema &S, NamedDecl *D,
                                          ArrayRef<const Expr *> E) {
  assert(E.size() != 0);
  auto Conjunction = fromConstraintExpr(S, D, E[0]);
  if (!Conjunction)
    return std::nullopt;
  for (unsigned I = 1; I < E.size(); ++I) {
    auto Next = fromConstraintExpr(S, D, E[I]);
    if (!Next)
      return std::nullopt;
    *Conjunction = NormalizedConstraint(S.Context, std::move(*Conjunction),
                                        std::move(*Next), CCK_Conjunction);
  }
  return Conjunction;
}

llvm::Optional<NormalizedConstraint>
NormalizedConstraint::fromConstraintExpr(Sema &S, NamedDecl *D, const Expr *E) {
  assert(E != nullptr);

  // C++ [temp.constr.normal]p1.1
  // [...]
  // - The normal form of an expression (E) is the normal form of E.
  // [...]
  E = E->IgnoreParenImpCasts();

  // C++2a [temp.param]p4:
  //     [...] If T is not a pack, then E is E', otherwise E is (E' && ...).
  // Fold expression is considered atomic constraints per current wording.
  // See http://cplusplus.github.io/concepts-ts/ts-active.html#28

  if (LogicalBinOp BO = E) {
    auto LHS = fromConstraintExpr(S, D, BO.getLHS());
    if (!LHS)
      return std::nullopt;
    auto RHS = fromConstraintExpr(S, D, BO.getRHS());
    if (!RHS)
      return std::nullopt;

    return NormalizedConstraint(S.Context, std::move(*LHS), std::move(*RHS),
                                BO.isAnd() ? CCK_Conjunction : CCK_Disjunction);
  } else if (auto *CSE = dyn_cast<const ConceptSpecializationExpr>(E)) {
    const NormalizedConstraint *SubNF;
    {
      Sema::InstantiatingTemplate Inst(
          S, CSE->getExprLoc(),
          Sema::InstantiatingTemplate::ConstraintNormalization{}, D,
          CSE->getSourceRange());
      // C++ [temp.constr.normal]p1.1
      // [...]
      // The normal form of an id-expression of the form C<A1, A2, ..., AN>,
      // where C names a concept, is the normal form of the
      // constraint-expression of C, after substituting A1, A2, ..., AN for C’s
      // respective template parameters in the parameter mappings in each atomic
      // constraint. If any such substitution results in an invalid type or
      // expression, the program is ill-formed; no diagnostic is required.
      // [...]
      ConceptDecl *CD = CSE->getNamedConcept();
      SubNF = S.getNormalizedAssociatedConstraints(CD,
                                                   {CD->getConstraintExpr()});
      if (!SubNF)
        return std::nullopt;
    }

    Optional<NormalizedConstraint> New;
    New.emplace(S.Context, *SubNF);

    if (substituteParameterMappings(S, *New, CSE))
      return std::nullopt;

    return New;
  }
  return NormalizedConstraint{new (S.Context) AtomicConstraint(S, E)};
}

using NormalForm =
    llvm::SmallVector<llvm::SmallVector<AtomicConstraint *, 2>, 4>;

static NormalForm makeCNF(const NormalizedConstraint &Normalized) {
  if (Normalized.isAtomic())
    return {{Normalized.getAtomicConstraint()}};

  NormalForm LCNF = makeCNF(Normalized.getLHS());
  NormalForm RCNF = makeCNF(Normalized.getRHS());
  if (Normalized.getCompoundKind() == NormalizedConstraint::CCK_Conjunction) {
    LCNF.reserve(LCNF.size() + RCNF.size());
    while (!RCNF.empty())
      LCNF.push_back(RCNF.pop_back_val());
    return LCNF;
  }

  // Disjunction
  NormalForm Res;
  Res.reserve(LCNF.size() * RCNF.size());
  for (auto &LDisjunction : LCNF)
    for (auto &RDisjunction : RCNF) {
      NormalForm::value_type Combined;
      Combined.reserve(LDisjunction.size() + RDisjunction.size());
      std::copy(LDisjunction.begin(), LDisjunction.end(),
                std::back_inserter(Combined));
      std::copy(RDisjunction.begin(), RDisjunction.end(),
                std::back_inserter(Combined));
      Res.emplace_back(Combined);
    }
  return Res;
}

static NormalForm makeDNF(const NormalizedConstraint &Normalized) {
  if (Normalized.isAtomic())
    return {{Normalized.getAtomicConstraint()}};

  NormalForm LDNF = makeDNF(Normalized.getLHS());
  NormalForm RDNF = makeDNF(Normalized.getRHS());
  if (Normalized.getCompoundKind() == NormalizedConstraint::CCK_Disjunction) {
    LDNF.reserve(LDNF.size() + RDNF.size());
    while (!RDNF.empty())
      LDNF.push_back(RDNF.pop_back_val());
    return LDNF;
  }

  // Conjunction
  NormalForm Res;
  Res.reserve(LDNF.size() * RDNF.size());
  for (auto &LConjunction : LDNF) {
    for (auto &RConjunction : RDNF) {
      NormalForm::value_type Combined;
      Combined.reserve(LConjunction.size() + RConjunction.size());
      std::copy(LConjunction.begin(), LConjunction.end(),
                std::back_inserter(Combined));
      std::copy(RConjunction.begin(), RConjunction.end(),
                std::back_inserter(Combined));
      Res.emplace_back(Combined);
    }
  }
  return Res;
}

template<typename AtomicSubsumptionEvaluator>
static bool subsumes(NormalForm PDNF, NormalForm QCNF,
                     AtomicSubsumptionEvaluator E) {
  // C++ [temp.constr.order] p2
  //   Then, P subsumes Q if and only if, for every disjunctive clause Pi in the
  //   disjunctive normal form of P, Pi subsumes every conjunctive clause Qj in
  //   the conjuctive normal form of Q, where [...]
  for (const auto &Pi : PDNF) {
    for (const auto &Qj : QCNF) {
      // C++ [temp.constr.order] p2
      //   - [...] a disjunctive clause Pi subsumes a conjunctive clause Qj if
      //     and only if there exists an atomic constraint Pia in Pi for which
      //     there exists an atomic constraint, Qjb, in Qj such that Pia
      //     subsumes Qjb.
      bool Found = false;
      for (const AtomicConstraint *Pia : Pi) {
        for (const AtomicConstraint *Qjb : Qj) {
          if (E(*Pia, *Qjb)) {
            Found = true;
            break;
          }
        }
        if (Found)
          break;
      }
      if (!Found)
        return false;
    }
  }
  return true;
}

template<typename AtomicSubsumptionEvaluator>
static bool subsumes(Sema &S, NamedDecl *DP, ArrayRef<const Expr *> P,
                     NamedDecl *DQ, ArrayRef<const Expr *> Q, bool &Subsumes,
                     AtomicSubsumptionEvaluator E) {
  // C++ [temp.constr.order] p2
  //   In order to determine if a constraint P subsumes a constraint Q, P is
  //   transformed into disjunctive normal form, and Q is transformed into
  //   conjunctive normal form. [...]
  auto *PNormalized = S.getNormalizedAssociatedConstraints(DP, P);
  if (!PNormalized)
    return true;
  const NormalForm PDNF = makeDNF(*PNormalized);

  auto *QNormalized = S.getNormalizedAssociatedConstraints(DQ, Q);
  if (!QNormalized)
    return true;
  const NormalForm QCNF = makeCNF(*QNormalized);

  Subsumes = subsumes(PDNF, QCNF, E);
  return false;
}

bool Sema::IsAtLeastAsConstrained(NamedDecl *D1,
                                  MutableArrayRef<const Expr *> AC1,
                                  NamedDecl *D2,
                                  MutableArrayRef<const Expr *> AC2,
                                  bool &Result) {
  if (const auto *FD1 = dyn_cast<FunctionDecl>(D1)) {
    auto IsExpectedEntity = [](const FunctionDecl *FD) {
      FunctionDecl::TemplatedKind Kind = FD->getTemplatedKind();
      return Kind == FunctionDecl::TK_NonTemplate ||
             Kind == FunctionDecl::TK_FunctionTemplate;
    };
    const auto *FD2 = dyn_cast<FunctionDecl>(D2);
    (void)IsExpectedEntity;
    (void)FD2;
    assert(IsExpectedEntity(FD1) && FD2 && IsExpectedEntity(FD2) &&
           "use non-instantiated function declaration for constraints partial "
           "ordering");
  }

  if (AC1.empty()) {
    Result = AC2.empty();
    return false;
  }
  if (AC2.empty()) {
    // TD1 has associated constraints and TD2 does not.
    Result = true;
    return false;
  }

  std::pair<NamedDecl *, NamedDecl *> Key{D1, D2};
  auto CacheEntry = SubsumptionCache.find(Key);
  if (CacheEntry != SubsumptionCache.end()) {
    Result = CacheEntry->second;
    return false;
  }

  unsigned Depth1 = CalculateTemplateDepthForConstraints(*this, D1, true);
  unsigned Depth2 = CalculateTemplateDepthForConstraints(*this, D2, true);

  for (size_t I = 0; I != AC1.size() && I != AC2.size(); ++I) {
    if (Depth2 > Depth1) {
      AC1[I] = AdjustConstraintDepth(*this, Depth2 - Depth1)
                   .TransformExpr(const_cast<Expr *>(AC1[I]))
                   .get();
    } else if (Depth1 > Depth2) {
      AC2[I] = AdjustConstraintDepth(*this, Depth1 - Depth2)
                   .TransformExpr(const_cast<Expr *>(AC2[I]))
                   .get();
    }
  }

  if (subsumes(*this, D1, AC1, D2, AC2, Result,
        [this] (const AtomicConstraint &A, const AtomicConstraint &B) {
          return A.subsumes(Context, B);
        }))
    return true;
  SubsumptionCache.try_emplace(Key, Result);
  return false;
}

bool Sema::MaybeEmitAmbiguousAtomicConstraintsDiagnostic(NamedDecl *D1,
    ArrayRef<const Expr *> AC1, NamedDecl *D2, ArrayRef<const Expr *> AC2) {
  if (isSFINAEContext())
    // No need to work here because our notes would be discarded.
    return false;

  if (AC1.empty() || AC2.empty())
    return false;

  auto NormalExprEvaluator =
      [this] (const AtomicConstraint &A, const AtomicConstraint &B) {
        return A.subsumes(Context, B);
      };

  const Expr *AmbiguousAtomic1 = nullptr, *AmbiguousAtomic2 = nullptr;
  auto IdenticalExprEvaluator =
      [&] (const AtomicConstraint &A, const AtomicConstraint &B) {
        if (!A.hasMatchingParameterMapping(Context, B))
          return false;
        const Expr *EA = A.ConstraintExpr, *EB = B.ConstraintExpr;
        if (EA == EB)
          return true;

        // Not the same source level expression - are the expressions
        // identical?
        llvm::FoldingSetNodeID IDA, IDB;
        EA->Profile(IDA, Context, /*Canonical=*/true);
        EB->Profile(IDB, Context, /*Canonical=*/true);
        if (IDA != IDB)
          return false;

        AmbiguousAtomic1 = EA;
        AmbiguousAtomic2 = EB;
        return true;
      };

  {
    // The subsumption checks might cause diagnostics
    SFINAETrap Trap(*this);
    auto *Normalized1 = getNormalizedAssociatedConstraints(D1, AC1);
    if (!Normalized1)
      return false;
    const NormalForm DNF1 = makeDNF(*Normalized1);
    const NormalForm CNF1 = makeCNF(*Normalized1);

    auto *Normalized2 = getNormalizedAssociatedConstraints(D2, AC2);
    if (!Normalized2)
      return false;
    const NormalForm DNF2 = makeDNF(*Normalized2);
    const NormalForm CNF2 = makeCNF(*Normalized2);

    bool Is1AtLeastAs2Normally = subsumes(DNF1, CNF2, NormalExprEvaluator);
    bool Is2AtLeastAs1Normally = subsumes(DNF2, CNF1, NormalExprEvaluator);
    bool Is1AtLeastAs2 = subsumes(DNF1, CNF2, IdenticalExprEvaluator);
    bool Is2AtLeastAs1 = subsumes(DNF2, CNF1, IdenticalExprEvaluator);
    if (Is1AtLeastAs2 == Is1AtLeastAs2Normally &&
        Is2AtLeastAs1 == Is2AtLeastAs1Normally)
      // Same result - no ambiguity was caused by identical atomic expressions.
      return false;
  }

  // A different result! Some ambiguous atomic constraint(s) caused a difference
  assert(AmbiguousAtomic1 && AmbiguousAtomic2);

  Diag(AmbiguousAtomic1->getBeginLoc(), diag::note_ambiguous_atomic_constraints)
      << AmbiguousAtomic1->getSourceRange();
  Diag(AmbiguousAtomic2->getBeginLoc(),
       diag::note_ambiguous_atomic_constraints_similar_expression)
      << AmbiguousAtomic2->getSourceRange();
  return true;
}

concepts::ExprRequirement::ExprRequirement(
    Expr *E, bool IsSimple, SourceLocation NoexceptLoc,
    ReturnTypeRequirement Req, SatisfactionStatus Status,
    ConceptSpecializationExpr *SubstitutedConstraintExpr) :
    Requirement(IsSimple ? RK_Simple : RK_Compound, Status == SS_Dependent,
                Status == SS_Dependent &&
                (E->containsUnexpandedParameterPack() ||
                 Req.containsUnexpandedParameterPack()),
                Status == SS_Satisfied), Value(E), NoexceptLoc(NoexceptLoc),
    TypeReq(Req), SubstitutedConstraintExpr(SubstitutedConstraintExpr),
    Status(Status) {
  assert((!IsSimple || (Req.isEmpty() && NoexceptLoc.isInvalid())) &&
         "Simple requirement must not have a return type requirement or a "
         "noexcept specification");
  assert((Status > SS_TypeRequirementSubstitutionFailure && Req.isTypeConstraint()) ==
         (SubstitutedConstraintExpr != nullptr));
}

concepts::ExprRequirement::ExprRequirement(
    SubstitutionDiagnostic *ExprSubstDiag, bool IsSimple,
    SourceLocation NoexceptLoc, ReturnTypeRequirement Req) :
    Requirement(IsSimple ? RK_Simple : RK_Compound, Req.isDependent(),
                Req.containsUnexpandedParameterPack(), /*IsSatisfied=*/false),
    Value(ExprSubstDiag), NoexceptLoc(NoexceptLoc), TypeReq(Req),
    Status(SS_ExprSubstitutionFailure) {
  assert((!IsSimple || (Req.isEmpty() && NoexceptLoc.isInvalid())) &&
         "Simple requirement must not have a return type requirement or a "
         "noexcept specification");
}

concepts::ExprRequirement::ReturnTypeRequirement::
ReturnTypeRequirement(TemplateParameterList *TPL) :
    TypeConstraintInfo(TPL, false) {
  assert(TPL->size() == 1);
  const TypeConstraint *TC =
      cast<TemplateTypeParmDecl>(TPL->getParam(0))->getTypeConstraint();
  assert(TC &&
         "TPL must have a template type parameter with a type constraint");
  auto *Constraint =
      cast<ConceptSpecializationExpr>(TC->getImmediatelyDeclaredConstraint());
  bool Dependent =
      Constraint->getTemplateArgsAsWritten() &&
      TemplateSpecializationType::anyInstantiationDependentTemplateArguments(
          Constraint->getTemplateArgsAsWritten()->arguments().drop_front(1));
  TypeConstraintInfo.setInt(Dependent ? true : false);
}

concepts::TypeRequirement::TypeRequirement(TypeSourceInfo *T) :
    Requirement(RK_Type, T->getType()->isInstantiationDependentType(),
                T->getType()->containsUnexpandedParameterPack(),
                // We reach this ctor with either dependent types (in which
                // IsSatisfied doesn't matter) or with non-dependent type in
                // which the existence of the type indicates satisfaction.
                /*IsSatisfied=*/true),
    Value(T),
    Status(T->getType()->isInstantiationDependentType() ? SS_Dependent
                                                        : SS_Satisfied) {}
