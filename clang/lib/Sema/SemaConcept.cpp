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

#include "clang/Sema/SemaConcept.h"
#include "TreeTransform.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"

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
  OverloadedOperatorKind getOp() const { return Op; }

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
} // namespace

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
    if (!PossibleNonPrimary)
      return;

    *PossibleNonPrimary =
        // We have the following case:
        // template<typename> requires func(0) struct S { };
        // The user probably isn't aware of the parentheses required around
        // the function call, and we're only going to parse 'func' as the
        // primary-expression, and complain that it is of non-bool type.
        //
        // However, if we're in a lambda, this might also be:
        // []<typename> requires var () {};
        // Which also looks like a function call due to the lambda parentheses,
        // but unlike the first case, isn't an error, so this check is skipped.
        (NextToken.is(tok::l_paren) &&
         (IsTrailingRequiresClause ||
          (Type->isDependentType() &&
           isa<UnresolvedLookupExpr>(ConstraintExpression) &&
           !dyn_cast_if_present<LambdaScopeInfo>(getCurFunction())) ||
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
         diag::err_non_bool_atomic_constraint)
        << Type << ConstraintExpression->getSourceRange();
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
  bool Inserted = false;
  SatisfactionStackRAII(Sema &SemaRef, const NamedDecl *ND,
                        const llvm::FoldingSetNodeID &FSNID)
      : SemaRef(SemaRef) {
    if (ND) {
      SemaRef.PushSatisfactionStackEntry(ND, FSNID);
      Inserted = true;
    }
  }
  ~SatisfactionStackRAII() {
    if (Inserted)
      SemaRef.PopSatisfactionStackEntry();
  }
};
} // namespace

static bool DiagRecursiveConstraintEval(
    Sema &S, llvm::FoldingSetNodeID &ID, const NamedDecl *Templ, const Expr *E,
    const MultiLevelTemplateArgumentList *MLTAL = nullptr) {
  E->Profile(ID, S.Context, /*Canonical=*/true);
  if (MLTAL) {
    for (const auto &List : *MLTAL)
      for (const auto &TemplateArg : List.Args)
        S.Context.getCanonicalTemplateArgument(TemplateArg)
            .Profile(ID, S.Context);
  }
  if (S.SatisfactionStackContains(Templ, ID)) {
    S.Diag(E->getExprLoc(), diag::err_constraint_depends_on_self)
        << E << E->getSourceRange();
    return true;
  }
  return false;
}

// Figure out the to-translation-unit depth for this function declaration for
// the purpose of seeing if they differ by constraints. This isn't the same as
// getTemplateDepth, because it includes already instantiated parents.
static unsigned
CalculateTemplateDepthForConstraints(Sema &S, const NamedDecl *ND,
                                     bool SkipForSpecialization = false) {
  MultiLevelTemplateArgumentList MLTAL = S.getTemplateInstantiationArgs(
      ND, ND->getLexicalDeclContext(), /*Final=*/false,
      /*Innermost=*/std::nullopt,
      /*RelativeToPrimary=*/true,
      /*Pattern=*/nullptr,
      /*ForConstraintInstantiation=*/true, SkipForSpecialization);
  return MLTAL.getNumLevels();
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

  bool AlreadyTransformed(QualType T) {
    if (T.isNull())
      return true;

    if (T->isInstantiationDependentType() || T->isVariablyModifiedType() ||
        T->containsUnexpandedParameterPack())
      return false;
    return true;
  }
};
} // namespace

namespace {

// FIXME: Convert it to DynamicRecursiveASTVisitor
class HashParameterMapping : public RecursiveASTVisitor<HashParameterMapping> {
  using inherited = RecursiveASTVisitor<HashParameterMapping>;
  friend inherited;

  Sema &SemaRef;
  const MultiLevelTemplateArgumentList &TemplateArgs;
  llvm::FoldingSetNodeID &ID;
  llvm::SmallVector<TemplateArgument, 10> UsedTemplateArgs;

  UnsignedOrNone OuterPackSubstIndex;

  bool shouldVisitTemplateInstantiations() const { return true; }

public:
  HashParameterMapping(Sema &SemaRef,
                       const MultiLevelTemplateArgumentList &TemplateArgs,
                       llvm::FoldingSetNodeID &ID,
                       UnsignedOrNone OuterPackSubstIndex)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs), ID(ID),
        OuterPackSubstIndex(OuterPackSubstIndex) {}

  bool VisitTemplateTypeParmType(TemplateTypeParmType *T) {
    // A lambda expression can introduce template parameters that don't have
    // corresponding template arguments yet.
    if (T->getDepth() >= TemplateArgs.getNumLevels())
      return true;

    TemplateArgument Arg = TemplateArgs(T->getDepth(), T->getIndex());

    if (T->isParameterPack() && SemaRef.ArgPackSubstIndex) {
      assert(Arg.getKind() == TemplateArgument::Pack &&
             "Missing argument pack");

      Arg = SemaRef.getPackSubstitutedTemplateArgument(Arg);
    }

    UsedTemplateArgs.push_back(
        SemaRef.Context.getCanonicalTemplateArgument(Arg));
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    NamedDecl *D = E->getDecl();
    NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D);
    if (!NTTP)
      return TraverseDecl(D);

    TemplateArgument Arg = TemplateArgs(NTTP->getDepth(), NTTP->getPosition());
    if (NTTP->isParameterPack() && SemaRef.ArgPackSubstIndex) {
      assert(Arg.getKind() == TemplateArgument::Pack &&
             "Missing argument pack");
      Arg = SemaRef.getPackSubstitutedTemplateArgument(Arg);
    }

    UsedTemplateArgs.push_back(
        SemaRef.Context.getCanonicalTemplateArgument(Arg));
    return true;
  }

  bool VisitTypedefType(TypedefType *TT) {
    return inherited::TraverseType(TT->desugar());
  }

  bool TraverseDecl(Decl *D) {
    if (auto *VD = dyn_cast<ValueDecl>(D)) {
      if (auto *Var = dyn_cast<VarDecl>(VD))
        TraverseStmt(Var->getInit());
      return TraverseType(VD->getType());
    }

    return inherited::TraverseDecl(D);
  }

  bool TraverseTypeLoc(TypeLoc TL, bool TraverseQualifier = true) {
    // We don't care about TypeLocs. So traverse Types instead.
    return TraverseType(TL.getType(), TraverseQualifier);
  }

  bool TraverseTagType(const TagType *T, bool TraverseQualifier) {
    // T's parent can be dependent while T doesn't have any template arguments.
    // We should have already traversed its qualifier.
    // FIXME: Add an assert to catch cases where we failed to profile the
    // concept. assert(!T->isDependentType() && "We missed a case in profiling
    // concepts!");
    return true;
  }

  bool TraverseInjectedClassNameType(InjectedClassNameType *T,
                                     bool TraverseQualifier) {
    return TraverseTemplateArguments(T->getTemplateArgs(SemaRef.Context));
  }

  bool TraverseTemplateArgument(const TemplateArgument &Arg) {
    if (!Arg.containsUnexpandedParameterPack() || Arg.isPackExpansion()) {
      // Act as if we are fully expanding this pack, if it is a PackExpansion.
      Sema::ArgPackSubstIndexRAII _1(SemaRef, std::nullopt);
      llvm::SaveAndRestore<UnsignedOrNone> _2(OuterPackSubstIndex,
                                              std::nullopt);
      return inherited::TraverseTemplateArgument(Arg);
    }

    Sema::ArgPackSubstIndexRAII _1(SemaRef, OuterPackSubstIndex);
    return inherited::TraverseTemplateArgument(Arg);
  }

  bool TraverseSizeOfPackExpr(SizeOfPackExpr *SOPE) {
    return TraverseDecl(SOPE->getPack());
  }

  bool VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    return inherited::TraverseStmt(E->getReplacement());
  }

  void VisitConstraint(const NormalizedConstraintWithParamMapping &Constraint) {
    if (!Constraint.hasParameterMapping()) {
      for (const auto &List : TemplateArgs)
        for (const TemplateArgument &Arg : List.Args)
          SemaRef.Context.getCanonicalTemplateArgument(Arg).Profile(
              ID, SemaRef.Context);
      return;
    }

    llvm::ArrayRef<TemplateArgumentLoc> Mapping =
        Constraint.getParameterMapping();
    for (auto &ArgLoc : Mapping) {
      TemplateArgument Canonical =
          SemaRef.Context.getCanonicalTemplateArgument(ArgLoc.getArgument());
      // We don't want sugars to impede the profile of cache.
      UsedTemplateArgs.push_back(Canonical);
      TraverseTemplateArgument(Canonical);
    }

    for (auto &Used : UsedTemplateArgs) {
      llvm::FoldingSetNodeID R;
      Used.Profile(R, SemaRef.Context);
      ID.AddNodeID(R);
    }
  }
};

class ConstraintSatisfactionChecker {
  Sema &S;
  const NamedDecl *Template;
  SourceLocation TemplateNameLoc;
  UnsignedOrNone PackSubstitutionIndex;

  ConstraintSatisfaction &Satisfaction;

private:
  ExprResult
  EvaluateAtomicConstraint(const Expr *AtomicExpr,
                           const MultiLevelTemplateArgumentList &MLTAL);

  UnsignedOrNone EvaluateFoldExpandedConstraintSize(
      const FoldExpandedConstraint &FE,
      const MultiLevelTemplateArgumentList &MLTAL);

  // XXX: It is SLOW! Use it very carefully.
  std::optional<MultiLevelTemplateArgumentList> SubstitutionInTemplateArguments(
      const NormalizedConstraintWithParamMapping &Constraint,
      MultiLevelTemplateArgumentList MLTAL,
      llvm::SmallVector<TemplateArgument> &SubstitutedOuterMost);

  ExprResult EvaluateSlow(const AtomicConstraint &Constraint,
                          const MultiLevelTemplateArgumentList &MLTAL);

  ExprResult Evaluate(const AtomicConstraint &Constraint,
                      const MultiLevelTemplateArgumentList &MLTAL);

  ExprResult EvaluateSlow(const FoldExpandedConstraint &Constraint,
                          const MultiLevelTemplateArgumentList &MLTAL);

  ExprResult Evaluate(const FoldExpandedConstraint &Constraint,
                      const MultiLevelTemplateArgumentList &MLTAL);

  ExprResult EvaluateSlow(const ConceptIdConstraint &Constraint,
                          const MultiLevelTemplateArgumentList &MLTAL,
                          unsigned int Size);

  ExprResult Evaluate(const ConceptIdConstraint &Constraint,
                      const MultiLevelTemplateArgumentList &MLTAL);

  ExprResult Evaluate(const CompoundConstraint &Constraint,
                      const MultiLevelTemplateArgumentList &MLTAL);

public:
  ConstraintSatisfactionChecker(Sema &SemaRef, const NamedDecl *Template,
                                SourceLocation TemplateNameLoc,
                                UnsignedOrNone PackSubstitutionIndex,
                                ConstraintSatisfaction &Satisfaction)
      : S(SemaRef), Template(Template), TemplateNameLoc(TemplateNameLoc),
        PackSubstitutionIndex(PackSubstitutionIndex),
        Satisfaction(Satisfaction) {}

  ExprResult Evaluate(const NormalizedConstraint &Constraint,
                      const MultiLevelTemplateArgumentList &MLTAL);
};

StringRef allocateStringFromConceptDiagnostic(const Sema &S,
                                              const PartialDiagnostic Diag) {
  SmallString<128> DiagString;
  DiagString = ": ";
  Diag.EmitToString(S.getDiagnostics(), DiagString);
  return S.getASTContext().backupStr(DiagString);
}

} // namespace

ExprResult ConstraintSatisfactionChecker::EvaluateAtomicConstraint(
    const Expr *AtomicExpr, const MultiLevelTemplateArgumentList &MLTAL) {
  EnterExpressionEvaluationContext ConstantEvaluated(
      S, Sema::ExpressionEvaluationContext::ConstantEvaluated,
      Sema::ReuseLambdaContextDecl);

  llvm::FoldingSetNodeID ID;
  if (Template &&
      DiagRecursiveConstraintEval(S, ID, Template, AtomicExpr, &MLTAL)) {
    Satisfaction.IsSatisfied = false;
    Satisfaction.ContainsErrors = true;
    return ExprEmpty();
  }
  SatisfactionStackRAII StackRAII(S, Template, ID);

  // Atomic constraint - substitute arguments and check satisfaction.
  ExprResult SubstitutedExpression = const_cast<Expr *>(AtomicExpr);
  {
    TemplateDeductionInfo Info(TemplateNameLoc);
    Sema::InstantiatingTemplate Inst(
        S, AtomicExpr->getBeginLoc(),
        Sema::InstantiatingTemplate::ConstraintSubstitution{},
        // FIXME: improve const-correctness of InstantiatingTemplate
        const_cast<NamedDecl *>(Template), Info, AtomicExpr->getSourceRange());
    if (Inst.isInvalid())
      return ExprError();

    // We do not want error diagnostics escaping here.
    Sema::SFINAETrap Trap(S);
    SubstitutedExpression =
        S.SubstConstraintExpr(const_cast<Expr *>(AtomicExpr), MLTAL);

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
      // FIXME: This is an unfortunate consequence of there
      //  being no serialization code for PartialDiagnostics and the fact
      //  that serializing them would likely take a lot more storage than
      //  just storing them as strings. We would still like, in the
      //  future, to serialize the proper PartialDiagnostic as serializing
      //  it as a string defeats the purpose of the diagnostic mechanism.
      Satisfaction.Details.emplace_back(
          new (S.Context) ConstraintSubstitutionDiagnostic{
              SubstDiag.first,
              allocateStringFromConceptDiagnostic(S, SubstDiag.second)});
      Satisfaction.IsSatisfied = false;
      return ExprEmpty();
    }
  }

  if (!S.CheckConstraintExpression(SubstitutedExpression.get()))
    return ExprError();

  // [temp.constr.atomic]p3: To determine if an atomic constraint is
  // satisfied, the parameter mapping and template arguments are first
  // substituted into its expression.  If substitution results in an
  // invalid type or expression, the constraint is not satisfied.
  // Otherwise, the lvalue-to-rvalue conversion is performed if necessary,
  // and E shall be a constant expression of type bool.
  //
  // Perform the L to R Value conversion if necessary. We do so for all
  // non-PRValue categories, else we fail to extend the lifetime of
  // temporaries, and that fails the constant expression check.
  if (!SubstitutedExpression.get()->isPRValue())
    SubstitutedExpression = ImplicitCastExpr::Create(
        S.Context, SubstitutedExpression.get()->getType(), CK_LValueToRValue,
        SubstitutedExpression.get(),
        /*BasePath=*/nullptr, VK_PRValue, FPOptionsOverride());

  return SubstitutedExpression;
}

std::optional<MultiLevelTemplateArgumentList>
ConstraintSatisfactionChecker::SubstitutionInTemplateArguments(
    const NormalizedConstraintWithParamMapping &Constraint,
    MultiLevelTemplateArgumentList MLTAL,
    llvm::SmallVector<TemplateArgument> &SubstitutedOuterMost) {

  if (!Constraint.hasParameterMapping())
    return std::move(MLTAL);

  TemplateDeductionInfo Info(Constraint.getBeginLoc());
  Sema::InstantiatingTemplate Inst(
      S, Constraint.getBeginLoc(),
      Sema::InstantiatingTemplate::ConstraintSubstitution{},
      // FIXME: improve const-correctness of InstantiatingTemplate
      const_cast<NamedDecl *>(Template), Info, Constraint.getSourceRange());
  if (Inst.isInvalid())
    return std::nullopt;

  Sema::SFINAETrap Trap(S);

  TemplateArgumentListInfo SubstArgs;
  Sema::ArgPackSubstIndexRAII SubstIndex(
      S, Constraint.getPackSubstitutionIndex()
             ? Constraint.getPackSubstitutionIndex()
             : PackSubstitutionIndex);

  if (S.SubstTemplateArgumentsInParameterMapping(
          Constraint.getParameterMapping(), Constraint.getBeginLoc(), MLTAL,
          SubstArgs, /*BuildPackExpansionTypes=*/true)) {
    Satisfaction.IsSatisfied = false;
    return std::nullopt;
  }

  Sema::CheckTemplateArgumentInfo CTAI;
  auto *TD = const_cast<TemplateDecl *>(
      cast<TemplateDecl>(Constraint.getConstraintDecl()));
  if (S.CheckTemplateArgumentList(TD, Constraint.getUsedTemplateParamList(),
                                  TD->getLocation(), SubstArgs,
                                  /*DefaultArguments=*/{},
                                  /*PartialTemplateArgs=*/false, CTAI))
    return std::nullopt;
  const NormalizedConstraint::OccurenceList &Used =
      Constraint.mappingOccurenceList();
  SubstitutedOuterMost =
      llvm::to_vector_of<TemplateArgument>(MLTAL.getOutermost());
  unsigned Offset = 0;
  for (unsigned I = 0, MappedIndex = 0; I < Used.size(); I++) {
    TemplateArgument Arg;
    if (Used[I])
      Arg = S.Context.getCanonicalTemplateArgument(
          CTAI.SugaredConverted[MappedIndex++]);
    if (I < SubstitutedOuterMost.size()) {
      SubstitutedOuterMost[I] = Arg;
      Offset = I + 1;
    } else {
      SubstitutedOuterMost.push_back(Arg);
      Offset = SubstitutedOuterMost.size();
    }
  }
  if (Offset < SubstitutedOuterMost.size())
    SubstitutedOuterMost.erase(SubstitutedOuterMost.begin() + Offset);

  MLTAL.replaceOutermostTemplateArguments(
      const_cast<NamedDecl *>(Constraint.getConstraintDecl()),
      SubstitutedOuterMost);
  return std::move(MLTAL);
}

ExprResult ConstraintSatisfactionChecker::EvaluateSlow(
    const AtomicConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  llvm::SmallVector<TemplateArgument> SubstitutedOuterMost;
  std::optional<MultiLevelTemplateArgumentList> SubstitutedArgs =
      SubstitutionInTemplateArguments(Constraint, MLTAL, SubstitutedOuterMost);
  if (!SubstitutedArgs) {
    Satisfaction.IsSatisfied = false;
    return ExprEmpty();
  }

  Sema::ArgPackSubstIndexRAII SubstIndex(S, PackSubstitutionIndex);
  ExprResult SubstitutedAtomicExpr = EvaluateAtomicConstraint(
      Constraint.getConstraintExpr(), *SubstitutedArgs);

  if (SubstitutedAtomicExpr.isInvalid())
    return ExprError();

  if (SubstitutedAtomicExpr.isUnset())
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
    Satisfaction.Details.emplace_back(
        new (S.Context) ConstraintSubstitutionDiagnostic{
            SubstitutedAtomicExpr.get()->getBeginLoc(),
            allocateStringFromConceptDiagnostic(S, Msg)});
    return SubstitutedAtomicExpr;
  }

  if (SubstitutedAtomicExpr.get()->isValueDependent()) {
    Satisfaction.IsSatisfied = true;
    Satisfaction.ContainsErrors = false;
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
    Satisfaction.Details.emplace_back(SubstitutedAtomicExpr.get());

  return SubstitutedAtomicExpr;
}

ExprResult ConstraintSatisfactionChecker::Evaluate(
    const AtomicConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  unsigned Size = Satisfaction.Details.size();
  llvm::FoldingSetNodeID ID;
  UnsignedOrNone OuterPackSubstIndex =
      Constraint.getPackSubstitutionIndex()
          ? Constraint.getPackSubstitutionIndex()
          : PackSubstitutionIndex;

  ID.AddPointer(Constraint.getConstraintExpr());
  ID.AddInteger(OuterPackSubstIndex.toInternalRepresentation());
  HashParameterMapping(S, MLTAL, ID, OuterPackSubstIndex)
      .VisitConstraint(Constraint);

  if (auto Iter = S.UnsubstitutedConstraintSatisfactionCache.find(ID);
      Iter != S.UnsubstitutedConstraintSatisfactionCache.end()) {

    auto &Cached = Iter->second.Satisfaction;
    Satisfaction.ContainsErrors = Cached.ContainsErrors;
    Satisfaction.IsSatisfied = Cached.IsSatisfied;
    Satisfaction.Details.insert(Satisfaction.Details.begin() + Size,
                                Cached.Details.begin(), Cached.Details.end());
    return Iter->second.SubstExpr;
  }

  ExprResult E = EvaluateSlow(Constraint, MLTAL);

  UnsubstitutedConstraintSatisfactionCacheResult Cache;
  Cache.Satisfaction.ContainsErrors = Satisfaction.ContainsErrors;
  Cache.Satisfaction.IsSatisfied = Satisfaction.IsSatisfied;
  std::copy(Satisfaction.Details.begin() + Size, Satisfaction.Details.end(),
            std::back_inserter(Cache.Satisfaction.Details));
  Cache.SubstExpr = E;
  S.UnsubstitutedConstraintSatisfactionCache.insert({ID, std::move(Cache)});

  return E;
}

UnsignedOrNone
ConstraintSatisfactionChecker::EvaluateFoldExpandedConstraintSize(
    const FoldExpandedConstraint &FE,
    const MultiLevelTemplateArgumentList &MLTAL) {

  // We should ignore errors in the presence of packs of different size.
  Sema::SFINAETrap Trap(S);

  Expr *Pattern = const_cast<Expr *>(FE.getPattern());

  SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  S.collectUnexpandedParameterPacks(Pattern, Unexpanded);
  assert(!Unexpanded.empty() && "Pack expansion without parameter packs?");
  bool Expand = true;
  bool RetainExpansion = false;
  UnsignedOrNone NumExpansions(std::nullopt);
  if (S.CheckParameterPacksForExpansion(
          Pattern->getExprLoc(), Pattern->getSourceRange(), Unexpanded, MLTAL,
          /*FailOnPackProducingTemplates=*/false, Expand, RetainExpansion,
          NumExpansions) ||
      !Expand || RetainExpansion)
    return std::nullopt;

  if (NumExpansions && S.getLangOpts().BracketDepth < *NumExpansions) {
    S.Diag(Pattern->getExprLoc(),
           clang::diag::err_fold_expression_limit_exceeded)
        << *NumExpansions << S.getLangOpts().BracketDepth
        << Pattern->getSourceRange();
    S.Diag(Pattern->getExprLoc(), diag::note_bracket_depth);
    return std::nullopt;
  }
  return NumExpansions;
}

ExprResult ConstraintSatisfactionChecker::EvaluateSlow(
    const FoldExpandedConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  bool Conjunction = Constraint.getFoldOperator() ==
                     FoldExpandedConstraint::FoldOperatorKind::And;
  unsigned EffectiveDetailEndIndex = Satisfaction.Details.size();

  llvm::SmallVector<TemplateArgument> SubstitutedOuterMost;
  // FIXME: Is PackSubstitutionIndex correct?
  llvm::SaveAndRestore _(PackSubstitutionIndex, S.ArgPackSubstIndex);
  std::optional<MultiLevelTemplateArgumentList> SubstitutedArgs =
      SubstitutionInTemplateArguments(
          static_cast<const NormalizedConstraintWithParamMapping &>(Constraint),
          MLTAL, SubstitutedOuterMost);
  if (!SubstitutedArgs) {
    Satisfaction.IsSatisfied = false;
    return ExprError();
  }

  ExprResult Out;
  UnsignedOrNone NumExpansions =
      EvaluateFoldExpandedConstraintSize(Constraint, *SubstitutedArgs);
  if (!NumExpansions)
    return ExprEmpty();

  if (*NumExpansions == 0) {
    Satisfaction.IsSatisfied = Conjunction;
    return ExprEmpty();
  }

  for (unsigned I = 0; I < *NumExpansions; I++) {
    Sema::ArgPackSubstIndexRAII SubstIndex(S, I);
    Satisfaction.IsSatisfied = false;
    Satisfaction.ContainsErrors = false;
    ExprResult Expr =
        ConstraintSatisfactionChecker(S, Template, TemplateNameLoc,
                                      UnsignedOrNone(I), Satisfaction)
            .Evaluate(Constraint.getNormalizedPattern(), *SubstitutedArgs);
    if (Expr.isUsable()) {
      if (Out.isUnset())
        Out = Expr;
      else
        Out = BinaryOperator::Create(S.Context, Out.get(), Expr.get(),
                                     Conjunction ? BinaryOperatorKind::BO_LAnd
                                                 : BinaryOperatorKind::BO_LOr,
                                     S.Context.BoolTy, VK_PRValue, OK_Ordinary,
                                     Constraint.getBeginLoc(),
                                     FPOptionsOverride{});
    } else {
      assert(!Satisfaction.IsSatisfied);
    }
    if (!Conjunction && Satisfaction.IsSatisfied) {
      Satisfaction.Details.erase(Satisfaction.Details.begin() +
                                     EffectiveDetailEndIndex,
                                 Satisfaction.Details.end());
      break;
    }
    if (Satisfaction.IsSatisfied != Conjunction)
      return Out;
  }

  return Out;
}

ExprResult ConstraintSatisfactionChecker::Evaluate(
    const FoldExpandedConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  llvm::FoldingSetNodeID ID;
  ID.AddPointer(Constraint.getPattern());
  HashParameterMapping(S, MLTAL, ID, std::nullopt).VisitConstraint(Constraint);

  if (auto Iter = S.UnsubstitutedConstraintSatisfactionCache.find(ID);
      Iter != S.UnsubstitutedConstraintSatisfactionCache.end()) {

    auto &Cached = Iter->second.Satisfaction;
    Satisfaction.ContainsErrors = Cached.ContainsErrors;
    Satisfaction.IsSatisfied = Cached.IsSatisfied;
    Satisfaction.Details.insert(Satisfaction.Details.end(),
                                Cached.Details.begin(), Cached.Details.end());
    return Iter->second.SubstExpr;
  }

  unsigned Size = Satisfaction.Details.size();

  ExprResult E = EvaluateSlow(Constraint, MLTAL);
  UnsubstitutedConstraintSatisfactionCacheResult Cache;
  Cache.Satisfaction.ContainsErrors = Satisfaction.ContainsErrors;
  Cache.Satisfaction.IsSatisfied = Satisfaction.IsSatisfied;
  std::copy(Satisfaction.Details.begin() + Size, Satisfaction.Details.end(),
            std::back_inserter(Cache.Satisfaction.Details));
  Cache.SubstExpr = E;
  S.UnsubstitutedConstraintSatisfactionCache.insert({ID, std::move(Cache)});
  return E;
}

ExprResult ConstraintSatisfactionChecker::EvaluateSlow(
    const ConceptIdConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL, unsigned Size) {
  const ConceptReference *ConceptId = Constraint.getConceptId();

  llvm::SmallVector<TemplateArgument> SubstitutedOuterMost;
  std::optional<MultiLevelTemplateArgumentList> SubstitutedArgs =
      SubstitutionInTemplateArguments(Constraint, MLTAL, SubstitutedOuterMost);

  if (!SubstitutedArgs) {
    Satisfaction.IsSatisfied = false;
    // FIXME: diagnostics?
    return ExprError();
  }

  Sema::SFINAETrap Trap(S);
  Sema::ArgPackSubstIndexRAII SubstIndex(
      S, Constraint.getPackSubstitutionIndex()
             ? Constraint.getPackSubstitutionIndex()
             : PackSubstitutionIndex);

  const ASTTemplateArgumentListInfo *Ori =
      ConceptId->getTemplateArgsAsWritten();
  TemplateDeductionInfo Info(TemplateNameLoc);
  Sema::InstantiatingTemplate _(
      S, TemplateNameLoc, Sema::InstantiatingTemplate::ConstraintSubstitution{},
      const_cast<NamedDecl *>(Template), Info, Constraint.getSourceRange());

  TemplateArgumentListInfo OutArgs(Ori->LAngleLoc, Ori->RAngleLoc);
  if (S.SubstTemplateArguments(Ori->arguments(), *SubstitutedArgs, OutArgs) ||
      Trap.hasErrorOccurred()) {
    Satisfaction.IsSatisfied = false;
    if (!Trap.hasErrorOccurred())
      return ExprError();

    PartialDiagnosticAt SubstDiag{SourceLocation(),
                                  PartialDiagnostic::NullDiagnostic()};
    Info.takeSFINAEDiagnostic(SubstDiag);
    // FIXME: This is an unfortunate consequence of there
    //  being no serialization code for PartialDiagnostics and the fact
    //  that serializing them would likely take a lot more storage than
    //  just storing them as strings. We would still like, in the
    //  future, to serialize the proper PartialDiagnostic as serializing
    //  it as a string defeats the purpose of the diagnostic mechanism.
    Satisfaction.Details.insert(
        Satisfaction.Details.begin() + Size,
        new (S.Context) ConstraintSubstitutionDiagnostic{
            SubstDiag.first,
            allocateStringFromConceptDiagnostic(S, SubstDiag.second)});
    return ExprError();
  }

  CXXScopeSpec SS;
  SS.Adopt(ConceptId->getNestedNameSpecifierLoc());

  ExprResult SubstitutedConceptId = S.CheckConceptTemplateId(
      SS, ConceptId->getTemplateKWLoc(), ConceptId->getConceptNameInfo(),
      ConceptId->getFoundDecl(), ConceptId->getNamedConcept(), &OutArgs,
      /*DoCheckConstraintSatisfaction=*/false);

  if (SubstitutedConceptId.isInvalid() || Trap.hasErrorOccurred())
    return ExprError();

  if (Size != Satisfaction.Details.size()) {
    Satisfaction.Details.insert(
        Satisfaction.Details.begin() + Size,
        UnsatisfiedConstraintRecord(
            SubstitutedConceptId.getAs<ConceptSpecializationExpr>()
                ->getConceptReference()));
  }
  return SubstitutedConceptId;
}

ExprResult ConstraintSatisfactionChecker::Evaluate(
    const ConceptIdConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  const ConceptReference *ConceptId = Constraint.getConceptId();

  UnsignedOrNone OuterPackSubstIndex =
      Constraint.getPackSubstitutionIndex()
          ? Constraint.getPackSubstitutionIndex()
          : PackSubstitutionIndex;

  Sema::InstantiatingTemplate _(S, ConceptId->getBeginLoc(),
                                Sema::InstantiatingTemplate::ConstraintsCheck{},
                                ConceptId->getNamedConcept(),
                                MLTAL.getInnermost(),
                                Constraint.getSourceRange());

  unsigned Size = Satisfaction.Details.size();

  ExprResult E = Evaluate(Constraint.getNormalizedConstraint(), MLTAL);

  if (!E.isUsable()) {
    Satisfaction.Details.insert(Satisfaction.Details.begin() + Size, ConceptId);
    return E;
  }

  // ConceptIdConstraint is only relevant for diagnostics,
  // so if the normalized constraint is satisfied, we should not
  // substitute into the constraint.
  if (Satisfaction.IsSatisfied)
    return E;

  llvm::FoldingSetNodeID ID;
  ID.AddPointer(Constraint.getConceptId());
  ID.AddInteger(OuterPackSubstIndex.toInternalRepresentation());
  HashParameterMapping(S, MLTAL, ID, OuterPackSubstIndex)
      .VisitConstraint(Constraint);

  if (auto Iter = S.UnsubstitutedConstraintSatisfactionCache.find(ID);
      Iter != S.UnsubstitutedConstraintSatisfactionCache.end()) {

    auto &Cached = Iter->second.Satisfaction;
    Satisfaction.ContainsErrors = Cached.ContainsErrors;
    Satisfaction.IsSatisfied = Cached.IsSatisfied;
    Satisfaction.Details.insert(Satisfaction.Details.begin() + Size,
                                Cached.Details.begin(), Cached.Details.end());
    return Iter->second.SubstExpr;
  }

  ExprResult CE = EvaluateSlow(Constraint, MLTAL, Size);
  if (CE.isInvalid())
    return E;
  UnsubstitutedConstraintSatisfactionCacheResult Cache;
  Cache.Satisfaction.ContainsErrors = Satisfaction.ContainsErrors;
  Cache.Satisfaction.IsSatisfied = Satisfaction.IsSatisfied;
  std::copy(Satisfaction.Details.begin() + Size, Satisfaction.Details.end(),
            std::back_inserter(Cache.Satisfaction.Details));
  Cache.SubstExpr = CE;
  S.UnsubstitutedConstraintSatisfactionCache.insert({ID, std::move(Cache)});
  return CE;
}

ExprResult ConstraintSatisfactionChecker::Evaluate(
    const CompoundConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {

  unsigned EffectiveDetailEndIndex = Satisfaction.Details.size();

  bool Conjunction =
      Constraint.getCompoundKind() == NormalizedConstraint::CCK_Conjunction;

  ExprResult LHS = Evaluate(Constraint.getLHS(), MLTAL);

  if (Conjunction && (!Satisfaction.IsSatisfied || Satisfaction.ContainsErrors))
    return LHS;

  if (!Conjunction && LHS.isUsable() && Satisfaction.IsSatisfied &&
      !Satisfaction.ContainsErrors)
    return LHS;

  Satisfaction.ContainsErrors = false;
  Satisfaction.IsSatisfied = false;

  ExprResult RHS = Evaluate(Constraint.getRHS(), MLTAL);

  if (RHS.isUsable() && Satisfaction.IsSatisfied &&
      !Satisfaction.ContainsErrors)
    Satisfaction.Details.erase(Satisfaction.Details.begin() +
                                   EffectiveDetailEndIndex,
                               Satisfaction.Details.end());

  if (!LHS.isUsable())
    return RHS;

  if (!RHS.isUsable())
    return LHS;

  return BinaryOperator::Create(S.Context, LHS.get(), RHS.get(),
                                Conjunction ? BinaryOperatorKind::BO_LAnd
                                            : BinaryOperatorKind::BO_LOr,
                                S.Context.BoolTy, VK_PRValue, OK_Ordinary,
                                Constraint.getBeginLoc(), FPOptionsOverride{});
}

ExprResult ConstraintSatisfactionChecker::Evaluate(
    const NormalizedConstraint &Constraint,
    const MultiLevelTemplateArgumentList &MLTAL) {
  switch (Constraint.getKind()) {
  case NormalizedConstraint::ConstraintKind::Atomic:
    return Evaluate(static_cast<const AtomicConstraint &>(Constraint), MLTAL);

  case NormalizedConstraint::ConstraintKind::FoldExpanded:
    return Evaluate(static_cast<const FoldExpandedConstraint &>(Constraint),
                    MLTAL);

  case NormalizedConstraint::ConstraintKind::ConceptId:
    return Evaluate(static_cast<const ConceptIdConstraint &>(Constraint),
                    MLTAL);

  case NormalizedConstraint::ConstraintKind::Compound:
    return Evaluate(static_cast<const CompoundConstraint &>(Constraint), MLTAL);
  }
  llvm_unreachable("Unknown ConstraintKind enum");
}

static bool CheckConstraintSatisfaction(
    Sema &S, const NamedDecl *Template,
    ArrayRef<AssociatedConstraint> AssociatedConstraints,
    const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange, ConstraintSatisfaction &Satisfaction,
    Expr **ConvertedExpr, const ConceptReference *TopLevelConceptId = nullptr) {

  if (ConvertedExpr)
    *ConvertedExpr = nullptr;

  if (AssociatedConstraints.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  if (TemplateArgsLists.isAnyArgInstantiationDependent()) {
    // No need to check satisfaction for dependent constraint expressions.
    Satisfaction.IsSatisfied = true;
    return false;
  }

  llvm::ArrayRef<TemplateArgument> Args;
  if (TemplateArgsLists.getNumLevels() != 0)
    Args = TemplateArgsLists.getInnermost();

  std::optional<Sema::InstantiatingTemplate> SynthesisContext;
  if (!TopLevelConceptId) {
    SynthesisContext.emplace(S, TemplateIDRange.getBegin(),
                             Sema::InstantiatingTemplate::ConstraintsCheck{},
                             const_cast<NamedDecl *>(Template), Args,
                             TemplateIDRange);
  }

  const NormalizedConstraint *C =
      S.getNormalizedAssociatedConstraints(Template, AssociatedConstraints);
  if (!C) {
    Satisfaction.IsSatisfied = false;
    return true;
  }

  if (TopLevelConceptId)
    C = ConceptIdConstraint::Create(S.getASTContext(), TopLevelConceptId,
                                    const_cast<NormalizedConstraint *>(C),
                                    Template, /*CSE=*/nullptr,
                                    S.ArgPackSubstIndex);

  ExprResult Res =
      ConstraintSatisfactionChecker(S, Template, TemplateIDRange.getBegin(),
                                    S.ArgPackSubstIndex, Satisfaction)
          .Evaluate(*C, TemplateArgsLists);

  if (Res.isInvalid())
    return true;

  if (Res.isUsable() && ConvertedExpr)
    *ConvertedExpr = Res.get();

  return false;
}

bool Sema::CheckConstraintSatisfaction(
    ConstrainedDeclOrNestedRequirement Entity,
    ArrayRef<AssociatedConstraint> AssociatedConstraints,
    const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange, ConstraintSatisfaction &OutSatisfaction,
    const ConceptReference *TopLevelConceptId, Expr **ConvertedExpr) {
  if (AssociatedConstraints.empty()) {
    OutSatisfaction.IsSatisfied = true;
    return false;
  }
  const auto *Template = Entity.dyn_cast<const NamedDecl *>();
  if (!Template) {
    return ::CheckConstraintSatisfaction(
        *this, nullptr, AssociatedConstraints, TemplateArgsLists,
        TemplateIDRange, OutSatisfaction, ConvertedExpr, TopLevelConceptId);
  }
  // Invalid templates could make their way here. Substituting them could result
  // in dependent expressions.
  if (Template->isInvalidDecl()) {
    OutSatisfaction.IsSatisfied = false;
    return true;
  }

  // A list of the template argument list flattened in a predictible manner for
  // the purposes of caching. The ConstraintSatisfaction type is in AST so it
  // has no access to the MultiLevelTemplateArgumentList, so this has to happen
  // here.
  llvm::SmallVector<TemplateArgument, 4> FlattenedArgs;
  for (auto List : TemplateArgsLists)
    for (const TemplateArgument &Arg : List.Args)
      FlattenedArgs.emplace_back(Context.getCanonicalTemplateArgument(Arg));

  const NamedDecl *Owner = Template;
  if (TopLevelConceptId)
    Owner = TopLevelConceptId->getNamedConcept();

  llvm::FoldingSetNodeID ID;
  ConstraintSatisfaction::Profile(ID, Context, Owner, FlattenedArgs);
  void *InsertPos;
  if (auto *Cached = SatisfactionCache.FindNodeOrInsertPos(ID, InsertPos)) {
    OutSatisfaction = *Cached;
    return false;
  }

  auto Satisfaction =
      std::make_unique<ConstraintSatisfaction>(Owner, FlattenedArgs);
  if (::CheckConstraintSatisfaction(
          *this, Template, AssociatedConstraints, TemplateArgsLists,
          TemplateIDRange, *Satisfaction, ConvertedExpr, TopLevelConceptId)) {
    OutSatisfaction = std::move(*Satisfaction);
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

bool Sema::CheckConstraintSatisfaction(
    const ConceptSpecializationExpr *ConstraintExpr,
    ConstraintSatisfaction &Satisfaction) {

  llvm::SmallVector<AssociatedConstraint, 1> Constraints;
  Constraints.emplace_back(
      ConstraintExpr->getNamedConcept()->getConstraintExpr());

  MultiLevelTemplateArgumentList MLTAL(ConstraintExpr->getNamedConcept(),
                                       ConstraintExpr->getTemplateArguments(),
                                       true);

  return CheckConstraintSatisfaction(
      ConstraintExpr->getNamedConcept(), Constraints, MLTAL,
      ConstraintExpr->getSourceRange(), Satisfaction,
      ConstraintExpr->getConceptReference());
}

bool Sema::SetupConstraintScope(
    FunctionDecl *FD, std::optional<ArrayRef<TemplateArgument>> TemplateArgs,
    const MultiLevelTemplateArgumentList &MLTAL,
    LocalInstantiationScope &Scope) {
  assert(!isLambdaCallOperator(FD) &&
         "Use LambdaScopeForCallOperatorInstantiationRAII to handle lambda "
         "instantiations");
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
    if (FunctionTemplateDecl *FromMemTempl =
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
std::optional<MultiLevelTemplateArgumentList>
Sema::SetupConstraintCheckingTemplateArgumentsAndScope(
    FunctionDecl *FD, std::optional<ArrayRef<TemplateArgument>> TemplateArgs,
    LocalInstantiationScope &Scope) {
  MultiLevelTemplateArgumentList MLTAL;

  // Collect the list of template arguments relative to the 'primary' template.
  // We need the entire list, since the constraint is completely uninstantiated
  // at this point.
  MLTAL =
      getTemplateInstantiationArgs(FD, FD->getLexicalDeclContext(),
                                   /*Final=*/false, /*Innermost=*/std::nullopt,
                                   /*RelativeToPrimary=*/true,
                                   /*Pattern=*/nullptr,
                                   /*ForConstraintInstantiation=*/true);
  // Lambdas are handled by LambdaScopeForCallOperatorInstantiationRAII.
  if (isLambdaCallOperator(FD))
    return MLTAL;
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
  // CheckFunctionTemplateConstraints after this will check it
  // better.
  if (FD->isDependentContext() ||
      FD->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  // A lambda conversion operator has the same constraints as the call operator
  // and constraints checking relies on whether we are in a lambda call operator
  // (and may refer to its parameters), so check the call operator instead.
  // Note that the declarations outside of the lambda should also be
  // considered. Turning on the 'ForOverloadResolution' flag results in the
  // LocalInstantiationScope not looking into its parents, but we can still
  // access Decls from the parents while building a lambda RAII scope later.
  if (const auto *MD = dyn_cast<CXXConversionDecl>(FD);
      MD && isLambdaConversionOperator(const_cast<CXXConversionDecl *>(MD)))
    return CheckFunctionConstraints(MD->getParent()->getLambdaCallOperator(),
                                    Satisfaction, UsageLoc,
                                    /*ShouldAddDeclsFromParentScope=*/true);

  DeclContext *CtxToSave = const_cast<FunctionDecl *>(FD);

  while (isLambdaCallOperator(CtxToSave) || FD->isTransparentContext()) {
    if (isLambdaCallOperator(CtxToSave))
      CtxToSave = CtxToSave->getParent()->getParent();
    else
      CtxToSave = CtxToSave->getNonTransparentContext();
  }

  ContextRAII SavedContext{*this, CtxToSave};
  LocalInstantiationScope Scope(*this, !ForOverloadResolution);
  std::optional<MultiLevelTemplateArgumentList> MLTAL =
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

  LambdaScopeForCallOperatorInstantiationRAII LambdaScope(
      *this, const_cast<FunctionDecl *>(FD), *MLTAL, Scope,
      ForOverloadResolution);

  return CheckConstraintSatisfaction(
      FD, FD->getTrailingRequiresClause(), *MLTAL,
      SourceRange(UsageLoc.isValid() ? UsageLoc : FD->getLocation()),
      Satisfaction);
}

static const Expr *SubstituteConstraintExpressionWithoutSatisfaction(
    Sema &S, const Sema::TemplateCompareNewDeclInfo &DeclInfo,
    const Expr *ConstrExpr) {
  MultiLevelTemplateArgumentList MLTAL = S.getTemplateInstantiationArgs(
      DeclInfo.getDecl(), DeclInfo.getDeclContext(), /*Final=*/false,
      /*Innermost=*/std::nullopt,
      /*RelativeToPrimary=*/true,
      /*Pattern=*/nullptr, /*ForConstraintInstantiation=*/true,
      /*SkipForSpecialization*/ false);

  if (MLTAL.getNumSubstitutedLevels() == 0)
    return ConstrExpr;

  Sema::SFINAETrap SFINAE(S);

  Sema::InstantiatingTemplate Inst(
      S, DeclInfo.getLocation(),
      Sema::InstantiatingTemplate::ConstraintNormalization{},
      const_cast<NamedDecl *>(DeclInfo.getDecl()), SourceRange{});
  if (Inst.isInvalid())
    return nullptr;

  // Set up a dummy 'instantiation' scope in the case of reference to function
  // parameters that the surrounding function hasn't been instantiated yet. Note
  // this may happen while we're comparing two templates' constraint
  // equivalence.
  std::optional<LocalInstantiationScope> ScopeForParameters;
  if (const NamedDecl *ND = DeclInfo.getDecl();
      ND && ND->isFunctionOrFunctionTemplate()) {
    ScopeForParameters.emplace(S, /*CombineWithOuterScope=*/true);
    const FunctionDecl *FD = ND->getAsFunction();
    if (FunctionTemplateDecl *Template = FD->getDescribedFunctionTemplate();
        Template && Template->getInstantiatedFromMemberTemplate())
      FD = Template->getInstantiatedFromMemberTemplate()->getTemplatedDecl();
    for (auto *PVD : FD->parameters()) {
      if (ScopeForParameters->getInstantiationOfIfExists(PVD))
        continue;
      if (!PVD->isParameterPack()) {
        ScopeForParameters->InstantiatedLocal(PVD, PVD);
        continue;
      }
      // This is hacky: we're mapping the parameter pack to a size-of-1 argument
      // to avoid building SubstTemplateTypeParmPackTypes for
      // PackExpansionTypes. The SubstTemplateTypeParmPackType node would
      // otherwise reference the AssociatedDecl of the template arguments, which
      // is, in this case, the template declaration.
      //
      // However, as we are in the process of comparing potential
      // re-declarations, the canonical declaration is the declaration itself at
      // this point. So if we didn't expand these packs, we would end up with an
      // incorrect profile difference because we will be profiling the
      // canonical types!
      //
      // FIXME: Improve the "no-transform" machinery in FindInstantiatedDecl so
      // that we can eliminate the Scope in the cases where the declarations are
      // not necessarily instantiated. It would also benefit the noexcept
      // specifier comparison.
      ScopeForParameters->MakeInstantiatedLocalArgPack(PVD);
      ScopeForParameters->InstantiatedLocalPackArg(PVD, PVD);
    }
  }

  std::optional<Sema::CXXThisScopeRAII> ThisScope;

  // See TreeTransform::RebuildTemplateSpecializationType. A context scope is
  // essential for having an injected class as the canonical type for a template
  // specialization type at the rebuilding stage. This guarantees that, for
  // out-of-line definitions, injected class name types and their equivalent
  // template specializations can be profiled to the same value, which makes it
  // possible that e.g. constraints involving C<Class<T>> and C<Class> are
  // perceived identical.
  std::optional<Sema::ContextRAII> ContextScope;
  const DeclContext *DC = [&] {
    if (!DeclInfo.getDecl())
      return DeclInfo.getDeclContext();
    return DeclInfo.getDecl()->getFriendObjectKind()
               ? DeclInfo.getLexicalDeclContext()
               : DeclInfo.getDeclContext();
  }();
  if (auto *RD = dyn_cast<CXXRecordDecl>(DC)) {
    ThisScope.emplace(S, const_cast<CXXRecordDecl *>(RD), Qualifiers());
    ContextScope.emplace(S, const_cast<DeclContext *>(cast<DeclContext>(RD)),
                         /*NewThisContext=*/false);
  }
  EnterExpressionEvaluationContext UnevaluatedContext(
      S, Sema::ExpressionEvaluationContext::Unevaluated,
      Sema::ReuseLambdaContextDecl);
  ExprResult SubstConstr = S.SubstConstraintExprWithoutSatisfaction(
      const_cast<clang::Expr *>(ConstrExpr), MLTAL);
  if (SFINAE.hasErrorOccurred() || !SubstConstr.isUsable())
    return nullptr;
  return SubstConstr.get();
}

bool Sema::AreConstraintExpressionsEqual(const NamedDecl *Old,
                                         const Expr *OldConstr,
                                         const TemplateCompareNewDeclInfo &New,
                                         const Expr *NewConstr) {
  if (OldConstr == NewConstr)
    return true;
  // C++ [temp.constr.decl]p4
  if (Old && !New.isInvalid() && !New.ContainsDecl(Old) &&
      Old->getLexicalDeclContext() != New.getLexicalDeclContext()) {
    if (const Expr *SubstConstr =
            SubstituteConstraintExpressionWithoutSatisfaction(*this, Old,
                                                              OldConstr))
      OldConstr = SubstConstr;
    else
      return false;
    if (const Expr *SubstConstr =
            SubstituteConstraintExpressionWithoutSatisfaction(*this, New,
                                                              NewConstr))
      NewConstr = SubstConstr;
    else
      return false;
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

  SmallVector<AssociatedConstraint, 3> ACs;
  FD->getDescribedFunctionTemplate()->getAssociatedConstraints(ACs);

  unsigned OldTemplateDepth = CalculateTemplateDepthForConstraints(*this, FD);
  for (const AssociatedConstraint &AC : ACs)
    if (ConstraintExpressionDependsOnEnclosingTemplate(FD, OldTemplateDepth,
                                                       AC.ConstraintExpr))
      return true;

  return false;
}

bool Sema::EnsureTemplateArgumentListConstraints(
    TemplateDecl *TD, const MultiLevelTemplateArgumentList &TemplateArgsLists,
    SourceRange TemplateIDRange) {
  ConstraintSatisfaction Satisfaction;
  llvm::SmallVector<AssociatedConstraint, 3> AssociatedConstraints;
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

static bool CheckFunctionConstraintsWithoutInstantiation(
    Sema &SemaRef, SourceLocation PointOfInstantiation,
    FunctionTemplateDecl *Template, ArrayRef<TemplateArgument> TemplateArgs,
    ConstraintSatisfaction &Satisfaction) {
  SmallVector<AssociatedConstraint, 3> TemplateAC;
  Template->getAssociatedConstraints(TemplateAC);
  if (TemplateAC.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  LocalInstantiationScope Scope(SemaRef);

  FunctionDecl *FD = Template->getTemplatedDecl();
  // Collect the list of template arguments relative to the 'primary'
  // template. We need the entire list, since the constraint is completely
  // uninstantiated at this point.

  MultiLevelTemplateArgumentList MLTAL;
  {
    // getTemplateInstantiationArgs uses this instantiation context to find out
    // template arguments for uninstantiated functions.
    // We don't want this RAII object to persist, because there would be
    // otherwise duplicate diagnostic notes.
    Sema::InstantiatingTemplate Inst(
        SemaRef, PointOfInstantiation,
        Sema::InstantiatingTemplate::ConstraintsCheck{}, Template, TemplateArgs,
        PointOfInstantiation);
    if (Inst.isInvalid())
      return true;
    MLTAL = SemaRef.getTemplateInstantiationArgs(
        /*D=*/FD, FD,
        /*Final=*/false, /*Innermost=*/{}, /*RelativeToPrimary=*/true,
        /*Pattern=*/nullptr, /*ForConstraintInstantiation=*/true);
  }

  Sema::ContextRAII SavedContext(SemaRef, FD);
  return SemaRef.CheckConstraintSatisfaction(
      Template, TemplateAC, MLTAL, PointOfInstantiation, Satisfaction);
}

bool Sema::CheckFunctionTemplateConstraints(
    SourceLocation PointOfInstantiation, FunctionDecl *Decl,
    ArrayRef<TemplateArgument> TemplateArgs,
    ConstraintSatisfaction &Satisfaction) {
  // In most cases we're not going to have constraints, so check for that first.
  FunctionTemplateDecl *Template = Decl->getPrimaryTemplate();

  if (!Template)
    return ::CheckFunctionConstraintsWithoutInstantiation(
        *this, PointOfInstantiation, Decl->getDescribedFunctionTemplate(),
        TemplateArgs, Satisfaction);

  // Note - code synthesis context for the constraints check is created
  // inside CheckConstraintsSatisfaction.
  SmallVector<AssociatedConstraint, 3> TemplateAC;
  Template->getAssociatedConstraints(TemplateAC);
  if (TemplateAC.empty()) {
    Satisfaction.IsSatisfied = true;
    return false;
  }

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  Sema::ContextRAII savedContext(*this, Decl);
  LocalInstantiationScope Scope(*this);

  std::optional<MultiLevelTemplateArgumentList> MLTAL =
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
  LambdaScopeForCallOperatorInstantiationRAII LambdaScope(*this, Decl, *MLTAL,
                                                          Scope);

  return CheckConstraintSatisfaction(Template, TemplateAC, *MLTAL,
                                     PointOfInstantiation, Satisfaction);
}

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::ExprRequirement *Req,
                                           bool First) {
  assert(!Req->isSatisfied() &&
         "Diagnose() can only be used on an unsatisfied requirement");
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
    S.Diag(Req->getNoexceptLoc(), diag::note_expr_requirement_noexcept_not_met)
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
      S.Diag(
          SubstDiag->DiagLoc,
          diag::
              note_expr_requirement_type_requirement_unknown_substitution_error)
          << (int)First << SubstDiag->SubstitutedEntity;
    break;
  }
  case concepts::ExprRequirement::SS_ConstraintsNotSatisfied: {
    ConceptSpecializationExpr *ConstraintExpr =
        Req->getReturnTypeRequirementSubstitutedConstraintExpr();
    S.DiagnoseUnsatisfiedConstraint(ConstraintExpr);
    break;
  }
  case concepts::ExprRequirement::SS_Satisfied:
    llvm_unreachable("We checked this above");
  }
}

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::TypeRequirement *Req,
                                           bool First) {
  assert(!Req->isSatisfied() &&
         "Diagnose() can only be used on an unsatisfied requirement");
  switch (Req->getSatisfactionStatus()) {
  case concepts::TypeRequirement::SS_Dependent:
    llvm_unreachable("Diagnosing a dependent requirement");
    return;
  case concepts::TypeRequirement::SS_SubstitutionFailure: {
    auto *SubstDiag = Req->getSubstitutionDiagnostic();
    if (!SubstDiag->DiagMessage.empty())
      S.Diag(SubstDiag->DiagLoc, diag::note_type_requirement_substitution_error)
          << (int)First << SubstDiag->SubstitutedEntity
          << SubstDiag->DiagMessage;
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

static void diagnoseUnsatisfiedConceptIdExpr(Sema &S,
                                             const ConceptReference *Concept,
                                             SourceLocation Loc, bool First) {
  if (Concept->getTemplateArgsAsWritten()->NumTemplateArgs == 1) {
    S.Diag(
        Loc,
        diag::
            note_single_arg_concept_specialization_constraint_evaluated_to_false)
        << (int)First
        << Concept->getTemplateArgsAsWritten()->arguments()[0].getArgument()
        << Concept->getNamedConcept();
  } else {
    S.Diag(Loc, diag::note_concept_specialization_constraint_evaluated_to_false)
        << (int)First << Concept;
  }
}

static void diagnoseUnsatisfiedConstraintExpr(
    Sema &S, const UnsatisfiedConstraintRecord &Record, SourceLocation Loc,
    bool First, concepts::NestedRequirement *Req = nullptr);

static void DiagnoseUnsatisfiedConstraint(
    Sema &S, ArrayRef<UnsatisfiedConstraintRecord> Records, SourceLocation Loc,
    bool First = true, concepts::NestedRequirement *Req = nullptr) {
  for (auto &Record : Records) {
    diagnoseUnsatisfiedConstraintExpr(S, Record, Loc, First, Req);
    Loc = {};
    First = isa<const ConceptReference *>(Record);
  }
}

static void diagnoseUnsatisfiedRequirement(Sema &S,
                                           concepts::NestedRequirement *Req,
                                           bool First) {
  DiagnoseUnsatisfiedConstraint(S, Req->getConstraintSatisfaction().records(),
                                Req->hasInvalidConstraint()
                                    ? SourceLocation()
                                    : Req->getConstraintExpr()->getExprLoc(),
                                First, Req);
}

static void diagnoseWellFormedUnsatisfiedConstraintExpr(Sema &S,
                                                        const Expr *SubstExpr,
                                                        bool First) {
  SubstExpr = SubstExpr->IgnoreParenImpCasts();
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(SubstExpr)) {
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
        if (!SimplifiedLHS.Diag && !SimplifiedRHS.Diag) {
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
  } else if (auto *CSE = dyn_cast<ConceptSpecializationExpr>(SubstExpr)) {
    // Drill down concept ids treated as atomic constraints
    S.DiagnoseUnsatisfiedConstraint(CSE, First);
    return;
  } else if (auto *TTE = dyn_cast<TypeTraitExpr>(SubstExpr);
             TTE && TTE->getTrait() == clang::TypeTrait::BTT_IsDeducible) {
    assert(TTE->getNumArgs() == 2);
    S.Diag(SubstExpr->getSourceRange().getBegin(),
           diag::note_is_deducible_constraint_evaluated_to_false)
        << TTE->getArg(0)->getType() << TTE->getArg(1)->getType();
    return;
  }

  S.Diag(SubstExpr->getSourceRange().getBegin(),
         diag::note_atomic_constraint_evaluated_to_false)
      << (int)First << SubstExpr;
  S.DiagnoseTypeTraitDetails(SubstExpr);
}

static void diagnoseUnsatisfiedConstraintExpr(
    Sema &S, const UnsatisfiedConstraintRecord &Record, SourceLocation Loc,
    bool First, concepts::NestedRequirement *Req) {
  if (auto *Diag =
          Record
              .template dyn_cast<const ConstraintSubstitutionDiagnostic *>()) {
    if (Req)
      S.Diag(Diag->first, diag::note_nested_requirement_substitution_error)
          << (int)First << Req->getInvalidConstraintEntity() << Diag->second;
    else
      S.Diag(Diag->first, diag::note_substituted_constraint_expr_is_ill_formed)
          << Diag->second;
    return;
  }
  if (const auto *Concept = dyn_cast<const ConceptReference *>(Record)) {
    if (Loc.isInvalid())
      Loc = Concept->getBeginLoc();
    diagnoseUnsatisfiedConceptIdExpr(S, Concept, Loc, First);
    return;
  }
  diagnoseWellFormedUnsatisfiedConstraintExpr(
      S, cast<const class Expr *>(Record), First);
}

void Sema::DiagnoseUnsatisfiedConstraint(
    const ConstraintSatisfaction &Satisfaction, SourceLocation Loc,
    bool First) {

  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");
  ::DiagnoseUnsatisfiedConstraint(*this, Satisfaction.Details, Loc, First);
}

void Sema::DiagnoseUnsatisfiedConstraint(
    const ConceptSpecializationExpr *ConstraintExpr, bool First) {

  const ASTConstraintSatisfaction &Satisfaction =
      ConstraintExpr->getSatisfaction();

  assert(!Satisfaction.IsSatisfied &&
         "Attempted to diagnose a satisfied constraint");

  ::DiagnoseUnsatisfiedConstraint(*this, Satisfaction.records(),
                                  ConstraintExpr->getBeginLoc(), First);
}

namespace {

class SubstituteParameterMappings {
  Sema &SemaRef;

  const MultiLevelTemplateArgumentList *MLTAL;
  const ASTTemplateArgumentListInfo *ArgsAsWritten;

  bool InFoldExpr;

  SubstituteParameterMappings(Sema &SemaRef,
                              const MultiLevelTemplateArgumentList *MLTAL,
                              const ASTTemplateArgumentListInfo *ArgsAsWritten,
                              bool InFoldExpr)
      : SemaRef(SemaRef), MLTAL(MLTAL), ArgsAsWritten(ArgsAsWritten),
        InFoldExpr(InFoldExpr) {}

  void buildParameterMapping(NormalizedConstraintWithParamMapping &N);

  bool substitute(NormalizedConstraintWithParamMapping &N);

  bool substitute(ConceptIdConstraint &CC);

public:
  SubstituteParameterMappings(Sema &SemaRef, bool InFoldExpr = false)
      : SemaRef(SemaRef), MLTAL(nullptr), ArgsAsWritten(nullptr),
        InFoldExpr(InFoldExpr) {}

  bool substitute(NormalizedConstraint &N);
};

void SubstituteParameterMappings::buildParameterMapping(
    NormalizedConstraintWithParamMapping &N) {
  TemplateParameterList *TemplateParams =
      cast<TemplateDecl>(N.getConstraintDecl())->getTemplateParameters();

  llvm::SmallBitVector OccurringIndices(TemplateParams->size());
  llvm::SmallBitVector OccurringIndicesForSubsumption(TemplateParams->size());

  if (N.getKind() == NormalizedConstraint::ConstraintKind::Atomic) {
    SemaRef.MarkUsedTemplateParameters(
        static_cast<AtomicConstraint &>(N).getConstraintExpr(),
        /*OnlyDeduced=*/false,
        /*Depth=*/0, OccurringIndices);

    SemaRef.MarkUsedTemplateParametersForSubsumptionParameterMapping(
        static_cast<AtomicConstraint &>(N).getConstraintExpr(),
        /*Depth=*/0, OccurringIndicesForSubsumption);

  } else if (N.getKind() ==
             NormalizedConstraint::ConstraintKind::FoldExpanded) {
    SemaRef.MarkUsedTemplateParameters(
        static_cast<FoldExpandedConstraint &>(N).getPattern(),
        /*OnlyDeduced=*/false,
        /*Depth=*/0, OccurringIndices);
  } else if (N.getKind() == NormalizedConstraint::ConstraintKind::ConceptId) {
    auto *Args = static_cast<ConceptIdConstraint &>(N)
                     .getConceptId()
                     ->getTemplateArgsAsWritten();
    if (Args)
      SemaRef.MarkUsedTemplateParameters(Args->arguments(),
                                         /*Depth=*/0, OccurringIndices);
  }
  TemplateArgumentLoc *TempArgs =
      new (SemaRef.Context) TemplateArgumentLoc[OccurringIndices.count()];
  llvm::SmallVector<NamedDecl *> UsedParams;
  for (unsigned I = 0, J = 0, C = TemplateParams->size(); I != C; ++I) {
    SourceLocation Loc = ArgsAsWritten->NumTemplateArgs > I
                             ? ArgsAsWritten->arguments()[I].getLocation()
                             : SourceLocation();
    // FIXME: Investigate why we couldn't always preserve the SourceLoc. We
    // can't assert Loc.isValid() now.
    if (OccurringIndices[I]) {
      NamedDecl *Param = TemplateParams->begin()[I];
      new (&(TempArgs)[J]) TemplateArgumentLoc(
          SemaRef.getIdentityTemplateArgumentLoc(Param, Loc));
      UsedParams.push_back(Param);
      J++;
    }
  }
  auto *UsedList = TemplateParameterList::Create(
      SemaRef.Context, TemplateParams->getTemplateLoc(),
      TemplateParams->getLAngleLoc(), UsedParams,
      /*RAngleLoc=*/SourceLocation(),
      /*RequiresClause=*/nullptr);
  unsigned Size = OccurringIndices.count();
  N.updateParameterMapping(
      std::move(OccurringIndices), std::move(OccurringIndicesForSubsumption),
      MutableArrayRef<TemplateArgumentLoc>{TempArgs, Size}, UsedList);
}

bool SubstituteParameterMappings::substitute(
    NormalizedConstraintWithParamMapping &N) {
  if (!N.hasParameterMapping())
    buildParameterMapping(N);

  SourceLocation InstLocBegin, InstLocEnd;
  llvm::ArrayRef Arguments = ArgsAsWritten->arguments();
  if (Arguments.empty()) {
    InstLocBegin = ArgsAsWritten->getLAngleLoc();
    InstLocEnd = ArgsAsWritten->getRAngleLoc();
  } else {
    auto SR = Arguments[0].getSourceRange();
    InstLocBegin = SR.getBegin();
    InstLocEnd = SR.getEnd();
  }
  Sema::InstantiatingTemplate Inst(
      SemaRef, InstLocBegin,
      Sema::InstantiatingTemplate::ParameterMappingSubstitution{},
      const_cast<NamedDecl *>(N.getConstraintDecl()),
      {InstLocBegin, InstLocEnd});
  if (Inst.isInvalid())
    return true;

  // TransformTemplateArguments is unable to preserve the source location of a
  // pack. The SourceLocation is necessary for the instantiation location.
  // FIXME: The BaseLoc will be used as the location of the pack expansion,
  // which is wrong.
  TemplateArgumentListInfo SubstArgs;
  if (SemaRef.SubstTemplateArgumentsInParameterMapping(
          N.getParameterMapping(), N.getBeginLoc(), *MLTAL, SubstArgs,
          /*BuildPackExpansionTypes=*/!InFoldExpr))
    return true;
  Sema::CheckTemplateArgumentInfo CTAI;
  auto *TD =
      const_cast<TemplateDecl *>(cast<TemplateDecl>(N.getConstraintDecl()));
  if (SemaRef.CheckTemplateArgumentList(TD, N.getUsedTemplateParamList(),
                                        TD->getLocation(), SubstArgs,
                                        /*DefaultArguments=*/{},
                                        /*PartialTemplateArgs=*/false, CTAI))
    return true;

  TemplateArgumentLoc *TempArgs =
      new (SemaRef.Context) TemplateArgumentLoc[CTAI.SugaredConverted.size()];

  for (unsigned I = 0; I < CTAI.SugaredConverted.size(); ++I) {
    SourceLocation Loc;
    // If this is an empty pack, we have no corresponding SubstArgs.
    if (I < SubstArgs.size())
      Loc = SubstArgs.arguments()[I].getLocation();

    TempArgs[I] = SemaRef.getTrivialTemplateArgumentLoc(
        CTAI.SugaredConverted[I], QualType(), Loc);
  }

  MutableArrayRef<TemplateArgumentLoc> Mapping(TempArgs,
                                               CTAI.SugaredConverted.size());
  N.updateParameterMapping(N.mappingOccurenceList(),
                           N.mappingOccurenceListForSubsumption(), Mapping,
                           N.getUsedTemplateParamList());
  return false;
}

bool SubstituteParameterMappings::substitute(ConceptIdConstraint &CC) {
  assert(CC.getConstraintDecl() && MLTAL && ArgsAsWritten);

  if (substitute(static_cast<NormalizedConstraintWithParamMapping &>(CC)))
    return true;

  auto *CSE = CC.getConceptSpecializationExpr();
  assert(CSE);
  assert(!CC.getBeginLoc().isInvalid());

  SourceLocation InstLocBegin, InstLocEnd;
  if (llvm::ArrayRef Arguments = ArgsAsWritten->arguments();
      Arguments.empty()) {
    InstLocBegin = ArgsAsWritten->getLAngleLoc();
    InstLocEnd = ArgsAsWritten->getRAngleLoc();
  } else {
    auto SR = Arguments[0].getSourceRange();
    InstLocBegin = SR.getBegin();
    InstLocEnd = SR.getEnd();
  }
  // This is useful for name lookup across modules; see Sema::getLookupModules.
  Sema::InstantiatingTemplate Inst(
      SemaRef, InstLocBegin,
      Sema::InstantiatingTemplate::ParameterMappingSubstitution{},
      const_cast<NamedDecl *>(CC.getConstraintDecl()),
      {InstLocBegin, InstLocEnd});
  if (Inst.isInvalid())
    return true;

  TemplateArgumentListInfo Out;
  // TransformTemplateArguments is unable to preserve the source location of a
  // pack. The SourceLocation is necessary for the instantiation location.
  // FIXME: The BaseLoc will be used as the location of the pack expansion,
  // which is wrong.
  const ASTTemplateArgumentListInfo *ArgsAsWritten =
      CSE->getTemplateArgsAsWritten();
  if (SemaRef.SubstTemplateArgumentsInParameterMapping(
          ArgsAsWritten->arguments(), CC.getBeginLoc(), *MLTAL, Out,
          /*BuildPackExpansionTypes=*/!InFoldExpr))
    return true;
  Sema::CheckTemplateArgumentInfo CTAI;
  if (SemaRef.CheckTemplateArgumentList(CSE->getNamedConcept(),
                                        CSE->getConceptNameInfo().getLoc(), Out,
                                        /*DefaultArgs=*/{},
                                        /*PartialTemplateArgs=*/false, CTAI,
                                        /*UpdateArgsWithConversions=*/false))
    return true;
  auto TemplateArgs = *MLTAL;
  TemplateArgs.replaceOutermostTemplateArguments(CSE->getNamedConcept(),
                                                 CTAI.SugaredConverted);
  return SubstituteParameterMappings(SemaRef, &TemplateArgs, ArgsAsWritten,
                                     InFoldExpr)
      .substitute(CC.getNormalizedConstraint());
}

bool SubstituteParameterMappings::substitute(NormalizedConstraint &N) {
  switch (N.getKind()) {
  case NormalizedConstraint::ConstraintKind::Atomic: {
    if (!MLTAL) {
      assert(!ArgsAsWritten);
      return false;
    }
    return substitute(static_cast<NormalizedConstraintWithParamMapping &>(N));
  }
  case NormalizedConstraint::ConstraintKind::FoldExpanded: {
    auto &FE = static_cast<FoldExpandedConstraint &>(N);
    if (!MLTAL) {
      llvm::SaveAndRestore _1(InFoldExpr, true);
      assert(!ArgsAsWritten);
      return substitute(FE.getNormalizedPattern());
    }
    Sema::ArgPackSubstIndexRAII _(SemaRef, std::nullopt);
    substitute(static_cast<NormalizedConstraintWithParamMapping &>(FE));
    return SubstituteParameterMappings(SemaRef, /*InFoldExpr=*/true)
        .substitute(FE.getNormalizedPattern());
  }
  case NormalizedConstraint::ConstraintKind::ConceptId: {
    auto &CC = static_cast<ConceptIdConstraint &>(N);
    if (MLTAL) {
      assert(ArgsAsWritten);
      return substitute(CC);
    }
    assert(!ArgsAsWritten);
    const ConceptSpecializationExpr *CSE = CC.getConceptSpecializationExpr();
    ConceptDecl *Concept = CSE->getNamedConcept();
    MultiLevelTemplateArgumentList MLTAL = SemaRef.getTemplateInstantiationArgs(
        Concept, Concept->getLexicalDeclContext(),
        /*Final=*/true, CSE->getTemplateArguments(),
        /*RelativeToPrimary=*/true,
        /*Pattern=*/nullptr,
        /*ForConstraintInstantiation=*/true);

    return SubstituteParameterMappings(
               SemaRef, &MLTAL, CSE->getTemplateArgsAsWritten(), InFoldExpr)
        .substitute(CC.getNormalizedConstraint());
  }
  case NormalizedConstraint::ConstraintKind::Compound: {
    auto &Compound = static_cast<CompoundConstraint &>(N);
    if (substitute(Compound.getLHS()))
      return true;
    return substitute(Compound.getRHS());
  }
  }
  llvm_unreachable("Unknown ConstraintKind enum");
}

} // namespace

NormalizedConstraint *NormalizedConstraint::fromAssociatedConstraints(
    Sema &S, const NamedDecl *D, ArrayRef<AssociatedConstraint> ACs) {
  assert(ACs.size() != 0);
  auto *Conjunction =
      fromConstraintExpr(S, D, ACs[0].ConstraintExpr, ACs[0].ArgPackSubstIndex);
  if (!Conjunction)
    return nullptr;
  for (unsigned I = 1; I < ACs.size(); ++I) {
    auto *Next = fromConstraintExpr(S, D, ACs[I].ConstraintExpr,
                                    ACs[I].ArgPackSubstIndex);
    if (!Next)
      return nullptr;
    Conjunction = CompoundConstraint::CreateConjunction(S.getASTContext(),
                                                        Conjunction, Next);
  }
  return Conjunction;
}

NormalizedConstraint *NormalizedConstraint::fromConstraintExpr(
    Sema &S, const NamedDecl *D, const Expr *E, UnsignedOrNone SubstIndex) {
  assert(E != nullptr);

  // C++ [temp.constr.normal]p1.1
  // [...]
  // - The normal form of an expression (E) is the normal form of E.
  // [...]
  E = E->IgnoreParenImpCasts();

  llvm::FoldingSetNodeID ID;
  if (D && DiagRecursiveConstraintEval(S, ID, D, E)) {
    return nullptr;
  }
  SatisfactionStackRAII StackRAII(S, D, ID);

  // C++2a [temp.param]p4:
  //     [...] If T is not a pack, then E is E', otherwise E is (E' && ...).
  // Fold expression is considered atomic constraints per current wording.
  // See http://cplusplus.github.io/concepts-ts/ts-active.html#28

  if (LogicalBinOp BO = E) {
    auto *LHS = fromConstraintExpr(S, D, BO.getLHS(), SubstIndex);
    if (!LHS)
      return nullptr;
    auto *RHS = fromConstraintExpr(S, D, BO.getRHS(), SubstIndex);
    if (!RHS)
      return nullptr;

    return CompoundConstraint::Create(
        S.Context, LHS, BO.isAnd() ? CCK_Conjunction : CCK_Disjunction, RHS);
  } else if (auto *CSE = dyn_cast<const ConceptSpecializationExpr>(E)) {
    NormalizedConstraint *SubNF;
    {
      Sema::InstantiatingTemplate Inst(
          S, CSE->getExprLoc(),
          Sema::InstantiatingTemplate::ConstraintNormalization{},
          // FIXME: improve const-correctness of InstantiatingTemplate
          const_cast<NamedDecl *>(D), CSE->getSourceRange());
      if (Inst.isInvalid())
        return nullptr;
      // C++ [temp.constr.normal]p1.1
      // [...]
      // The normal form of an id-expression of the form C<A1, A2, ..., AN>,
      // where C names a concept, is the normal form of the
      // constraint-expression of C, after substituting A1, A2, ..., AN for C’s
      // respective template parameters in the parameter mappings in each atomic
      // constraint. If any such substitution results in an invalid type or
      // expression, the program is ill-formed; no diagnostic is required.
      // [...]

      // Use canonical declarations to merge ConceptDecls across
      // different modules.
      ConceptDecl *CD = CSE->getNamedConcept()->getCanonicalDecl();
      SubNF = NormalizedConstraint::fromAssociatedConstraints(
          S, CD, AssociatedConstraint(CD->getConstraintExpr(), SubstIndex));

      if (!SubNF)
        return nullptr;
    }

    return ConceptIdConstraint::Create(S.getASTContext(),
                                       CSE->getConceptReference(), SubNF, D,
                                       CSE, SubstIndex);

  } else if (auto *FE = dyn_cast<const CXXFoldExpr>(E);
             FE && S.getLangOpts().CPlusPlus26 &&
             (FE->getOperator() == BinaryOperatorKind::BO_LAnd ||
              FE->getOperator() == BinaryOperatorKind::BO_LOr)) {

    // Normalize fold expressions in C++26.

    FoldExpandedConstraint::FoldOperatorKind Kind =
        FE->getOperator() == BinaryOperatorKind::BO_LAnd
            ? FoldExpandedConstraint::FoldOperatorKind::And
            : FoldExpandedConstraint::FoldOperatorKind::Or;

    if (FE->getInit()) {
      auto *LHS = fromConstraintExpr(S, D, FE->getLHS(), SubstIndex);
      auto *RHS = fromConstraintExpr(S, D, FE->getRHS(), SubstIndex);
      if (!LHS || !RHS)
        return nullptr;

      if (FE->isRightFold())
        LHS = FoldExpandedConstraint::Create(S.getASTContext(),
                                             FE->getPattern(), D, Kind, LHS);
      else
        RHS = FoldExpandedConstraint::Create(S.getASTContext(),
                                             FE->getPattern(), D, Kind, RHS);

      return CompoundConstraint::Create(
          S.getASTContext(), LHS,
          (FE->getOperator() == BinaryOperatorKind::BO_LAnd ? CCK_Conjunction
                                                            : CCK_Disjunction),
          RHS);
    }
    auto *Sub = fromConstraintExpr(S, D, FE->getPattern(), SubstIndex);
    if (!Sub)
      return nullptr;
    return FoldExpandedConstraint::Create(S.getASTContext(), FE->getPattern(),
                                          D, Kind, Sub);
  }
  return AtomicConstraint::Create(S.getASTContext(), E, D, SubstIndex);
}

const NormalizedConstraint *Sema::getNormalizedAssociatedConstraints(
    ConstrainedDeclOrNestedRequirement ConstrainedDeclOrNestedReq,
    ArrayRef<AssociatedConstraint> AssociatedConstraints) {
  if (!ConstrainedDeclOrNestedReq) {
    auto *Normalized = NormalizedConstraint::fromAssociatedConstraints(
        *this, nullptr, AssociatedConstraints);
    if (!Normalized ||
        SubstituteParameterMappings(*this).substitute(*Normalized))
      return nullptr;

    return Normalized;
  }

  // FIXME: ConstrainedDeclOrNestedReq is never a NestedRequirement!
  const NamedDecl *ND =
      ConstrainedDeclOrNestedReq.dyn_cast<const NamedDecl *>();
  auto CacheEntry = NormalizationCache.find(ConstrainedDeclOrNestedReq);
  if (CacheEntry == NormalizationCache.end()) {
    auto *Normalized = NormalizedConstraint::fromAssociatedConstraints(
        *this, ND, AssociatedConstraints);
    CacheEntry =
        NormalizationCache.try_emplace(ConstrainedDeclOrNestedReq, Normalized)
            .first;
    if (!Normalized ||
        SubstituteParameterMappings(*this).substitute(*Normalized))
      return nullptr;
  }
  return CacheEntry->second;
}

bool FoldExpandedConstraint::AreCompatibleForSubsumption(
    const FoldExpandedConstraint &A, const FoldExpandedConstraint &B) {

  // [C++26] [temp.constr.fold]
  // Two fold expanded constraints are compatible for subsumption
  // if their respective constraints both contain an equivalent unexpanded pack.

  llvm::SmallVector<UnexpandedParameterPack> APacks, BPacks;
  Sema::collectUnexpandedParameterPacks(const_cast<Expr *>(A.getPattern()),
                                        APacks);
  Sema::collectUnexpandedParameterPacks(const_cast<Expr *>(B.getPattern()),
                                        BPacks);

  for (const UnexpandedParameterPack &APack : APacks) {
    auto ADI = getDepthAndIndex(APack);
    if (!ADI)
      continue;
    auto It = llvm::find_if(BPacks, [&](const UnexpandedParameterPack &BPack) {
      return getDepthAndIndex(BPack) == ADI;
    });
    if (It != BPacks.end())
      return true;
  }
  return false;
}

bool Sema::IsAtLeastAsConstrained(const NamedDecl *D1,
                                  MutableArrayRef<AssociatedConstraint> AC1,
                                  const NamedDecl *D2,
                                  MutableArrayRef<AssociatedConstraint> AC2,
                                  bool &Result) {
#ifndef NDEBUG
  if (const auto *FD1 = dyn_cast<FunctionDecl>(D1)) {
    auto IsExpectedEntity = [](const FunctionDecl *FD) {
      FunctionDecl::TemplatedKind Kind = FD->getTemplatedKind();
      return Kind == FunctionDecl::TK_NonTemplate ||
             Kind == FunctionDecl::TK_FunctionTemplate;
    };
    const auto *FD2 = dyn_cast<FunctionDecl>(D2);
    assert(IsExpectedEntity(FD1) && FD2 && IsExpectedEntity(FD2) &&
           "use non-instantiated function declaration for constraints partial "
           "ordering");
  }
#endif

  if (AC1.empty()) {
    Result = AC2.empty();
    return false;
  }
  if (AC2.empty()) {
    // TD1 has associated constraints and TD2 does not.
    Result = true;
    return false;
  }

  std::pair<const NamedDecl *, const NamedDecl *> Key{D1, D2};
  auto CacheEntry = SubsumptionCache.find(Key);
  if (CacheEntry != SubsumptionCache.end()) {
    Result = CacheEntry->second;
    return false;
  }

  unsigned Depth1 = CalculateTemplateDepthForConstraints(*this, D1, true);
  unsigned Depth2 = CalculateTemplateDepthForConstraints(*this, D2, true);

  for (size_t I = 0; I != AC1.size() && I != AC2.size(); ++I) {
    if (Depth2 > Depth1) {
      AC1[I].ConstraintExpr =
          AdjustConstraintDepth(*this, Depth2 - Depth1)
              .TransformExpr(const_cast<Expr *>(AC1[I].ConstraintExpr))
              .get();
    } else if (Depth1 > Depth2) {
      AC2[I].ConstraintExpr =
          AdjustConstraintDepth(*this, Depth1 - Depth2)
              .TransformExpr(const_cast<Expr *>(AC2[I].ConstraintExpr))
              .get();
    }
  }

  SubsumptionChecker SC(*this);
  std::optional<bool> Subsumes = SC.Subsumes(D1, AC1, D2, AC2);
  if (!Subsumes) {
    // Normalization failed
    return true;
  }
  Result = *Subsumes;
  SubsumptionCache.try_emplace(Key, *Subsumes);
  return false;
}

bool Sema::MaybeEmitAmbiguousAtomicConstraintsDiagnostic(
    const NamedDecl *D1, ArrayRef<AssociatedConstraint> AC1,
    const NamedDecl *D2, ArrayRef<AssociatedConstraint> AC2) {
  if (isSFINAEContext())
    // No need to work here because our notes would be discarded.
    return false;

  if (AC1.empty() || AC2.empty())
    return false;

  const Expr *AmbiguousAtomic1 = nullptr, *AmbiguousAtomic2 = nullptr;
  auto IdenticalExprEvaluator = [&](const AtomicConstraint &A,
                                    const AtomicConstraint &B) {
    if (!A.hasMatchingParameterMapping(Context, B))
      return false;
    const Expr *EA = A.getConstraintExpr(), *EB = B.getConstraintExpr();
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

    auto *Normalized2 = getNormalizedAssociatedConstraints(D2, AC2);
    if (!Normalized2)
      return false;

    SubsumptionChecker SC(*this);

    bool Is1AtLeastAs2Normally = SC.Subsumes(Normalized1, Normalized2);
    bool Is2AtLeastAs1Normally = SC.Subsumes(Normalized2, Normalized1);

    SubsumptionChecker SC2(*this, IdenticalExprEvaluator);
    bool Is1AtLeastAs2 = SC2.Subsumes(Normalized1, Normalized2);
    bool Is2AtLeastAs1 = SC2.Subsumes(Normalized2, Normalized1);

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

//
//
// ------------------------ Subsumption -----------------------------------
//
//
SubsumptionChecker::SubsumptionChecker(Sema &SemaRef,
                                       SubsumptionCallable Callable)
    : SemaRef(SemaRef), Callable(Callable), NextID(1) {}

uint16_t SubsumptionChecker::getNewLiteralId() {
  assert((unsigned(NextID) + 1 < std::numeric_limits<uint16_t>::max()) &&
         "too many constraints!");
  return NextID++;
}

auto SubsumptionChecker::find(const AtomicConstraint *Ori) -> Literal {
  auto &Elems = AtomicMap[Ori->getConstraintExpr()];
  // C++ [temp.constr.order] p2
  //   - an atomic constraint A subsumes another atomic constraint B
  //     if and only if the A and B are identical [...]
  //
  // C++ [temp.constr.atomic] p2
  //   Two atomic constraints are identical if they are formed from the
  //   same expression and the targets of the parameter mappings are
  //   equivalent according to the rules for expressions [...]

  // Because subsumption of atomic constraints is an identity
  // relationship that does not require further analysis
  // We cache the results such that if an atomic constraint literal
  // subsumes another, their literal will be the same

  llvm::FoldingSetNodeID ID;
  ID.AddBoolean(Ori->hasParameterMapping());
  if (Ori->hasParameterMapping()) {
    const auto &Mapping = Ori->getParameterMapping();
    const NormalizedConstraint::OccurenceList &Indexes =
        Ori->mappingOccurenceListForSubsumption();
    for (auto [Idx, TAL] : llvm::enumerate(Mapping)) {
      if (Indexes[Idx])
        SemaRef.getASTContext()
            .getCanonicalTemplateArgument(TAL.getArgument())
            .Profile(ID, SemaRef.getASTContext());
    }
  }
  auto It = Elems.find(ID);
  if (It == Elems.end()) {
    It = Elems
             .insert({ID,
                      MappedAtomicConstraint{
                          Ori, {getNewLiteralId(), Literal::Atomic}}})
             .first;
    ReverseMap[It->second.ID.Value] = Ori;
  }
  return It->getSecond().ID;
}

auto SubsumptionChecker::find(const FoldExpandedConstraint *Ori) -> Literal {
  auto &Elems = FoldMap[Ori->getPattern()];

  FoldExpendedConstraintKey K;
  K.Kind = Ori->getFoldOperator();

  auto It = llvm::find_if(Elems, [&K](const FoldExpendedConstraintKey &Other) {
    return K.Kind == Other.Kind;
  });
  if (It == Elems.end()) {
    K.ID = {getNewLiteralId(), Literal::FoldExpanded};
    It = Elems.insert(Elems.end(), std::move(K));
    ReverseMap[It->ID.Value] = Ori;
  }
  return It->ID;
}

auto SubsumptionChecker::CNF(const NormalizedConstraint &C) -> CNFFormula {
  return SubsumptionChecker::Normalize<CNFFormula>(C);
}
auto SubsumptionChecker::DNF(const NormalizedConstraint &C) -> DNFFormula {
  return SubsumptionChecker::Normalize<DNFFormula>(C);
}

///
/// \brief SubsumptionChecker::Normalize
///
/// Normalize a formula to Conjunctive Normal Form or
/// Disjunctive normal form.
///
/// Each Atomic (and Fold Expanded) constraint gets represented by
/// a single id to reduce space.
///
/// To minimize risks of exponential blow up, if two atomic
/// constraints subsumes each other (same constraint and mapping),
/// they are represented by the same literal.
///
template <typename FormulaType>
FormulaType SubsumptionChecker::Normalize(const NormalizedConstraint &NC) {
  FormulaType Res;

  auto Add = [&, this](Clause C) {
    // Sort each clause and remove duplicates for faster comparisons.
    llvm::sort(C);
    C.erase(llvm::unique(C), C.end());
    AddUniqueClauseToFormula(Res, std::move(C));
  };

  switch (NC.getKind()) {
  case NormalizedConstraint::ConstraintKind::Atomic:
    return {{find(&static_cast<const AtomicConstraint &>(NC))}};

  case NormalizedConstraint::ConstraintKind::FoldExpanded:
    return {{find(&static_cast<const FoldExpandedConstraint &>(NC))}};

  case NormalizedConstraint::ConstraintKind::ConceptId:
    return Normalize<FormulaType>(
        static_cast<const ConceptIdConstraint &>(NC).getNormalizedConstraint());

  case NormalizedConstraint::ConstraintKind::Compound: {
    const auto &Compound = static_cast<const CompoundConstraint &>(NC);
    FormulaType Left, Right;
    SemaRef.runWithSufficientStackSpace(SourceLocation(), [&] {
      Left = Normalize<FormulaType>(Compound.getLHS());
      Right = Normalize<FormulaType>(Compound.getRHS());
    });

    if (Compound.getCompoundKind() == FormulaType::Kind) {
      Res = std::move(Left);
      Res.reserve(Left.size() + Right.size());
      std::for_each(std::make_move_iterator(Right.begin()),
                    std::make_move_iterator(Right.end()), Add);
      return Res;
    }

    Res.reserve(Left.size() * Right.size());
    for (const auto &LTransform : Left) {
      for (const auto &RTransform : Right) {
        Clause Combined;
        Combined.reserve(LTransform.size() + RTransform.size());
        llvm::copy(LTransform, std::back_inserter(Combined));
        llvm::copy(RTransform, std::back_inserter(Combined));
        Add(std::move(Combined));
      }
    }
    return Res;
  }
  }
  llvm_unreachable("Unknown ConstraintKind enum");
}

void SubsumptionChecker::AddUniqueClauseToFormula(Formula &F, Clause C) {
  for (auto &Other : F) {
    if (llvm::equal(C, Other))
      return;
  }
  F.push_back(C);
}

std::optional<bool> SubsumptionChecker::Subsumes(
    const NamedDecl *DP, ArrayRef<AssociatedConstraint> P, const NamedDecl *DQ,
    ArrayRef<AssociatedConstraint> Q) {
  const NormalizedConstraint *PNormalized =
      SemaRef.getNormalizedAssociatedConstraints(DP, P);
  if (!PNormalized)
    return std::nullopt;

  const NormalizedConstraint *QNormalized =
      SemaRef.getNormalizedAssociatedConstraints(DQ, Q);
  if (!QNormalized)
    return std::nullopt;

  return Subsumes(PNormalized, QNormalized);
}

bool SubsumptionChecker::Subsumes(const NormalizedConstraint *P,
                                  const NormalizedConstraint *Q) {

  DNFFormula DNFP = DNF(*P);
  CNFFormula CNFQ = CNF(*Q);
  return Subsumes(DNFP, CNFQ);
}

bool SubsumptionChecker::Subsumes(const DNFFormula &PDNF,
                                  const CNFFormula &QCNF) {
  for (const auto &Pi : PDNF) {
    for (const auto &Qj : QCNF) {
      // C++ [temp.constr.order] p2
      //   - [...] a disjunctive clause Pi subsumes a conjunctive clause Qj if
      //     and only if there exists an atomic constraint Pia in Pi for which
      //     there exists an atomic constraint, Qjb, in Qj such that Pia
      //     subsumes Qjb.
      if (!DNFSubsumes(Pi, Qj))
        return false;
    }
  }
  return true;
}

bool SubsumptionChecker::DNFSubsumes(const Clause &P, const Clause &Q) {

  return llvm::any_of(P, [&](Literal LP) {
    return llvm::any_of(Q, [this, LP](Literal LQ) { return Subsumes(LP, LQ); });
  });
}

bool SubsumptionChecker::Subsumes(const FoldExpandedConstraint *A,
                                  const FoldExpandedConstraint *B) {
  std::pair<const FoldExpandedConstraint *, const FoldExpandedConstraint *> Key{
      A, B};

  auto It = FoldSubsumptionCache.find(Key);
  if (It == FoldSubsumptionCache.end()) {
    // C++ [temp.constr.order]
    // a fold expanded constraint A subsumes another fold expanded
    // constraint B if they are compatible for subsumption, have the same
    // fold-operator, and the constraint of A subsumes that of B.
    bool DoesSubsume =
        A->getFoldOperator() == B->getFoldOperator() &&
        FoldExpandedConstraint::AreCompatibleForSubsumption(*A, *B) &&
        Subsumes(&A->getNormalizedPattern(), &B->getNormalizedPattern());
    It = FoldSubsumptionCache.try_emplace(std::move(Key), DoesSubsume).first;
  }
  return It->second;
}

bool SubsumptionChecker::Subsumes(Literal A, Literal B) {
  if (A.Kind != B.Kind)
    return false;
  switch (A.Kind) {
  case Literal::Atomic:
    if (!Callable)
      return A.Value == B.Value;
    return Callable(
        *static_cast<const AtomicConstraint *>(ReverseMap[A.Value]),
        *static_cast<const AtomicConstraint *>(ReverseMap[B.Value]));
  case Literal::FoldExpanded:
    return Subsumes(
        static_cast<const FoldExpandedConstraint *>(ReverseMap[A.Value]),
        static_cast<const FoldExpandedConstraint *>(ReverseMap[B.Value]));
  }
  llvm_unreachable("unknown literal kind");
}
