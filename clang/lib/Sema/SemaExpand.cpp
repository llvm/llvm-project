//===-- SemaExpand.cpp - Semantic Analysis for Expansion Statements--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++26 expansion statements,
//  aka 'template for'.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace sema;

namespace {
struct IterableExpansionStmtData {
  enum class State {
    NotIterable,
    Error,
    Ok,
  };

  DeclStmt *RangeDecl = nullptr;
  DeclStmt *BeginDecl = nullptr;
  DeclStmt *EndDecl = nullptr;
  Expr *Initializer = nullptr;
  State TheState = State::NotIterable;

  bool isIterable() const { return TheState == State::Ok; }
  bool hasError() { return TheState == State::Error; }
};
} // namespace

static bool CheckExpansionSize(Sema &S, uint64_t NumInstantiations,
                               SourceLocation Loc) {
  unsigned Max = S.LangOpts.MaxTemplateForExpansions;
  if (Max != 0 && NumInstantiations > Max) {
    S.Diag(Loc, diag::err_expansion_too_big) << NumInstantiations << Max;
    S.Diag(Loc, diag::note_use_fexpansion_limit);
    return true;
  }

  return false;
}

// Build a 'DeclRefExpr' designating the template parameter '__N'.
static DeclRefExpr *BuildIndexDRE(Sema &S, CXXExpansionStmtDecl *ESD) {
  return S.BuildDeclRefExpr(ESD->getIndexTemplateParm(),
                            S.Context.getPointerDiffType(), VK_PRValue,
                            ESD->getBeginLoc());
}

static bool FinaliseExpansionVar(Sema &S, VarDecl *ExpansionVar,
                                 ExprResult Initializer) {
  if (Initializer.isInvalid()) {
    S.ActOnInitializerError(ExpansionVar);
    return true;
  }

  S.AddInitializerToDecl(ExpansionVar, Initializer.get(), /*DirectInit=*/false);
  return ExpansionVar->isInvalidDecl();
}

static auto InitListContainsPack(const InitListExpr *ILE) {
  return llvm::any_of(ArrayRef(ILE->getInits(), ILE->getNumInits()),
                      [](const Expr *E) { return isa<PackExpansionExpr>(E); });
}

static bool HasDependentSize(const CXXExpansionStmtPattern *Pattern) {
  switch (Pattern->getKind()) {
  case CXXExpansionStmtPattern::ExpansionStmtKind::Enumerating: {
    auto *SelectExpr = cast<CXXExpansionSelectExpr>(
        Pattern->getExpansionVariable()->getInit());
    return InitListContainsPack(SelectExpr->getRangeExpr());
  }

  case CXXExpansionStmtPattern::ExpansionStmtKind::Iterating: {
    const Expr *Begin = Pattern->getBeginVar()->getInit();
    const Expr *End = Pattern->getEndVar()->getInit();
    return Begin->isInstantiationDependent() || End->isInstantiationDependent();
  }

  case CXXExpansionStmtPattern::ExpansionStmtKind::Dependent:
    return true;

  case CXXExpansionStmtPattern::ExpansionStmtKind::Destructuring:
    return false;
  }

  llvm_unreachable("invalid pattern kind");
}

static IterableExpansionStmtData
TryBuildIterableExpansionStmtInitializer(Sema &S, Expr *ExpansionInitializer,
                                         Expr *Index, SourceLocation ColonLoc,
                                         bool VarIsConstexpr) {
  IterableExpansionStmtData Data;

  // C++26 [stmt.expand]p3: An expression is expansion-iterable if it does not
  // have array type [...]
  QualType Ty = ExpansionInitializer->getType().getNonReferenceType();
  if (Ty->isArrayType())
    return Data;

  // Lookup member and ADL 'begin()'/'end()'. Only check if they exist; even if
  // they're deleted, inaccessible, etc., this is still an iterating expansion
  // statement, albeit an ill-formed one.
  DeclarationNameInfo BeginName(&S.PP.getIdentifierTable().get("begin"),
                                ColonLoc);
  DeclarationNameInfo EndName(&S.PP.getIdentifierTable().get("end"), ColonLoc);

  // Try member lookup first.
  bool FoundBeginEnd = false;
  if (auto *Record = Ty->getAsCXXRecordDecl()) {
    LookupResult BeginLR(S, BeginName, Sema::LookupMemberName);
    LookupResult EndLR(S, EndName, Sema::LookupMemberName);
    FoundBeginEnd = S.LookupQualifiedName(BeginLR, Record) &&
                    S.LookupQualifiedName(EndLR, Record);
  }

  // Try ADL.
  //
  // If overload resolution for 'begin()' *and* 'end()' succeeds (irrespective
  // of whether it results in a usable candidate), then assume this is an
  // iterating expansion statement.
  auto HasADLCandidate = [&](DeclarationName Name) {
    OverloadCandidateSet Candidates(ColonLoc, OverloadCandidateSet::CSK_Normal);
    OverloadCandidateSet::iterator Best;

    S.AddArgumentDependentLookupCandidates(Name, ColonLoc, ExpansionInitializer,
                                           /*ExplicitTemplateArgs=*/nullptr,
                                           Candidates);

    return Candidates.BestViableFunction(S, ColonLoc, Best) !=
           OR_No_Viable_Function;
  };

  if (!FoundBeginEnd && (!HasADLCandidate(BeginName.getName()) ||
                         !HasADLCandidate(EndName.getName())))
    return Data;

  auto Ctx = Sema::ExpressionEvaluationContext::PotentiallyEvaluated;
  if (VarIsConstexpr)
    Ctx = Sema::ExpressionEvaluationContext::ImmediateFunctionContext;
  EnterExpressionEvaluationContext ExprEvalCtx(S, Ctx);

  // The declarations should be attached to the parent decl context.
  Sema::ContextRAII CtxGuard(
      S, S.CurContext->getEnclosingNonExpansionStatementContext(),
      /*NewThis=*/false);

  // Ok, we know that this is supposed to be an iterable expansion statement;
  // delegate to the for-range code to build the range/begin/end variables.
  //
  // Any failure at this point is a hard error.
  Data.TheState = IterableExpansionStmtData::State::Error;
  Scope *Scope = S.getCurScope();

  // CWG 3131: The declaration of 'range' is of the form
  //
  //     constexpr[opt] decltype(auto) range = (expansion-initializer);
  //
  // where 'constexpr' is present iff the for-range-declaration is 'constexpr'.
  StmtResult Var = S.BuildCXXForRangeRangeVar(
      Scope, S.ActOnParenExpr(ColonLoc, ColonLoc, ExpansionInitializer).get(),
      S.Context.getAutoType(QualType(), AutoTypeKeyword::DecltypeAuto,
                            /*IsDependent*/ false),
      VarIsConstexpr);
  if (Var.isInvalid())
    return Data;

  // CWG 3131: Discussion around this core issue (though as of the time of
  // writing not the resolution itself) suggests that the other variables we
  // create here should likewise be 'constexpr' iff the range variable is
  // declared 'constexpr'.
  //
  // FIXME: As of CWG 3131, 'end' is no longer used outside the lambda that
  // performs the size calculation (despite that, CWG 3131 currently still
  // lists it in the generated code, but this is likely an oversight). Ideally,
  // we should only create 'begin' here instead, but that requires another
  // substantial refactor of the for-range code.
  auto *RangeVar = cast<DeclStmt>(Var.get());
  Sema::ForRangeBeginEndInfo Info = S.BuildCXXForRangeBeginEndVars(
      Scope, cast<VarDecl>(RangeVar->getSingleDecl()), ColonLoc,
      /*CoawaitLoc=*/{},
      /*LifetimeExtendTemps=*/{}, Sema::BFRK_Build, VarIsConstexpr);

  if (!Info.isValid())
    return Data;

  StmtResult BeginStmt = S.ActOnDeclStmt(
      S.ConvertDeclToDeclGroup(Info.BeginVar), ColonLoc, ColonLoc);
  StmtResult EndStmt = S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(Info.EndVar),
                                       ColonLoc, ColonLoc);
  if (BeginStmt.isInvalid() || EndStmt.isInvalid())
    return Data;

  // Build '*(begin + i)'.
  DeclRefExpr *Begin = S.BuildDeclRefExpr(
      Info.BeginVar, Info.BeginVar->getType().getNonReferenceType(), VK_LValue,
      ColonLoc);

  ExprResult BeginPlusI =
      S.ActOnBinOp(Scope, ColonLoc, tok::plus, Begin, Index);
  if (BeginPlusI.isInvalid())
    return Data;

  ExprResult Deref =
      S.ActOnUnaryOp(Scope, ColonLoc, tok::star, BeginPlusI.get());
  if (Deref.isInvalid())
    return Data;

  Deref = S.MaybeCreateExprWithCleanups(Deref.get());
  Data.BeginDecl = BeginStmt.getAs<DeclStmt>();
  Data.EndDecl = EndStmt.getAs<DeclStmt>();
  Data.RangeDecl = RangeVar;
  Data.Initializer = Deref.get();
  Data.TheState = IterableExpansionStmtData::State::Ok;
  return Data;
}

static StmtResult BuildDestructuringDecompositionDecl(
    Sema &S, Expr *ExpansionInitializer, SourceLocation ColonLoc,
    bool VarIsConstexpr,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  auto Ctx = Sema::ExpressionEvaluationContext::PotentiallyEvaluated;
  if (VarIsConstexpr)
    Ctx = Sema::ExpressionEvaluationContext::ImmediateFunctionContext;
  EnterExpressionEvaluationContext ExprEvalCtx(S, Ctx);

  // The declarations should be attached to the parent decl context.
  Sema::ContextRAII CtxGuard(
      S, S.CurContext->getEnclosingNonExpansionStatementContext(),
      /*NewThis=*/false);

  UnsignedOrNone Arity =
      S.GetDecompositionElementCount(ExpansionInitializer->getType(), ColonLoc);

  if (!Arity) {
    S.Diag(ExpansionInitializer->getBeginLoc(),
           diag::err_expansion_stmt_invalid_init)
        << ExpansionInitializer->getType()
        << ExpansionInitializer->getSourceRange();
    return StmtError();
  }

  if (CheckExpansionSize(S, *Arity, ColonLoc))
    return StmtError();

  QualType AutoRRef = S.Context.getAutoRRefDeductType();
  SmallVector<BindingDecl *> Bindings;
  for (unsigned I = 0; I < *Arity; ++I)
    Bindings.push_back(BindingDecl::Create(
        S.Context, S.CurContext, ColonLoc,
        S.getPreprocessor().getIdentifierInfo("__u" + std::to_string(I)),
        AutoRRef));

  TypeSourceInfo *TSI = S.Context.getTrivialTypeSourceInfo(AutoRRef);
  auto *DD =
      DecompositionDecl::Create(S.Context, S.CurContext, ColonLoc, ColonLoc,
                                AutoRRef, TSI, SC_Auto, Bindings);

  if (VarIsConstexpr)
    DD->setConstexpr(true);

  S.ApplyForRangeOrExpansionStatementLifetimeExtension(DD, LifetimeExtendTemps);
  S.AddInitializerToDecl(DD, ExpansionInitializer, false);
  return S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(DD), ColonLoc, ColonLoc);
}

CXXExpansionStmtDecl *
Sema::ActOnCXXExpansionStmtDecl(unsigned TemplateDepth,
                                SourceLocation TemplateKWLoc) {
  // Create a template parameter '__N'. This will be used to denote the index
  // of the element that we're instantiating. CWG 3044 requires this type to
  // be 'ptrdiff_t' for iterating expansion statements, so use that in all
  // cases.
  IdentifierInfo *ParmName = &Context.Idents.get("__N");
  QualType ParmTy = Context.getPointerDiffType();
  TypeSourceInfo *ParmTI =
      Context.getTrivialTypeSourceInfo(ParmTy, TemplateKWLoc);

  auto *TParam = NonTypeTemplateParmDecl::Create(
      Context, Context.getTranslationUnitDecl(), TemplateKWLoc, TemplateKWLoc,
      TemplateDepth, /*Position=*/0, ParmName, ParmTy, /*ParameterPack=*/false,
      ParmTI);

  return BuildCXXExpansionStmtDecl(CurContext, TemplateKWLoc, TParam);
}

CXXExpansionStmtDecl *
Sema::BuildCXXExpansionStmtDecl(DeclContext *Ctx, SourceLocation TemplateKWLoc,
                                NonTypeTemplateParmDecl *NTTP) {
  auto *Result =
      CXXExpansionStmtDecl::Create(Context, Ctx, TemplateKWLoc, NTTP);
  Ctx->addDecl(Result);
  return Result;
}

ExprResult Sema::ActOnCXXExpansionInitList(MultiExprArg SubExprs,
                                           SourceLocation LBraceLoc,
                                           SourceLocation RBraceLoc) {
  return new (Context) InitListExpr(Context, LBraceLoc, SubExprs, RBraceLoc);
}

StmtResult Sema::ActOnCXXExpansionStmtPattern(
    CXXExpansionStmtDecl *ESD, Stmt *Init, Stmt *ExpansionVarStmt,
    Expr *ExpansionInitializer, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  if (!ExpansionInitializer || !ExpansionVarStmt)
    return StmtError();

  assert(CurContext->isExpansionStmt());
  auto *DS = cast<DeclStmt>(ExpansionVarStmt);
  if (!DS->isSingleDecl()) {
    Diag(DS->getBeginLoc(), diag::err_type_defined_in_for_range);
    return StmtError();
  }

  VarDecl *ExpansionVar = dyn_cast<VarDecl>(DS->getSingleDecl());
  if (!ExpansionVar || ExpansionVar->isInvalidDecl() ||
      ExpansionInitializer->containsErrors())
    return StmtError();

  // This is an enumerating expansion statement.
  if (auto *ILE = dyn_cast<InitListExpr>(ExpansionInitializer)) {
    assert(ILE->isSyntacticForm());
    ExprResult Initializer =
        BuildCXXExpansionSelectExpr(ILE, BuildIndexDRE(*this, ESD));
    if (FinaliseExpansionVar(*this, ExpansionVar, Initializer))
      return StmtError();

    // Note that lifetime extension only applies to destructuring expansion
    // statements, so we just ignore 'LifetimeExtendedTemps' entirely for other
    // types of expansion statements (this is CWG 3043).
    //
    // TODO: CWG 3131 makes it so the 'range' variable of an iterating
    // expansion statement need no longer be 'constexpr'... so do we want
    // lifetime extension for iterating expansion statements after all?
    return BuildCXXEnumeratingExpansionStmtPattern(ESD, Init, DS, LParenLoc,
                                                   ColonLoc, RParenLoc);
  }

  if (ExpansionInitializer->hasPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(ExpansionInitializer);
    if (R.isInvalid())
      return StmtError();
    ExpansionInitializer = R.get();
  }

  if (DiagnoseUnexpandedParameterPack(ExpansionInitializer))
    return StmtError();

  return BuildNonEnumeratingCXXExpansionStmtPattern(
      ESD, Init, DS, ExpansionInitializer, LParenLoc, ColonLoc, RParenLoc,
      LifetimeExtendTemps);
}

StmtResult Sema::BuildCXXEnumeratingExpansionStmtPattern(
    Decl *ESD, Stmt *Init, Stmt *ExpansionVar, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc) {
  return CXXExpansionStmtPattern::CreateEnumerating(
      Context, cast<CXXExpansionStmtDecl>(ESD), Init,
      cast<DeclStmt>(ExpansionVar), LParenLoc, ColonLoc, RParenLoc);
}

StmtResult Sema::BuildNonEnumeratingCXXExpansionStmtPattern(
    CXXExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVarStmt,
    Expr *ExpansionInitializer, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  VarDecl *ExpansionVar = cast<VarDecl>(ExpansionVarStmt->getSingleDecl());

  // Reject lambdas early.
  if (auto *RD = ExpansionInitializer->getType()->getAsCXXRecordDecl();
      RD && RD->isLambda()) {
    Diag(ExpansionInitializer->getBeginLoc(), diag::err_expansion_stmt_lambda);
    return StmtError();
  }

  if (ExpansionInitializer->isTypeDependent()) {
    ActOnDependentForRangeInitializer(ExpansionVar, BFRK_Build);
    return CXXExpansionStmtPattern::CreateDependent(
        Context, ESD, Init, ExpansionVarStmt, ExpansionInitializer, LParenLoc,
        ColonLoc, RParenLoc);
  }

  if (RequireCompleteType(ExpansionInitializer->getExprLoc(),
                          ExpansionInitializer->getType(),
                          diag::err_expansion_stmt_incomplete))
    return StmtError();

  if (ExpansionInitializer->getType()->isVariableArrayType()) {
    Diag(ExpansionInitializer->getExprLoc(), diag::err_expansion_stmt_vla)
        << ExpansionInitializer->getType();
    return StmtError();
  }

  // Otherwise, if it can be an iterating expansion statement, it is one.
  DeclRefExpr *Index = BuildIndexDRE(*this, ESD);
  IterableExpansionStmtData Data = TryBuildIterableExpansionStmtInitializer(
      *this, ExpansionInitializer, Index, ColonLoc,
      ExpansionVar->isConstexpr());
  if (Data.hasError()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  if (Data.isIterable()) {
    if (FinaliseExpansionVar(*this, ExpansionVar, Data.Initializer))
      return StmtError();

    return CXXExpansionStmtPattern::CreateIterating(
        Context, ESD, Init, ExpansionVarStmt, Data.RangeDecl, Data.BeginDecl,
        Data.EndDecl, LParenLoc, ColonLoc, RParenLoc);
  }

  // If not, try destructuring.
  StmtResult DecompDeclStmt = BuildDestructuringDecompositionDecl(
      *this, ExpansionInitializer, ColonLoc, ExpansionVar->isConstexpr(),
      LifetimeExtendTemps);
  if (DecompDeclStmt.isInvalid()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  auto *DS = DecompDeclStmt.getAs<DeclStmt>();
  auto *DD = cast<DecompositionDecl>(DS->getSingleDecl());
  if (DD->isInvalidDecl())
    return StmtError();

  // Synthesise an InitListExpr to store the bindings; this essentially lets us
  // desugar the expansion of a destructuring expansion statement to that of an
  // enumerating expansion statement.
  SmallVector<Expr *> Bindings;
  for (BindingDecl *BD : DD->bindings()) {
    auto *HVD = BD->getHoldingVar();
    Bindings.push_back(HVD ? HVD->getInit() : BD->getBinding());
  }

  ExprResult Select = BuildCXXExpansionSelectExpr(
      new (Context) InitListExpr(Context, ColonLoc, Bindings, ColonLoc),
      Index);

  if (Select.isInvalid()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  if (FinaliseExpansionVar(*this, ExpansionVar, Select))
    return StmtError();

  return CXXExpansionStmtPattern::CreateDestructuring(
      Context, ESD, Init, ExpansionVarStmt, DS, LParenLoc, ColonLoc, RParenLoc);
}

StmtResult Sema::FinishCXXExpansionStmt(Stmt *Exp, Stmt *Body) {
  if (!Exp || !Body)
    return StmtError();

  auto *Expansion = cast<CXXExpansionStmtPattern>(Exp);
  assert(!Expansion->getDecl()->getInstantiations() &&
         "should not rebuild expansion statement after instantiation");

  Expansion->setBody(Body);
  if (HasDependentSize(Expansion))
    return Expansion;

  // Now that we're expanding this, exit the context of the expansion stmt
  // so that we no longer treat this as dependent.
  ContextRAII CtxGuard(*this, CurContext->getParent(),
                       /*NewThis=*/false);

  // Even if the size isn't technically dependent, delay expansion until
  // we're no longer in a template if this is an iterating expansion statement
  // since evaluating a lambda declared in a template doesn't work too well.
  if (CurContext->isDependentContext() && Expansion->isIterating())
    return Expansion;

  // This can fail if this is an iterating expansion statement.
  std::optional<uint64_t> NumInstantiations = ComputeExpansionSize(Expansion);
  if (!NumInstantiations)
    return StmtError();

  if (CheckExpansionSize(*this, *NumInstantiations, Expansion->getColonLoc()))
    return StmtError();

  // Collect shared statements.
  SmallVector<Stmt *, 1> Shared;
  if (Expansion->getInit())
    Shared.push_back(Expansion->getInit());

  if (Expansion->isIterating()) {
    Shared.push_back(Expansion->getRangeVarStmt());
    Shared.push_back(Expansion->getBeginVarStmt());
    Shared.push_back(Expansion->getEndVarStmt());
  } else if (Expansion->isDestructuring()) {
    Shared.push_back(Expansion->getDecompositionDeclStmt());
    MarkAnyDeclReferenced(Exp->getBeginLoc(), Expansion->getDecompositionDecl(),
                          true);
  }

  // Return an empty statement if the range is empty.
  if (*NumInstantiations == 0) {
    Expansion->getDecl()->setInstantiations(
        CXXExpansionStmtInstantiation::Create(
            Context, Expansion->getBeginLoc(), Expansion->getEndLoc(),
            /*Instantiations=*/{}, Shared, Expansion->isDestructuring()));
    return Expansion;
  }

  // Create a compound statement binding the expansion variable and body.
  Stmt *VarAndBody[] = {Expansion->getExpansionVarStmt(), Body};
  Stmt *CombinedBody =
      CompoundStmt::Create(Context, VarAndBody, FPOptionsOverride(),
                           Body->getBeginLoc(), Body->getEndLoc());

  // Expand the body for each instantiation.
  SmallVector<Stmt *, 4> Instantiations;
  CXXExpansionStmtDecl *ESD = Expansion->getDecl();
  for (uint64_t I = 0; I < *NumInstantiations; ++I) {
    TemplateArgument Arg{Context, llvm::APSInt::get(I),
                         Context.getPointerDiffType()};
    MultiLevelTemplateArgumentList MTArgList(ESD, Arg, true);
    MTArgList.addOuterRetainedLevels(
        Expansion->getDecl()->getIndexTemplateParm()->getDepth());

    LocalInstantiationScope LIScope(*this, /*CombineWithOuterScope=*/true);
    NonSFINAEContext _(*this);
    InstantiatingTemplate Inst(*this, Body->getBeginLoc(), Expansion, Arg,
                               Body->getSourceRange());

    StmtResult Instantiation = SubstStmt(CombinedBody, MTArgList);
    if (Instantiation.isInvalid())
      return StmtError();
    Instantiations.push_back(Instantiation.get());
  }

  auto *InstantiationsStmt = CXXExpansionStmtInstantiation::Create(
      Context, Expansion->getBeginLoc(), Expansion->getEndLoc(), Instantiations,
      Shared, Expansion->isDestructuring());

  Expansion->getDecl()->setInstantiations(InstantiationsStmt);
  return Expansion;
}

ExprResult Sema::BuildCXXExpansionSelectExpr(InitListExpr *Range, Expr *Idx) {
  if (Idx->isValueDependent() || InitListContainsPack(Range))
    return new (Context) CXXExpansionSelectExpr(Context, Range, Idx);

  // The index is a DRE to a template parameter; we should never
  // fail to evaluate it.
  Expr::EvalResult ER;
  if (!Idx->EvaluateAsInt(ER, Context))
    llvm_unreachable("Failed to evaluate expansion index");

  uint64_t I = ER.Val.getInt().getZExtValue();
  return Range->getInit(I);
}

std::optional<uint64_t>
Sema::ComputeExpansionSize(CXXExpansionStmtPattern *Expansion) {
  assert(!HasDependentSize(Expansion));

  if (Expansion->isEnumerating())
    return cast<CXXExpansionSelectExpr>(
               Expansion->getExpansionVariable()->getInit())
        ->getRangeExpr()
        ->getNumInits();

  // CWG 3131: N is the result of evaluating the expression
  //
  // [&] consteval {
  //    std::ptrdiff_t result = 0;
  //    auto b = begin-expr;
  //    auto e = end-expr;
  //    for (; b != e; ++b) ++result;
  //    return result;
  // }()
  if (Expansion->isIterating()) {
    SourceLocation Loc = Expansion->getColonLoc();
    EnterExpressionEvaluationContext ExprEvalCtx(
        *this, ExpressionEvaluationContext::ConstantEvaluated);

    // This is mostly copied from ParseLambdaExpressionAfterIntroducer().
    ParseScope LambdaScope(*this, Scope::LambdaScope | Scope::DeclScope |
                                      Scope::FunctionDeclarationScope |
                                      Scope::FunctionPrototypeScope);
    AttributeFactory AttrFactory;
    LambdaIntroducer Intro;
    Intro.Range = SourceRange(Loc, Loc);
    Intro.Default = LCD_ByRef; // CWG 3131
    Intro.DefaultLoc = Loc;
    DeclSpec DS(AttrFactory);
    Declarator D(DS, ParsedAttributesView::none(),
                 DeclaratorContext::LambdaExpr);
    PushLambdaScope();
    ActOnLambdaExpressionAfterIntroducer(Intro, getCurScope());

    // Make the lambda 'consteval'.
    {
      ParseScope Prototype(*this, Scope::FunctionPrototypeScope |
                                      Scope::FunctionDeclarationScope |
                                      Scope::DeclScope);
      const char *PrevSpec = nullptr;
      unsigned DiagId = 0;
      DS.SetConstexprSpec(ConstexprSpecKind::Consteval, Loc, PrevSpec, DiagId);
      assert(DiagId == 0 && PrevSpec == nullptr);
      ActOnLambdaClosureParameters(getCurScope(), /*ParamInfo=*/{});
      ActOnLambdaClosureQualifiers(Intro, /*MutableLoc=*/SourceLocation());
    }

    ParseScope BodyScope(*this, Scope::BlockScope | Scope::FnScope |
                                    Scope::DeclScope |
                                    Scope::CompoundStmtScope);

    ActOnStartOfLambdaDefinition(Intro, D, DS);

    // Enter the compound statement that is the lambda body.
    ActOnStartOfCompoundStmt(/*IsStmtExpr=*/false);
    ActOnAfterCompoundStatementLeadingPragmas();
    auto PopScopesOnReturn = llvm::make_scope_exit([&] {
      ActOnFinishOfCompoundStmt();
      ActOnLambdaError(Loc, getCurScope());
    });

    // std::ptrdiff_t result = 0;
    QualType PtrDiffT = Context.getPointerDiffType();
    VarDecl *ResultVar = VarDecl::Create(
        Context, CurContext, Loc, Loc, &PP.getIdentifierTable().get("__result"),
        PtrDiffT, Context.getTrivialTypeSourceInfo(PtrDiffT, Loc), SC_None);
    Expr *Zero = ActOnIntegerConstant(Loc, 0).get();
    AddInitializerToDecl(ResultVar, Zero, false);
    StmtResult ResultVarStmt =
        ActOnDeclStmt(ConvertDeclToDeclGroup(ResultVar), Loc, Loc);
    if (ResultVarStmt.isInvalid() || ResultVar->isInvalidDecl())
      return std::nullopt;

    // Start the for loop.
    ParseScope ForScope(*this, Scope::DeclScope | Scope::ControlScope);

    // auto b = begin-expr;
    // auto e = end-expr;
    ForRangeBeginEndInfo Info = BuildCXXForRangeBeginEndVars(
        getCurScope(), Expansion->getRangeVar(), Loc,
        /*CoawaitLoc=*/{},
        /*LifetimeExtendTemps=*/{}, BFRK_Build, /*Constexpr=*/false);
    if (!Info.isValid())
      return std::nullopt;

    StmtResult BeginStmt =
        ActOnDeclStmt(ConvertDeclToDeclGroup(Info.BeginVar), Loc, Loc);
    StmtResult EndStmt =
        ActOnDeclStmt(ConvertDeclToDeclGroup(Info.EndVar), Loc, Loc);
    if (BeginStmt.isInvalid() || EndStmt.isInvalid())
      return std::nullopt;

    // b != e
    auto GetDeclRef = [&](VarDecl *VD) -> DeclRefExpr * {
      return BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(),
                              VK_LValue, Loc);
    };

    DeclRefExpr *Begin = GetDeclRef(Info.BeginVar);
    DeclRefExpr *End = GetDeclRef(Info.EndVar);
    ExprResult NotEqual =
        ActOnBinOp(getCurScope(), Loc, tok::exclaimequal, Begin, End);
    if (NotEqual.isInvalid())
      return std::nullopt;
    ConditionResult Condition = ActOnCondition(
        getCurScope(), Loc, NotEqual.get(), ConditionKind::Boolean,
        /*MissingOk=*/false);
    if (Condition.isInvalid())
      return std::nullopt;

    // ++b
    Begin = GetDeclRef(Info.BeginVar);
    ExprResult Increment =
        ActOnUnaryOp(getCurScope(), Loc, tok::plusplus, Begin);
    if (Increment.isInvalid())
      return std::nullopt;
    FullExprArg ThirdPart = MakeFullDiscardedValueExpr(Increment.get());

    // Enter the body of the for loop.
    ParseScope InnerScope(*this, Scope::DeclScope);
    getCurScope()->decrementMSManglingNumber();

    // ++result;
    DeclRefExpr *ResultDeclRef = BuildDeclRefExpr(
        ResultVar, ResultVar->getType().getNonReferenceType(), VK_LValue, Loc);
    ExprResult IncrementResult =
        ActOnUnaryOp(getCurScope(), Loc, tok::plusplus, ResultDeclRef);
    if (IncrementResult.isInvalid())
      return std::nullopt;
    StmtResult IncrementStmt = ActOnExprStmt(IncrementResult.get());
    if (IncrementStmt.isInvalid())
      return std::nullopt;

    // Exit the for loop.
    InnerScope.Exit();
    ForScope.Exit();
    StmtResult ForLoop = ActOnForStmt(Loc, Loc, /*First=*/nullptr, Condition,
                                      ThirdPart, Loc, IncrementStmt.get());
    if (ForLoop.isInvalid())
      return std::nullopt;

    // return result;
    ResultDeclRef = BuildDeclRefExpr(
        ResultVar, ResultVar->getType().getNonReferenceType(), VK_LValue, Loc);
    StmtResult Return = ActOnReturnStmt(Loc, ResultDeclRef, getCurScope());
    if (Return.isInvalid())
      return std::nullopt;

    // Finally, we can build the compound statement that is the lambda body.
    StmtResult LambdaBody =
        ActOnCompoundStmt(Loc, Loc,
                          {ResultVarStmt.get(), BeginStmt.get(), EndStmt.get(),
                           ForLoop.get(), Return.get()},
                          /*isStmtExpr=*/false);
    if (LambdaBody.isInvalid())
      return std::nullopt;

    ActOnFinishOfCompoundStmt();
    BodyScope.Exit();
    LambdaScope.Exit();
    PopScopesOnReturn.release();
    ExprResult Lambda = ActOnLambdaExpr(Loc, LambdaBody.get());
    if (Lambda.isInvalid())
      return std::nullopt;

    // Invoke the lambda.
    ExprResult Call =
        ActOnCallExpr(getCurScope(), Lambda.get(), Loc, /*ArgExprs=*/{}, Loc);
    if (Call.isInvalid())
      return std::nullopt;

    Expr::EvalResult ER;
    SmallVector<PartialDiagnosticAt, 4> Notes;
    ER.Diag = &Notes;
    if (!Call.get()->EvaluateAsInt(ER, Context)) {
      Diag(Loc, diag::err_expansion_size_expr_not_ice);
      for (const auto &[Location, PDiag] : Notes)
        Diag(Location, PDiag);
      return std::nullopt;
    }

    // It shouldn't be possible for this to be negative since we compute this
    // via the built-in '++' on a ptrdiff_t.
    assert(ER.Val.getInt().isNonNegative());
    return ER.Val.getInt().getZExtValue();
  }

  assert(Expansion->isDestructuring());
  return Expansion->getDecompositionDecl()->bindings().size();
}
