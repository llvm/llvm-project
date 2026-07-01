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

namespace {
struct IterableExpansionStmtData {
  enum class State {
    NotIterable,
    Error,
    Ok,
  };

  DeclStmt *RangeDecl = nullptr;
  DeclStmt *BeginDecl = nullptr;
  DeclStmt *IterDecl = nullptr;
  State TheState = State::NotIterable;

  bool isIterable() const { return TheState == State::Ok; }
  bool hasError() { return TheState == State::Error; }
};
} // namespace

// Build a 'DeclRefExpr' designating the template parameter that is used as
// the expansion index
static DeclRefExpr *BuildIndexDRE(Sema &S, CXXExpansionStmtDecl *ESD) {
  return S.BuildDeclRefExpr(ESD->getIndexTemplateParm(),
                            S.Context.getPointerDiffType(), VK_PRValue,
                            ESD->getBeginLoc());
}

static bool FinalizeExpansionVar(Sema &S, VarDecl *ExpansionVar,
                                 ExprResult Initializer) {
  if (Initializer.isInvalid()) {
    S.ActOnInitializerError(ExpansionVar);
    return true;
  }

  S.AddInitializerToDecl(ExpansionVar, Initializer.get(), /*DirectInit=*/false);
  return ExpansionVar->isInvalidDecl();
}

static auto InitListContainsPack(const InitListExpr *ILE) {
  return llvm::any_of(ILE->inits(),
                      [](const Expr *E) { return isa<PackExpansionExpr>(E); });
}

static bool HasDependentSize(const DeclContext *CurContext,
                             const CXXExpansionStmtPattern *Pattern) {
  switch (Pattern->getKind()) {
  case CXXExpansionStmtPattern::ExpansionStmtKind::Enumerating: {
    auto *SelectExpr = cast<CXXExpansionSelectExpr>(
        Pattern->getExpansionVariable()->getInit());
    return InitListContainsPack(SelectExpr->getRangeExpr());
  }

  case CXXExpansionStmtPattern::ExpansionStmtKind::Iterating:
    // Even if the size isn't technically dependent, delay expansion until
    // we're no longer in a template since evaluating a lambda declared in
    // a template doesn't work too well.
    assert(CurContext->isExpansionStmt());
    return CurContext->getParent()->isDependentContext();

  case CXXExpansionStmtPattern::ExpansionStmtKind::Dependent:
    return true;

  case CXXExpansionStmtPattern::ExpansionStmtKind::Destructuring:
    llvm_unreachable("TODO");
  }

  llvm_unreachable("invalid pattern kind");
}

static IterableExpansionStmtData TryBuildIterableExpansionStmtInitializer(
    Sema &S, Expr *ExpansionInitializer, Expr *Index, SourceLocation ColonLoc,
    bool VarIsConstexpr,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
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

  bool FoundBeginEnd = false;
  if (auto *Record = Ty->getAsCXXRecordDecl()) {
    LookupResult BeginLR(S, BeginName, Sema::LookupMemberName);
    LookupResult EndLR(S, EndName, Sema::LookupMemberName);
    FoundBeginEnd = S.LookupQualifiedName(BeginLR, Record) &&
                    S.LookupQualifiedName(EndLR, Record);
  }

  // If member lookup doesn't yield anything, try ADL.
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
  Sema::ContextRAII CtxGuard(S, S.CurContext->getParent(),
                             /*NewThis=*/false);

  // We know that this is supposed to be an iterable expansion statement;
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
      S.Context.getAutoType(DeducedKind::Undeduced, QualType(),
                            AutoTypeKeyword::DecltypeAuto),
      VarIsConstexpr);
  if (Var.isInvalid())
    return Data;

  // CWG 3140: 'range', 'begin', and 'iter' are 'constexpr' iff the
  // for-range-declaration is declared 'constexpr'.
  //
  // FIXME: As of CWG 3140, we should only create 'begin' here, and not 'end',
  // but that requires another substantial refactor of the for-range code.
  auto *RangeVar = cast<DeclStmt>(Var.get());
  Sema::ForRangeBeginEndInfo Info = S.BuildCXXForRangeBeginEndVars(
      Scope, cast<VarDecl>(RangeVar->getSingleDecl()), ColonLoc,
      /*CoawaitLoc=*/{},
      /*LifetimeExtendTemps=*/{}, Sema::BFRK_Build, VarIsConstexpr);

  if (!Info.isValid())
    return Data;

  // CWG 3140: At runtime, we only need to evaluate 'begin', whereas 'end' is
  // only used at compile-time; we'll rebuild it when we compute the expansion
  // size, so only build 'begin' here.
  StmtResult BeginStmt = S.ActOnDeclStmt(
      S.ConvertDeclToDeclGroup(Info.BeginVar), ColonLoc, ColonLoc);
  if (BeginStmt.isInvalid())
    return Data;

  // TODO: Build 'constexpr auto iter = begin + decltype(begin - begin){i};'.
  S.Diag(ColonLoc, diag::err_iterating_expansion_stmt_unsupported);
  return Data;

#if 0 // This will be used once we support iterating expansion statements.
  // Store it in a variable.
  // See also Sema::BuildCXXForRangeBeginEndVars().
  const auto DepthStr = std::to_string(Scope->getDepth() / 2);
  IdentifierInfo *Name =
      S.PP.getIdentifierInfo(std::string("__iter") + DepthStr);
  VarDecl *IterVar = S.BuildForRangeVarDecl(
      ColonLoc, S.Context.getAutoDeductType(), Name, VarIsConstexpr);
  S.AddInitializerToDecl(IterVar, BeginPlusI.get(), /*DirectInit=*/false);
  if (IterVar->isInvalidDecl())
    return Data;

  StmtResult IterVarStmt =
      S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(IterVar), ColonLoc, ColonLoc);
  if (IterVarStmt.isInvalid())
    return Data;

  // CWG 3149: Apply lifetime extension to iterating expansion statements.
  S.ApplyForRangeOrExpansionStatementLifetimeExtension(
      cast<VarDecl>(RangeVar->getSingleDecl()), LifetimeExtendTemps);

  Data.BeginDecl = BeginStmt.getAs<DeclStmt>();
  Data.RangeDecl = RangeVar;
  Data.IterDecl = IterVarStmt.getAs<DeclStmt>();
  Data.TheState = IterableExpansionStmtData::State::Ok;
  return Data;
#endif
}

CXXExpansionStmtDecl *
Sema::ActOnCXXExpansionStmtDecl(unsigned TemplateDepth,
                                SourceLocation TemplateKWLoc) {
  // Create a template parameter. This will be used to denote the index
  // of the element that we're instantiating. CWG 3044 requires this type to
  // be 'ptrdiff_t' for iterating expansion statements, so use that in all
  // cases.
  QualType ParmTy = Context.getPointerDiffType();
  TypeSourceInfo *ParmTI =
      Context.getTrivialTypeSourceInfo(ParmTy, TemplateKWLoc);

  auto *TParam = NonTypeTemplateParmDecl::Create(
      Context, Context.getTranslationUnitDecl(), TemplateKWLoc, TemplateKWLoc,
      TemplateDepth, /*Position=*/0, /*Id=*/nullptr, ParmTy, /*ParameterPack=*/false,
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
    if (FinalizeExpansionVar(*this, ExpansionVar, Initializer))
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
      *this, ExpansionInitializer, Index, ColonLoc, ExpansionVar->isConstexpr(),
      LifetimeExtendTemps);
  if (Data.hasError()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  if (Data.isIterable()) {
    // Build '*iter'.
    auto *IterVar = cast<VarDecl>(Data.IterDecl->getSingleDecl());
    DeclRefExpr *IterDRE = BuildDeclRefExpr(
        IterVar, IterVar->getType().getNonReferenceType(), VK_LValue, ColonLoc);
    ExprResult Deref =
        ActOnUnaryOp(getCurScope(), ColonLoc, tok::star, IterDRE);
    if (Deref.isInvalid()) {
      ActOnInitializerError(ExpansionVar);
      return StmtError();
    }

    Deref = MaybeCreateExprWithCleanups(Deref.get());

    if (FinalizeExpansionVar(*this, ExpansionVar, Deref.get()))
      return StmtError();

    return CXXExpansionStmtPattern::CreateIterating(
        Context, ESD, Init, ExpansionVarStmt, Data.RangeDecl, Data.BeginDecl,
        Data.IterDecl, LParenLoc, ColonLoc, RParenLoc);
  }

  Diag(ESD->getLocation(), diag::err_expansion_statements_todo);
  return StmtError();
}

StmtResult Sema::FinishCXXExpansionStmt(Stmt *Exp, Stmt *Body) {
  if (!Exp || !Body)
    return StmtError();

  auto *Expansion = cast<CXXExpansionStmtPattern>(Exp);
  assert(!Expansion->getDecl()->getInstantiations() &&
         "should not rebuild expansion statement after instantiation");

  Expansion->setBody(Body);
  if (HasDependentSize(CurContext, Expansion))
    return Expansion;

  // Now that we're expanding this, exit the context of the expansion stmt
  // so that we no longer treat this as dependent.
  ContextRAII CtxGuard(*this, CurContext->getParent(),
                       /*NewThis=*/false);

  // This can fail if this is an iterating expansion statement.
  std::optional<uint64_t> NumInstantiations = ComputeExpansionSize(Expansion);
  if (!NumInstantiations)
    return StmtError();

  // Collect preamble statements.
  //
  // There are at most 3 of these: for iterating expansion statements, these
  // consist of the '__range' and '__begin' variables, and for destructuring
  // expansion statements of the DecompositionDecl whose initializer we're
  // expanding. Finally, any expansion statement may have an init-statement
  // as well.
  SmallVector<Stmt *, 3> Preamble;
  if (Expansion->getInit())
    Preamble.push_back(Expansion->getInit());

  if (Expansion->isIterating()) {
    Preamble.push_back(Expansion->getRangeVarStmt());
    Preamble.push_back(Expansion->getBeginVarStmt());
  } else {
    assert(Expansion->isEnumerating() && "TODO");
  }

  // Return an empty statement if the range is empty.
  if (*NumInstantiations == 0) {
    Expansion->getDecl()->setInstantiations(
        CXXExpansionStmtInstantiation::Create(Context, Expansion->getDecl(),
                                              /*Instantiations=*/{}, Preamble,
                                              Expansion->isDestructuring()));
    return Expansion;
  }

  // Create a compound statement binding the expansion variable and body,
  // as well as the 'iter' variable if this is an iterating expansion statement.
  SmallVector<Stmt *, 3> StmtsToInstantiate;
  if (Expansion->isIterating())
    StmtsToInstantiate.push_back(Expansion->getIterVarStmt());
  StmtsToInstantiate.push_back(Expansion->getExpansionVarStmt());
  StmtsToInstantiate.push_back(Body);
  Stmt *CombinedBody =
      CompoundStmt::Create(Context, StmtsToInstantiate, FPOptionsOverride(),
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
      Context, Expansion->getDecl(), Instantiations, Preamble,
      Expansion->isDestructuring() || Expansion->isIterating());

  Expansion->getDecl()->setInstantiations(InstantiationsStmt);
  return Expansion;
}

ExprResult Sema::BuildCXXExpansionSelectExpr(InitListExpr *Range, Expr *Idx) {
  if (Idx->isValueDependent() || InitListContainsPack(Range))
    return new (Context) CXXExpansionSelectExpr(Context, Range, Idx);

  // The index is a DRE to a template parameter; we should never
  // fail to evaluate it.
  uint64_t I = Idx->EvaluateKnownConstInt(Context).getZExtValue();
  return Range->getInit(I);
}

std::optional<uint64_t>
Sema::ComputeExpansionSize(CXXExpansionStmtPattern *Expansion) {
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

    // TODO: Build the lambda and evaluate it.
    Diag(Loc, diag::err_iterating_expansion_stmt_unsupported);
    return std::nullopt;

#if 0 // This will be used once we support iterating expansion statements.
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
#endif
  }

  llvm_unreachable("TODO");
}
