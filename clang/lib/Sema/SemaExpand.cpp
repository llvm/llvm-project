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

  // TODO: CWG 3131 changes how this range is declared.
  StmtResult Var = S.BuildCXXForRangeRangeVar(Scope, ExpansionInitializer,
                                              /*ForExpansionStmt=*/true);
  if (Var.isInvalid())
    return Data;

  auto *RangeVar = cast<DeclStmt>(Var.get());
  Sema::ForRangeBeginEndInfo Info = S.BuildCXXForRangeBeginEndVars(
      Scope, cast<VarDecl>(RangeVar->getSingleDecl()), ColonLoc,
      /*CoawaitLoc=*/{},
      /*LifetimeExtendTemps=*/{}, Sema::BFRK_Build, /*ForExpansionStmt=*/true);

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

static StmtResult BuildDestructuringCXXExpansionStmt(
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
  auto *TParamList = TemplateParameterList::Create(
      Context, TemplateKWLoc, TemplateKWLoc, {NTTP}, TemplateKWLoc,
      /*RequiresClause=*/nullptr);
  auto *Result =
      CXXExpansionStmtDecl::Create(Context, Ctx, TemplateKWLoc, TParamList);
  Ctx->addDecl(Result);
  return Result;
}

ExprResult Sema::ActOnCXXExpansionInitList(MultiExprArg SubExprs,
                                           SourceLocation LBraceLoc,
                                           SourceLocation RBraceLoc) {
  return CXXExpansionInitListExpr::Create(Context, SubExprs, LBraceLoc,
                                          RBraceLoc);
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
  if (auto *ILE = dyn_cast<CXXExpansionInitListExpr>(ExpansionInitializer)) {
    ExprResult Initializer =
        BuildCXXExpansionInitListSelectExpr(ILE, BuildIndexDRE(*this, ESD));
    if (FinaliseExpansionVar(*this, ExpansionVar, Initializer))
      return StmtError();

    // Note that lifetime extension only applies to destructuring expansion
    // statements, so we just ignore 'LifetimeExtendedTemps' entirely for other
    // types of expansion statements (this is CWG 3043).
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

  // Reject lambdas early.
  if (auto *RD = ExpansionInitializer->getType()->getAsCXXRecordDecl();
      RD && RD->isLambda()) {
    Diag(ExpansionInitializer->getBeginLoc(), diag::err_expansion_stmt_lambda);
    return StmtError();
  }

  return BuildNonEnumeratingCXXExpansionStmtPattern(
      ESD, Init, DS, ExpansionInitializer, LParenLoc, ColonLoc, RParenLoc,
      LifetimeExtendTemps);
}

StmtResult Sema::BuildCXXEnumeratingExpansionStmtPattern(
    Decl *ESD, Stmt *Init, Stmt *ExpansionVar, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc) {
  return new (Context) CXXEnumeratingExpansionStmtPattern(
      cast<CXXExpansionStmtDecl>(ESD), Init, cast<DeclStmt>(ExpansionVar),
      LParenLoc, ColonLoc, RParenLoc);
}

StmtResult Sema::BuildNonEnumeratingCXXExpansionStmtPattern(
    CXXExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVarStmt,
    Expr *ExpansionInitializer, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  VarDecl *ExpansionVar = cast<VarDecl>(ExpansionVarStmt->getSingleDecl());

  if (ExpansionInitializer->isTypeDependent()) {
    ActOnDependentForRangeInitializer(ExpansionVar, BFRK_Build);
    return new (Context) CXXDependentExpansionStmtPattern(
        ESD, Init, ExpansionVarStmt, ExpansionInitializer, LParenLoc, ColonLoc,
        RParenLoc);
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

    return new (Context) CXXIteratingExpansionStmtPattern(
        ESD, Init, ExpansionVarStmt, Data.RangeDecl, Data.BeginDecl,
        Data.EndDecl, LParenLoc, ColonLoc, RParenLoc);
  }

  // If not, try destructuring.
  StmtResult DecompDeclStmt = BuildDestructuringCXXExpansionStmt(
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

  ExprResult Select = BuildCXXDestructuringExpansionSelectExpr(DD, Index);
  if (Select.isInvalid()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  if (FinaliseExpansionVar(*this, ExpansionVar, Select))
    return StmtError();

  return new (Context) CXXDestructuringExpansionStmtPattern(
      ESD, Init, ExpansionVarStmt, DS, LParenLoc, ColonLoc, RParenLoc);
}

StmtResult Sema::FinishCXXExpansionStmt(Stmt *Exp, Stmt *Body) {
  if (!Exp || !Body)
    return StmtError();

  auto *Expansion = cast<CXXExpansionStmtPattern>(Exp);
  assert(!Expansion->getDecl()->getInstantiations() &&
         "should not rebuild expansion statement after instantiation");

  Expansion->setBody(Body);
  if (Expansion->hasDependentSize())
    return Expansion;

  // This can fail if this is an iterating expansion statement.
  std::optional<uint64_t> NumInstantiations = ComputeExpansionSize(Expansion);
  if (!NumInstantiations)
    return StmtError();

  // Collect shared statements.
  SmallVector<Stmt *, 1> Shared;
  if (Expansion->getInit())
    Shared.push_back(Expansion->getInit());

  if (auto *Iter = dyn_cast<CXXIteratingExpansionStmtPattern>(Expansion)) {
    Shared.push_back(Iter->getRangeVarStmt());
    Shared.push_back(Iter->getBeginVarStmt());
    Shared.push_back(Iter->getEndVarStmt());
  } else if (auto *Destructuring =
                 dyn_cast<CXXDestructuringExpansionStmtPattern>(Expansion)) {
    Shared.push_back(Destructuring->getDecompositionDeclStmt());
  }

  // Return an empty statement if the range is empty.
  if (*NumInstantiations == 0) {
    Expansion->getDecl()->setInstantiations(
        CXXExpansionStmtInstantiation::Create(
            Context, Expansion->getBeginLoc(), Expansion->getEndLoc(),
            /*Instantiations=*/{}, Shared,
            isa<CXXDestructuringExpansionStmtPattern>(Expansion)));
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
    // Now that we're expanding this, exit the context of the expansion stmt
    // so that we no longer treat this as dependent.
    ContextRAII CtxGuard(*this, CurContext->getParent(),
                         /*NewThis=*/false);

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
      Shared, isa<CXXDestructuringExpansionStmtPattern>(Expansion));

  Expansion->getDecl()->setInstantiations(InstantiationsStmt);
  return Expansion;
}

ExprResult
Sema::BuildCXXExpansionInitListSelectExpr(CXXExpansionInitListExpr *Range,
                                          Expr *Idx) {
  if (Range->containsPackExpansion() || Idx->isValueDependent())
    return new (Context) CXXExpansionInitListSelectExpr(Context, Range, Idx);

  // The index is a DRE to a template parameter; we should never
  // fail to evaluate it.
  Expr::EvalResult ER;
  if (!Idx->EvaluateAsInt(ER, Context))
    llvm_unreachable("Failed to evaluate expansion index");

  uint64_t I = ER.Val.getInt().getZExtValue();
  return Range->getExprs()[I];
}

ExprResult Sema::BuildCXXDestructuringExpansionSelectExpr(DecompositionDecl *DD,
                                                          Expr *Idx) {
  if (Idx->isValueDependent())
    return new (Context) CXXDestructuringExpansionSelectExpr(Context, DD, Idx);

  Expr::EvalResult ER;
  if (!Idx->EvaluateAsInt(ER, Context))
    llvm_unreachable("Failed to evaluate expansion index");

  uint64_t I = ER.Val.getInt().getZExtValue();
  MarkAnyDeclReferenced(Idx->getBeginLoc(), DD, true);
  if (auto *BD = DD->bindings()[I]; auto *HVD = BD->getHoldingVar())
    return HVD->getInit();
  else
    return BD->getBinding();
}

std::optional<uint64_t>
Sema::ComputeExpansionSize(CXXExpansionStmtPattern *Expansion) {
  assert(!Expansion->hasDependentSize());

  if (isa<CXXEnumeratingExpansionStmtPattern>(Expansion))
    return cast<CXXExpansionInitListSelectExpr>(
               Expansion->getExpansionVariable()->getInit())
        ->getRangeExpr()
        ->getExprs()
        .size();

  // By [stmt.expand]5.2, N is the result of evaluating the expression
  //
  // [] consteval {
  //    std::ptrdiff_t result = 0;
  //    for (auto i = begin; i != end; ++i) ++result;
  //    return result;
  // }()
  // TODO: CWG 3131 changes this lambda a bit.
  if (auto *Iterating = dyn_cast<CXXIteratingExpansionStmtPattern>(Expansion)) {
    EnterExpressionEvaluationContext ExprEvalCtx(
        *this, ExpressionEvaluationContext::ConstantEvaluated);

    // FIXME: Actually do that; unfortunately, conjuring a lambda out of thin
    // air in Sema is a massive pain, so for now just cheat by computing
    // 'end - begin'.
    SourceLocation Loc = Iterating->getColonLoc();
    DeclRefExpr *Begin = BuildDeclRefExpr(
        Iterating->getBeginVar(),
        Iterating->getBeginVar()->getType().getNonReferenceType(), VK_LValue,
        Loc);

    DeclRefExpr *End = BuildDeclRefExpr(
        Iterating->getEndVar(),
        Iterating->getEndVar()->getType().getNonReferenceType(), VK_LValue,
        Loc);

    ExprResult N = ActOnBinOp(getCurScope(), Loc, tok::minus, End, Begin);
    if (N.isInvalid())
      return std::nullopt;

    Expr::EvalResult ER;
    SmallVector<PartialDiagnosticAt, 4> Notes;
    ER.Diag = &Notes;
    if (!N.get()->EvaluateAsInt(ER, Context)) {
      Diag(Loc, diag::err_expansion_size_expr_not_ice);
      for (const auto &[Location, PDiag] : Notes)
        Diag(Location, PDiag);
      return std::nullopt;
    }

    if (ER.Val.getInt().isNegative()) {
      Diag(Loc, diag::err_expansion_size_negative) << ER.Val.getInt();
      return std::nullopt;
    }

    return ER.Val.getInt().getZExtValue();
  }

  if (auto *Destructuring =
          dyn_cast<CXXDestructuringExpansionStmtPattern>(Expansion))
    return Destructuring->getDecompositionDecl()->bindings().size();

  llvm_unreachable("Invalid expansion statement class");
}
