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

  Diag(ESD->getLocation(), diag::err_expansion_statements_todo);
  return StmtError();
}

StmtResult Sema::BuildCXXEnumeratingExpansionStmtPattern(
    Decl *ESD, Stmt *Init, Stmt *ExpansionVar, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc) {
  return new (Context) CXXEnumeratingExpansionStmtPattern(
      cast<CXXExpansionStmtDecl>(ESD), Init, cast<DeclStmt>(ExpansionVar),
      LParenLoc, ColonLoc, RParenLoc);
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

  assert(isa<CXXEnumeratingExpansionStmtPattern>(Expansion) && "TODO");

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

std::optional<uint64_t>
Sema::ComputeExpansionSize(CXXExpansionStmtPattern *Expansion) {
  assert(!Expansion->hasDependentSize());

  if (isa<CXXEnumeratingExpansionStmtPattern>(Expansion))
    return cast<CXXExpansionInitListSelectExpr>(
               Expansion->getExpansionVariable()->getInit())
        ->getRangeExpr()
        ->getExprs()
        .size();

  llvm_unreachable("TODO");
}
