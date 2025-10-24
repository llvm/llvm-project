//===-- SemaExpand.cpp - Semantic Analysis for Expansion Statements--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ 26 expansion statements,
//  aka 'template for'.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"

#include <clang/Sema/Template.h>

using namespace clang;
using namespace sema;

static unsigned ExtractParmVarDeclDepth(Expr *E) {
  if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto *PVD = cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
      return PVD->getDepth();
  } else if (auto *SNTTPE = cast<SubstNonTypeTemplateParmExpr>(E)) {
    if (auto *PVD = cast<NonTypeTemplateParmDecl>(SNTTPE->getAssociatedDecl()))
      return PVD->getDepth();
  }
  return 0;
}

/*
// Returns 'true' if the 'Range' is an iterable expression, and 'false'
// otherwise. If 'true', then 'Result' contains the resulting
// 'CXXIterableExpansionSelectExpr' (or error).
static bool TryMakeCXXIterableExpansionSelectExpr(
    Sema &S, Expr *Range, Expr *Index, VarDecl *ExpansionVar,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps,
    ExprResult &SelectResult) {
  auto Ctx = Sema::ExpressionEvaluationContext::PotentiallyEvaluated;
  if (ExpansionVar->isConstexpr())
    // TODO: Shouldn’t this be 'ConstantEvaluated'?
    Ctx = Sema::ExpressionEvaluationContext::ImmediateFunctionContext;
  EnterExpressionEvaluationContext ExprEvalCtx(S, Ctx);

  // C++26 [stmt.expand]p3: An expression is expansion-iterable if it does not
  // have array type [...]
  if (Range->getType()->isArrayType())
    return false;

  SourceLocation RangeLoc = Range->getExprLoc();
  DeclarationNameInfo BeginName(&S.PP.getIdentifierTable().get("begin"),
                                RangeLoc);
  LookupResult BeginLR(S, BeginName, Sema::LookupMemberName);
  if (auto *RD = Range->getType()->getAsCXXRecordDecl())
    S.LookupQualifiedName(BeginLR, RD);

  VarDecl *RangeVar;
  Expr *VarRef;
  {
    assert(isa<ExpansionStmtDecl>(S.CurContext));
    DeclContext *DC = S.CurContext->getEnclosingNonExpansionStatementContext();
    IdentifierInfo *II = &S.PP.getIdentifierTable().get("__range");
    QualType QT = Range->getType().withConst();
    TypeSourceInfo *TSI = S.Context.getTrivialTypeSourceInfo(QT);
    RangeVar = VarDecl::Create(S.Context, DC, Range->getBeginLoc(),
                               Range->getBeginLoc(), II, QT, TSI, SC_Auto);

    if (ExpansionVar->isConstexpr())
      RangeVar->setConstexpr(true);
    else if (!LifetimeExtendTemps.empty()) {
      // TODO: The original patch was performing lifetime extension here, but
      // CWG 3043 seems to have removed that clause. Is that actually what we
      // want here?
      // S.ApplyForRangeOrExpansionStatementLifetimeExtension(
      //     RangeVar, LifetimeExtendTemps);
    }

    S.AddInitializerToDecl(RangeVar, Range, /*DirectInit=#1#false);
    if (RangeVar->isInvalidDecl())
      return false;

    DeclarationNameInfo Name(II, Range->getBeginLoc());
    VarRef = S.BuildDeclRefExpr(RangeVar, Range->getType(), VK_LValue, Name,
                                /*CXXScopeSpec=#1#nullptr, RangeVar);
  }

  ExprResult BeginResult;
  {
    OverloadCandidateSet CandidateSet(RangeLoc,
                                      OverloadCandidateSet::CSK_Normal);
    Sema::ForRangeStatus Status =
        S.BuildForRangeBeginEndCall(RangeLoc, RangeLoc, BeginName, BeginLR,
                                    &CandidateSet, VarRef, &BeginResult);
    if (Status != Sema::FRS_Success)
      return false;

    assert(!BeginResult.isInvalid());
  }
  SelectResult = ExprError();

  // At this point, we know that this is supposed to be an iterable expansion
  // statement, so any failure here is a hard error.
  ExprResult BeginPlusIndex = S.ActOnBinOp(S.getCurScope(), RangeLoc, tok::plus,
                                           BeginResult.get(), Index);
  if (BeginPlusIndex.isInvalid()) {
    SelectResult = ExprError();
    return true;
  }

  ExprResult Deref = S.ActOnUnaryOp(S.getCurScope(), RangeLoc, tok::star,
                                    BeginPlusIndex.get());
  if (Deref.isInvalid()) {
    SelectResult = ExprError();
    return true;
  }

  SelectResult = S.BuildCXXIterableExpansionSelectExpr(RangeVar, Impl.get());
  return true;
}*/

/// Determine whether this should be an iterable expansion statement, and, if
/// so, synthesise the various AST nodes that are required for one.
///
/// \return ExprEmpty() if this is not an iterable expansion statement.
/// \return ExprError() if there was a hard error.
/// \return A CXXIterableExpansionSelectExpr otherwise.
static ExprResult TryBuildIterableExpansionSelectExpr(Sema &S, Scope *Scope,
                                                      Expr *Range, Expr *Index,
                                                      VarDecl *ExpansionVar,
                                                      SourceLocation ColonLoc) {
  llvm_unreachable("TODO");
  /*// C++26 [stmt.expand]p3: An expression is expansion-iterable if it does not
  // have array type [...]
  if (Range->getType()->isArrayType())
    return ExprEmpty();

  // Build the 'range', 'begin', and 'end' variables.
  DeclStmt* RangeVar{};
  auto BuildBeginEnd = [&](Sema::BuildForRangeKind Kind) ->
  Sema::ForRangeBeginEndInfo { StmtResult Var =
        S.BuildCXXForRangeRangeVar(Scope, Range, /*ForExpansionStmt=#1#true);
    if (!Var.isUsable())
      return {};

    RangeVar = cast<DeclStmt>(Var.get());
    return S.BuildCXXForRangeBeginEndVars(
        Scope, cast<VarDecl>(RangeVar->getSingleDecl()), ColonLoc,
        /*CoawaitLoc=#1#{},
        /*LifetimeExtendTemps=#1#{}, Kind, /*ForExpansionStmt=#1#true);
  };

  // The construction of begin-expr and end-expr proceeds as for range-based for
  // loops, except that the 'begin' and 'end' variables are 'static constexpr'.
  //
  // FIXME: Instead of doing this jank, do the lookup for begin/end manually
  // (or factor it out from the for-range code), and only then build the begin/end
  // expression.
  {
    Sema::SFINAETrap Trap(S);
    if (!BuildBeginEnd(Sema::BFRK_Check).isValid())
      return ExprEmpty();
  }

  // Ok, we have confirmed that this is possible; rebuild it without the trap.
  Sema::ForRangeBeginEndInfo Info =BuildBeginEnd(Sema::BFRK_Build);
  if (!Info.isValid())
    return ExprError();

  // By [stmt.expand]5.2, N is the result of evaluating the expression
  //
  // [] consteval {
  //    std::ptrdiff_t result = 0;
  //    for (auto i = begin; i != end; ++i, ++result);
  //    return result;
  // }()
  //
  // FIXME: Actually do that; unfortunately, conjuring a lambda out of thin
  // air in Sema is a massive pain, so for now just cheat by computing
  // 'end - begin'.
  auto CreateBeginDRE = [&] {
    return S.BuildDeclRefExpr(Info.BeginVar,
                              Info.BeginVar->getType().getNonReferenceType(),
                              VK_LValue, ColonLoc);
  };

  DeclRefExpr *Begin = CreateBeginDRE();
  DeclRefExpr *End = S.BuildDeclRefExpr(
      Info.EndVar, Info.EndVar->getType().getNonReferenceType(), VK_LValue,
      ColonLoc);

  ExprResult N = S.ActOnBinOp(Scope, ColonLoc, tok::minus, Begin, End);
  if (N.isInvalid())
    return ExprError();

  // Build '*(begin + i)'.
  Begin = CreateBeginDRE();
  ExprResult BeginPlusI = S.ActOnBinOp(Scope, ColonLoc, tok::plus, Begin,
  Index); if (BeginPlusI.isInvalid()) return ExprError();

  ExprResult Deref = S.ActOnUnaryOp(Scope, ColonLoc, tok::star,
  BeginPlusI.get()); if (Deref.isInvalid()) return ExprError();

  Deref = S.MaybeCreateExprWithCleanups(Deref.get());*/
}

ExpansionStmtDecl *Sema::ActOnExpansionStmtDecl(unsigned TemplateDepth,
                                                SourceLocation TemplateKWLoc) {
  // Create a template parameter '__N'. This will be used to denote the index
  // of the element that we're instantiating. The wording around iterable
  // expansion statements (which are the only kind of expansion statements that
  // actually use this parameter in an expression) implies that its type should
  // be 'ptrdiff_t', so use that in all cases.
  IdentifierInfo *ParmName = &Context.Idents.get("__N");
  QualType ParmTy = Context.getPointerDiffType();
  TypeSourceInfo *ParmTI =
      Context.getTrivialTypeSourceInfo(ParmTy, TemplateKWLoc);

  auto *TParam = NonTypeTemplateParmDecl::Create(
      Context, Context.getTranslationUnitDecl(), TemplateKWLoc, TemplateKWLoc,
      TemplateDepth, /*Position=*/0, ParmName, ParmTy, /*ParameterPack=*/false,
      ParmTI);

  return BuildExpansionStmtDecl(CurContext, TemplateKWLoc, TParam);
}

ExpansionStmtDecl *Sema::BuildExpansionStmtDecl(DeclContext *Ctx,
                                                SourceLocation TemplateKWLoc,
                                                NonTypeTemplateParmDecl *NTTP) {
  auto *TParamList = TemplateParameterList::Create(
      Context, TemplateKWLoc, TemplateKWLoc, {NTTP}, TemplateKWLoc,
      /*RequiresClause=*/nullptr);
  auto *Result =
      ExpansionStmtDecl::Create(Context, Ctx, TemplateKWLoc, TParamList);
  Ctx->addDecl(Result);
  return Result;
}

ExprResult Sema::ActOnCXXExpansionInitList(MultiExprArg SubExprs,
                                           SourceLocation LBraceLoc,
                                           SourceLocation RBraceLoc) {
  return CXXExpansionInitListExpr::Create(Context, SubExprs, LBraceLoc,
                                          RBraceLoc);
}

StmtResult Sema::ActOnCXXExpansionStmt(
    ExpansionStmtDecl *ESD, Stmt *Init, Stmt *ExpansionVarStmt,
    Expr *ExpansionInitializer, SourceLocation ForLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc, BuildForRangeKind Kind,
    ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  // TODO: Do we actually need a BuildForRangeKind here at all?
  if (!ExpansionInitializer || !ExpansionVarStmt || Kind == BFRK_Check)
    return StmtError();

  auto *DS = cast<DeclStmt>(ExpansionVarStmt);
  if (!DS->isSingleDecl()) {
    Diag(DS->getBeginLoc(), diag::err_type_defined_in_for_range);
    return StmtError();
  }

  VarDecl *ExpansionVar = cast<VarDecl>(DS->getSingleDecl());
  if (!ExpansionVar || ExpansionVar->isInvalidDecl())
    return StmtError();

  ExprResult ER = BuildCXXExpansionInitializer(ESD, ExpansionInitializer);
  if (ER.isInvalid()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  Expr *Initializer = ER.get();
  AddInitializerToDecl(ExpansionVar, Initializer, /*DirectInit=*/false);
  if (ExpansionVar->isInvalidDecl())
    return StmtError();

  if (isa<CXXExpansionInitListExpr>(ExpansionInitializer))
    return BuildCXXEnumeratingExpansionStmt(ESD, Init, DS, ForLoc, LParenLoc,
                                            ColonLoc, RParenLoc);


  llvm_unreachable("TODO");
}

StmtResult Sema::BuildCXXEnumeratingExpansionStmt(Decl *ESD, Stmt *Init,
                                                  Stmt *ExpansionVar,
                                                  SourceLocation ForLoc,
                                                  SourceLocation LParenLoc,
                                                  SourceLocation ColonLoc,
                                                  SourceLocation RParenLoc) {
  return new (Context) CXXEnumeratingExpansionStmt(
      cast<ExpansionStmtDecl>(ESD), Init, cast<DeclStmt>(ExpansionVar), ForLoc,
      LParenLoc, ColonLoc, RParenLoc);
}

/*
StmtResult Sema::BuildCXXExpansionStmt(
      ExpansionStmtDecl *ESD, Stmt *Init, Stmt *ExpansionVarStmt,
      Expr *ExpansionInitializer, SourceLocation ForLoc,
      SourceLocation LParenLoc, SourceLocation ColonLoc,
      SourceLocation RParenLoc,
      ArrayRef<MaterializeTemporaryExpr *> LifetimeExtendTemps) {
  auto *ExpansionVar = cast<DeclStmt>(ExpansionVarStmt);
  Expr *Initializer = cast<VarDecl>(ExpansionVar->getSingleDecl())->getInit();
  assert(Initializer);

  if (auto *WithCleanups = dyn_cast<ExprWithCleanups>(Initializer))
    Initializer = WithCleanups->getSubExpr();

  if (Initializer->isTypeDependent())
    llvm_unreachable("TODO");

  if (isa<CXXDependentExpansionInitListSelectExpr>(Initializer))
    return CXXExpansionStmt::Create(Context, Init, ExpansionVar,
                                    ESD->getLocation(), ForLoc, LParenLoc,
                                    ColonLoc, RParenLoc);

  llvm_unreachable("TODO");
  /*else if (isa<CXXDestructurableExpansionSelectExpr>(Initializer)) {
    return BuildCXXDestructurableExpansionStmt(TemplateKWLoc, ForLoc, LParenLoc,
                                               Init, ExpansionVarStmt, ColonLoc,
                                               RParenLoc, Index);
  } else if (auto *IESE = dyn_cast<CXXIterableExpansionSelectExpr>(Initializer))
  { ExprResult Size = makeIterableExpansionSizeExpr(*this, IESE->getRangeVar());
    if (Size.isInvalid()) {
      Diag(IESE->getExprLoc(), diag::err_compute_expansion_size_index) << 0;
      return StmtError();
    }
    return BuildCXXIterableExpansionStmt(TemplateKWLoc, ForLoc, LParenLoc, Init,
                                         ExpansionVarStmt, ColonLoc, RParenLoc,
                                         Index, Size.get());
  }
  llvm_unreachable("unknown expansion select expression");#1#
}*/

StmtResult Sema::FinishCXXExpansionStmt(Stmt* Exp, Stmt *Body) {
  if (!Exp || !Body)
    return StmtError();

  auto *Expansion = cast<CXXExpansionStmt>(Exp);
  assert(!Expansion->getDecl()->getInstantiations() &&
         "should not rebuild expansion statement after instantiation");

  // Diagnose identifier labels.
  // TODO: Do this somewhere, somehow, but not every time we instantiate this.
  /*struct DiagnoseLabels : DynamicRecursiveASTVisitor {
    Sema &SemaRef;
    DiagnoseLabels(Sema &S) : SemaRef(S) {}
    bool VisitLabelStmt(LabelStmt *S) override {
      SemaRef.Diag(S->getIdentLoc(), diag::err_expanded_identifier_label);
      return false;
    }
  } Visitor(*this);
  if (!Visitor.TraverseStmt(Body))
    return StmtError();*/

  Expansion->setBody(Body);
  if (Expansion->hasDependentSize())
    return Expansion;

  // Collect shared statements.
  SmallVector<Stmt*, 1> Shared;
  if (Expansion->getInit())
    Shared.push_back(Expansion->getInit());

  // Return an empty statement if the range is empty.
  size_t NumInstantiations = Expansion->getNumInstantiations();
  if (NumInstantiations == 0) {
    Expansion->getDecl()->setInstantiations(
        CXXExpansionInstantiationStmt::Create(Context, Expansion->getBeginLoc(),
                                              /*Instantiations=*/{}, Shared));
    return Expansion;
  }

  // Create a compound statement binding loop and body.
  Stmt *VarAndBody[] = {Expansion->getExpansionVarStmt(), Body};
  Stmt *CombinedBody =
      CompoundStmt::Create(Context, VarAndBody, FPOptionsOverride(),
                           Expansion->getBeginLoc(), Expansion->getEndLoc());

  // Expand the body for each instantiation.
  SmallVector<Stmt *, 4> Instantiations;
  ExpansionStmtDecl *ESD = Expansion->getDecl();
  for (size_t I = 0; I < NumInstantiations; ++I) {
    // Now that we're expanding this, exit the context of the expansion stmt
    // so that we no longer treat this as dependent.
    ContextRAII CtxGuard(*this,
                         CurContext->getEnclosingNonExpansionStatementContext(),
                         /*NewThis=*/false);

    TemplateArgument Arg{Context, llvm::APSInt::get(I),
                         Context.getPointerDiffType()};
    MultiLevelTemplateArgumentList MTArgList(ESD, Arg, true);
    MTArgList.addOuterRetainedLevels(
        Expansion->getDecl()->getIndexTemplateParm()->getDepth());

    LocalInstantiationScope LIScope(*this, /*CombineWithOuterScope=*/true);
    InstantiatingTemplate Inst(*this, Body->getBeginLoc(), Expansion, Arg,
                               Body->getSourceRange());

    StmtResult Instantiation = SubstStmt(CombinedBody, MTArgList);

    if (Instantiation.isInvalid())
      return StmtError();
    Instantiations.push_back(Instantiation.get());
  }

  auto *InstantiationsStmt = CXXExpansionInstantiationStmt::Create(
      Context, Expansion->getBeginLoc(), Instantiations, Shared);

  Expansion->getDecl()->setInstantiations(InstantiationsStmt);
  return Expansion;
}

ExprResult Sema::BuildCXXExpansionInitializer(ExpansionStmtDecl *ESD,
                                              Expr *ExpansionInitializer) {
  if (ExpansionInitializer->containsErrors())
    return ExprError();

  // This should only happen when we first parse the statement.
  //
  // Note that lifetime extension only applies to destructurable expansion
  // statements, so we just ignore 'LifetimeExtendedTemps' entirely for other
  // types of expansion statements (this is CWG 3043).
  if (auto *ILE = dyn_cast<CXXExpansionInitListExpr>(ExpansionInitializer)) {
    // Build a 'DeclRefExpr' designating the template parameter '__N'.
    DeclRefExpr *Index =
        BuildDeclRefExpr(ESD->getIndexTemplateParm(), Context.getSizeType(),
                         VK_PRValue, ESD->getBeginLoc());

    return BuildCXXExpansionInitListSelectExpr(ILE, Index);
  }

  if (ExpansionInitializer->isTypeDependent())
    return ExpansionInitializer;

  ExpansionInitializer->dumpColor();
  llvm_unreachable("TODO: handle this expansion initialiser");
  /*ExprResult IterableExprResult = TryBuildIterableExpansionSelectExpr(
      *this, Range, Index, ExpansionVar, LifetimeExtendTemps,
      IterableExprResult);
  if (!IterableExprResult.isUnset())
    return IterableExprResult;

  return BuildDestructurableExpansionSelectExpr(
      *this, Range, Index, ExpansionVar, LifetimeExtendTemps);*/
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
    llvm_unreachable("Failed to evaluate expansion init list index");

  size_t I = ER.Val.getInt().getZExtValue();
  return Range->getExprs()[I];
}
