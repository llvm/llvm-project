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
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"

#include <clang/Sema/EnterExpressionEvaluationContext.h>
#include <clang/Sema/Template.h>

using namespace clang;
using namespace sema;

namespace {
struct IterableExpansionStmtData {
  enum class State {
    NotIterable,
    Error,
    Ok,
  };

  DeclStmt *RangeDecl{};
  DeclStmt *BeginDecl{};
  DeclStmt *EndDecl{};
  Expr *Initializer{};
  State TheState = State::NotIterable;

  bool isIterable() const { return TheState == State::Ok; }
  bool hasError() { return TheState == State::Error; }
};
} // namespace

  /*
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
*/

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
    LookupResult EndLR(S, BeginName, Sema::LookupMemberName);
    FoundBeginEnd = S.LookupQualifiedName(BeginLR, Record) &&
                    S.LookupQualifiedName(EndLR, Record);
  }

  // Try ADL.
  if (!FoundBeginEnd) {
    OverloadCandidateSet Candidates(ColonLoc,
                                      OverloadCandidateSet::CSK_Normal);

    S.AddArgumentDependentLookupCandidates(BeginName.getName(), ColonLoc,
                                           ExpansionInitializer, nullptr,
                                           Candidates);

    if (Candidates.empty())
      return Data;

    Candidates.clear(OverloadCandidateSet::CSK_Normal);
    S.AddArgumentDependentLookupCandidates(EndName.getName(), ColonLoc,
                                           ExpansionInitializer, nullptr,
                                           Candidates);

    if (Candidates.empty())
      return Data;
  }

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

  assert(CurContext->isExpansionStmt());
  auto *DS = cast<DeclStmt>(ExpansionVarStmt);
  if (!DS->isSingleDecl()) {
    Diag(DS->getBeginLoc(), diag::err_type_defined_in_for_range);
    return StmtError();
  }

  VarDecl *ExpansionVar = cast<VarDecl>(DS->getSingleDecl());
  if (!ExpansionVar || ExpansionVar->isInvalidDecl() ||
      ExpansionInitializer->containsErrors())
    return StmtError();

  auto FinaliseExpansionVar = [&](ExprResult Initializer) {
    if (Initializer.isInvalid()) {
      ActOnInitializerError(ExpansionVar);
      return true;
    }

    AddInitializerToDecl(ExpansionVar, Initializer.get(), /*DirectInit=*/false);
    return ExpansionVar->isInvalidDecl();
  };

  // Build a 'DeclRefExpr' designating the template parameter '__N'.
  DeclRefExpr *Index = BuildDeclRefExpr(ESD->getIndexTemplateParm(),
                                        Context.getPointerDiffType(),
                                        VK_PRValue, ESD->getBeginLoc());

  // This is an enumerating expansion statement.
  if (auto *ILE = dyn_cast<CXXExpansionInitListExpr>(ExpansionInitializer)) {

    ExprResult Initializer = BuildCXXExpansionInitListSelectExpr(ILE, Index);
    if (FinaliseExpansionVar(Initializer))
      return StmtError();

    // Note that lifetime extension only applies to destructurable expansion
    // statements, so we just ignore 'LifetimeExtendedTemps' entirely for other
    // types of expansion statements (this is CWG 3043).
    return BuildCXXEnumeratingExpansionStmt(ESD, Init, DS, ForLoc, LParenLoc,
                                            ColonLoc, RParenLoc);
  }

  if (ExpansionInitializer->isTypeDependent())
    llvm_unreachable("TODO: Dependent expansion initializer");

  // Otherwise, if it can be an iterating expansion statement, it is one.
  IterableExpansionStmtData Data = TryBuildIterableExpansionStmtInitializer(
      *this, ExpansionInitializer, Index, ColonLoc, ExpansionVar->isConstexpr());
  if (Data.hasError()) {
    ActOnInitializerError(ExpansionVar);
    return StmtError();
  }

  if (Data.isIterable()) {
    if (FinaliseExpansionVar(Data.Initializer))
      return StmtError();

    return new (Context) CXXIteratingExpansionStmt(
        ESD, Init, DS, Data.RangeDecl, Data.BeginDecl, Data.EndDecl, ForLoc,
        LParenLoc, ColonLoc, RParenLoc);
  }


  llvm_unreachable("TODO: Destructuring expansion statement");
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

  // This can fail if this is an iterating expansion statement.
  std::optional<uint64_t> NumInstantiations = ComputeExpansionSize(Expansion);
  if (!NumInstantiations)
    return StmtError();

  // TODO: Actually make this configurable. It is set to 32 for now so our
  // tests don't take for ever to run; we should pick a larger default value
  // once we add an option for this and then pass '-fexpansion-limit=32' to
  // the tests.
  static constexpr uint64_t MaxExpansionSize = 32;
  if (MaxExpansionSize != 0 && *NumInstantiations > MaxExpansionSize) {
    Diag(Expansion->getColonLoc(), diag::err_expansion_too_big)
        << *NumInstantiations << MaxExpansionSize;
    Diag(Expansion->getColonLoc(), diag::note_use_fexpansion_limit);
    return StmtError();
  }

  // Collect shared statements.
  SmallVector<Stmt*, 1> Shared;
  if (Expansion->getInit())
    Shared.push_back(Expansion->getInit());

  if (auto *Iter = dyn_cast<CXXIteratingExpansionStmt>(Expansion)) {
    Shared.push_back(Iter->getRangeVarStmt());
    Shared.push_back(Iter->getBeginVarStmt());
    Shared.push_back(Iter->getEndVarStmt());
  }

  // Return an empty statement if the range is empty.
  if (*NumInstantiations == 0) {
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
  for (uint64_t I = 0; I < *NumInstantiations; ++I) {
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

  uint64_t I = ER.Val.getInt().getZExtValue();
  return Range->getExprs()[I];
}

std::optional<uint64_t>  Sema::ComputeExpansionSize(CXXExpansionStmt *Expansion) {
  assert(!Expansion->hasDependentSize());

  if (isa<CXXEnumeratingExpansionStmt>(Expansion)) {
    uint64_t Size = cast<CXXExpansionInitListSelectExpr>(
               Expansion->getExpansionVariable()->getInit())
        ->getRangeExpr()
        ->getExprs()
        .size();

    return Size;
  }

  // By [stmt.expand]5.2, N is the result of evaluating the expression
  //
  // [] consteval {
  //    std::ptrdiff_t result = 0;
  //    for (auto i = begin; i != end; ++i, ++result);
  //    return result;
  // }()
  if (auto *Iterating = dyn_cast<CXXIteratingExpansionStmt>(Expansion)) {
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
      for (const auto& [Location, PDiag] : Notes)
        Diag(Location, PDiag);
      return std::nullopt;
    }

    if (ER.Val.getInt().isNegative()) {
      Diag(Loc, diag::err_expansion_size_negative) << ER.Val.getInt();
      return std::nullopt;
    }

    return ER.Val.getInt().getZExtValue();
  }

  llvm_unreachable("Invalid expansion statement class");
}
