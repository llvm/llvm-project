//===--- RedundantBranchConditionCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantBranchConditionCheck.h"
#include "../utils/Aliasing.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using clang::tidy::utils::hasPtrOrReferenceInFunc;

namespace clang {
namespace tidy {
namespace bugprone {

static const char CondVarStr[] = "cond_var";
static const char OuterIfStr[] = "outer_if";
static const char InnerIfStr[] = "inner_if";
static const char OuterIfVar1Str[] = "outer_if_var1";
static const char OuterIfVar2Str[] = "outer_if_var2";
static const char InnerIfVar1Str[] = "inner_if_var1";
static const char InnerIfVar2Str[] = "inner_if_var2";
static const char FuncStr[] = "func";

/// Returns whether `Var` is changed in range (`PrevS`..`NextS`).
static bool isChangedBefore(const Stmt *S, const Stmt *NextS, const Stmt *PrevS,
                            const VarDecl *Var, ASTContext *Context) {
  ExprMutationAnalyzer MutAn(*S, *Context);
  const auto &SM = Context->getSourceManager();
  const Stmt *MutS = MutAn.findMutation(Var);
  return MutS &&
         SM.isBeforeInTranslationUnit(PrevS->getEndLoc(),
                                      MutS->getBeginLoc()) &&
         SM.isBeforeInTranslationUnit(MutS->getEndLoc(), NextS->getBeginLoc());
}

void RedundantBranchConditionCheck::registerMatchers(MatchFinder *Finder) {
  const auto ImmutableVar =
      varDecl(anyOf(parmVarDecl(), hasLocalStorage()), hasType(isInteger()),
              unless(hasType(isVolatileQualified())))
          .bind(CondVarStr);
  Finder->addMatcher(
      ifStmt(
          hasCondition(ignoringParenImpCasts(anyOf(
              declRefExpr(hasDeclaration(ImmutableVar)).bind(OuterIfVar1Str),
              binaryOperator(hasOperatorName("&&"),
                             hasEitherOperand(ignoringParenImpCasts(
                                 declRefExpr(hasDeclaration(ImmutableVar))
                                     .bind(OuterIfVar2Str))))))),
          hasThen(hasDescendant(
              ifStmt(hasCondition(ignoringParenImpCasts(
                         anyOf(declRefExpr(hasDeclaration(varDecl(
                                            equalsBoundNode(CondVarStr))))
                                .bind(InnerIfVar1Str),
                               binaryOperator(
                                   hasAnyOperatorName("&&", "||"),
                                   hasEitherOperand(ignoringParenImpCasts(
                                       declRefExpr(hasDeclaration(varDecl(
                                                 equalsBoundNode(CondVarStr))))
                                     .bind(InnerIfVar2Str))))))))
                  .bind(InnerIfStr))),
          forFunction(functionDecl().bind(FuncStr)))
          .bind(OuterIfStr),
      this);
  // FIXME: Handle longer conjunctive and disjunctive clauses.
}

void RedundantBranchConditionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *OuterIf = Result.Nodes.getNodeAs<IfStmt>(OuterIfStr);
  const auto *InnerIf = Result.Nodes.getNodeAs<IfStmt>(InnerIfStr);
  const auto *CondVar = Result.Nodes.getNodeAs<VarDecl>(CondVarStr);
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>(FuncStr);

  const DeclRefExpr *OuterIfVar, *InnerIfVar;
  if (const auto *Inner = Result.Nodes.getNodeAs<DeclRefExpr>(InnerIfVar1Str))
    InnerIfVar = Inner;
  else
    InnerIfVar = Result.Nodes.getNodeAs<DeclRefExpr>(InnerIfVar2Str);
  if (const auto *Outer = Result.Nodes.getNodeAs<DeclRefExpr>(OuterIfVar1Str))
    OuterIfVar = Outer;
  else
    OuterIfVar = Result.Nodes.getNodeAs<DeclRefExpr>(OuterIfVar2Str);

  if (OuterIfVar && InnerIfVar) {
    if (isChangedBefore(OuterIf->getThen(), InnerIfVar, OuterIfVar, CondVar,
                        Result.Context))
      return;

    if (isChangedBefore(OuterIf->getCond(), InnerIfVar, OuterIfVar, CondVar,
                        Result.Context))
      return;
  }

  // If the variable has an alias then it can be changed by that alias as well.
  // FIXME: could potentially support tracking pointers and references in the
  // future to improve catching true positives through aliases.
  if (hasPtrOrReferenceInFunc(Func, CondVar))
    return;

  auto Diag = diag(InnerIf->getBeginLoc(), "redundant condition %0") << CondVar;

  // For standalone condition variables and for "or" binary operations we simply
  // remove the inner `if`.
  const auto *BinOpCond =
      dyn_cast<BinaryOperator>(InnerIf->getCond()->IgnoreParenImpCasts());

  if (isa<DeclRefExpr>(InnerIf->getCond()->IgnoreParenImpCasts()) ||
      (BinOpCond && BinOpCond->getOpcode() == BO_LOr)) {
    SourceLocation IfBegin = InnerIf->getBeginLoc();
    const Stmt *Body = InnerIf->getThen();
    const Expr *OtherSide = nullptr;
    if (BinOpCond) {
      const auto *LeftDRE =
          dyn_cast<DeclRefExpr>(BinOpCond->getLHS()->IgnoreParenImpCasts());
      if (LeftDRE && LeftDRE->getDecl() == CondVar)
        OtherSide = BinOpCond->getRHS();
      else
        OtherSide = BinOpCond->getLHS();
    }

    SourceLocation IfEnd = Body->getBeginLoc().getLocWithOffset(-1);

    // For compound statements also remove the left brace.
    if (isa<CompoundStmt>(Body))
      IfEnd = Body->getBeginLoc();

    // If the other side has side effects then keep it.
    if (OtherSide && OtherSide->HasSideEffects(*Result.Context)) {
      SourceLocation BeforeOtherSide =
          OtherSide->getBeginLoc().getLocWithOffset(-1);
      SourceLocation AfterOtherSide =
          Lexer::findNextToken(OtherSide->getEndLoc(), *Result.SourceManager,
                               getLangOpts())
              ->getLocation();
      Diag << FixItHint::CreateRemoval(
                  CharSourceRange::getTokenRange(IfBegin, BeforeOtherSide))
           << FixItHint::CreateInsertion(AfterOtherSide, ";")
           << FixItHint::CreateRemoval(
                  CharSourceRange::getTokenRange(AfterOtherSide, IfEnd));
    } else {
      Diag << FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(IfBegin, IfEnd));
    }

    // For compound statements also remove the right brace at the end.
    if (isa<CompoundStmt>(Body))
      Diag << FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(Body->getEndLoc(), Body->getEndLoc()));

    // For "and" binary operations we remove the "and" operation with the
    // condition variable from the inner if.
  } else {
    const auto *CondOp =
        cast<BinaryOperator>(InnerIf->getCond()->IgnoreParenImpCasts());
    const auto *LeftDRE =
        dyn_cast<DeclRefExpr>(CondOp->getLHS()->IgnoreParenImpCasts());
    if (LeftDRE && LeftDRE->getDecl() == CondVar) {
      SourceLocation BeforeRHS =
          CondOp->getRHS()->getBeginLoc().getLocWithOffset(-1);
      Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
          CondOp->getLHS()->getBeginLoc(), BeforeRHS));
    } else {
      SourceLocation AfterLHS =
          Lexer::findNextToken(CondOp->getLHS()->getEndLoc(),
                               *Result.SourceManager, getLangOpts())
              ->getLocation();
      Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
          AfterLHS, CondOp->getRHS()->getEndLoc()));
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
