//===--- InfiniteLoopCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InfiniteLoopCheck.h"
#include "../utils/Aliasing.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"

using namespace clang::ast_matchers;
using clang::tidy::utils::hasPtrOrReferenceInFunc;

namespace clang {
namespace tidy {
namespace bugprone {

static internal::Matcher<Stmt>
loopEndingStmt(internal::Matcher<Stmt> Internal) {
  // FIXME: Cover noreturn ObjC methods (and blocks?).
  return stmt(anyOf(
      mapAnyOf(breakStmt, returnStmt, gotoStmt, cxxThrowExpr).with(Internal),
      callExpr(Internal, callee(functionDecl(isNoReturn())))));
}

/// Return whether `Var` was changed in `LoopStmt`.
static bool isChanged(const Stmt *LoopStmt, const VarDecl *Var,
                      ASTContext *Context) {
  if (const auto *ForLoop = dyn_cast<ForStmt>(LoopStmt))
    return (ForLoop->getInc() &&
            ExprMutationAnalyzer(*ForLoop->getInc(), *Context)
                .isMutated(Var)) ||
           (ForLoop->getBody() &&
            ExprMutationAnalyzer(*ForLoop->getBody(), *Context)
                .isMutated(Var)) ||
           (ForLoop->getCond() &&
            ExprMutationAnalyzer(*ForLoop->getCond(), *Context).isMutated(Var));

  return ExprMutationAnalyzer(*LoopStmt, *Context).isMutated(Var);
}

/// Return whether `Cond` is a variable that is possibly changed in `LoopStmt`.
static bool isVarThatIsPossiblyChanged(const Decl *Func, const Stmt *LoopStmt,
                                       const Stmt *Cond, ASTContext *Context) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    if (const auto *Var = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (!Var->isLocalVarDeclOrParm())
        return true;

      if (Var->getType().isVolatileQualified())
        return true;

      if (!Var->getType().getTypePtr()->isIntegerType())
        return true;

      return hasPtrOrReferenceInFunc(Func, Var) ||
             isChanged(LoopStmt, Var, Context);
      // FIXME: Track references.
    }
  } else if (isa<MemberExpr, CallExpr,
                 ObjCIvarRefExpr, ObjCPropertyRefExpr, ObjCMessageExpr>(Cond)) {
    // FIXME: Handle MemberExpr.
    return true;
  } else if (const auto *CE = dyn_cast<CastExpr>(Cond)) {
    QualType T = CE->getType();
    while (true) {
      if (T.isVolatileQualified())
        return true;

      if (!T->isAnyPointerType() && !T->isReferenceType())
        break;

      T = T->getPointeeType();
    }
  }

  return false;
}

/// Return whether at least one variable of `Cond` changed in `LoopStmt`.
static bool isAtLeastOneCondVarChanged(const Decl *Func, const Stmt *LoopStmt,
                                       const Stmt *Cond, ASTContext *Context) {
  if (isVarThatIsPossiblyChanged(Func, LoopStmt, Cond, Context))
    return true;

  for (const Stmt *Child : Cond->children()) {
    if (!Child)
      continue;

    if (isAtLeastOneCondVarChanged(Func, LoopStmt, Child, Context))
      return true;
  }
  return false;
}

/// Return the variable names in `Cond`.
static std::string getCondVarNames(const Stmt *Cond) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    if (const auto *Var = dyn_cast<VarDecl>(DRE->getDecl()))
      return std::string(Var->getName());
  }

  std::string Result;
  for (const Stmt *Child : Cond->children()) {
    if (!Child)
      continue;

    std::string NewNames = getCondVarNames(Child);
    if (!Result.empty() && !NewNames.empty())
      Result += ", ";
    Result += NewNames;
  }
  return Result;
}

static bool isKnownFalse(const Expr &Cond, const ASTContext &Ctx) {
  if (Cond.isValueDependent())
    return false;
  bool Result = false;
  if (Cond.EvaluateAsBooleanCondition(Result, Ctx))
    return !Result;
  return false;
}

void InfiniteLoopCheck::registerMatchers(MatchFinder *Finder) {
  const auto LoopCondition = allOf(
      hasCondition(
          expr(forCallable(decl().bind("func"))).bind("condition")),
      unless(hasBody(hasDescendant(
          loopEndingStmt(forCallable(equalsBoundNode("func")))))));

  Finder->addMatcher(mapAnyOf(whileStmt, doStmt, forStmt)
                         .with(LoopCondition)
                         .bind("loop-stmt"),
                     this);
}

void InfiniteLoopCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cond = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *LoopStmt = Result.Nodes.getNodeAs<Stmt>("loop-stmt");
  const auto *Func = Result.Nodes.getNodeAs<Decl>("func");

  if (isKnownFalse(*Cond, *Result.Context))
    return;

  bool ShouldHaveConditionVariables = true;
  if (const auto *While = dyn_cast<WhileStmt>(LoopStmt)) {
    if (const VarDecl *LoopVarDecl = While->getConditionVariable()) {
      if (const Expr *Init = LoopVarDecl->getInit()) {
        ShouldHaveConditionVariables = false;
        Cond = Init;
      }
    }
  }

  if (isAtLeastOneCondVarChanged(Func, LoopStmt, Cond, Result.Context))
    return;

  std::string CondVarNames = getCondVarNames(Cond);
  if (ShouldHaveConditionVariables && CondVarNames.empty())
    return;

  if (CondVarNames.empty()) {
    diag(LoopStmt->getBeginLoc(),
         "this loop is infinite; it does not check any variables in the"
         " condition");
  } else {
    diag(LoopStmt->getBeginLoc(),
         "this loop is infinite; none of its condition variables (%0)"
         " are updated in the loop body")
      << CondVarNames;
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
