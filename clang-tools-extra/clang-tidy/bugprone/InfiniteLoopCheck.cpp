//===--- InfiniteLoopCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InfiniteLoopCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

static internal::Matcher<Stmt>
loopEndingStmt(internal::Matcher<Stmt> Internal) {
  return stmt(anyOf(breakStmt(Internal), returnStmt(Internal),
                    gotoStmt(Internal), cxxThrowExpr(Internal),
                    callExpr(Internal, callee(functionDecl(isNoReturn())))));
}

/// Return whether `S` is a reference to the declaration of `Var`.
static bool isAccessForVar(const Stmt *S, const VarDecl *Var) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(S))
    return DRE->getDecl() == Var;

  return false;
}

/// Return whether `Var` has a pointer or reference in `S`.
static bool isPtrOrReferenceForVar(const Stmt *S, const VarDecl *Var) {
  if (const auto *DS = dyn_cast<DeclStmt>(S)) {
    for (const Decl *D : DS->getDeclGroup()) {
      if (const auto *LeftVar = dyn_cast<VarDecl>(D)) {
        if (LeftVar->hasInit() && LeftVar->getType()->isReferenceType()) {
          return isAccessForVar(LeftVar->getInit(), Var);
        }
      }
    }
  } else if (const auto *UnOp = dyn_cast<UnaryOperator>(S)) {
    if (UnOp->getOpcode() == UO_AddrOf)
      return isAccessForVar(UnOp->getSubExpr(), Var);
  }

  return false;
}

/// Return whether `Var` has a pointer or reference in `S`.
static bool hasPtrOrReferenceInStmt(const Stmt *S, const VarDecl *Var) {
  if (isPtrOrReferenceForVar(S, Var))
    return true;

  for (const Stmt *Child : S->children()) {
    if (!Child)
      continue;

    if (hasPtrOrReferenceInStmt(Child, Var))
      return true;
  }

  return false;
}

/// Return whether `Var` has a pointer or reference in `Func`.
static bool hasPtrOrReferenceInFunc(const FunctionDecl *Func,
                                    const VarDecl *Var) {
  return hasPtrOrReferenceInStmt(Func->getBody(), Var);
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
static bool isVarThatIsPossiblyChanged(const FunctionDecl *Func,
                                       const Stmt *LoopStmt, const Stmt *Cond,
                                       ASTContext *Context) {
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
  } else if (isa<MemberExpr>(Cond) || isa<CallExpr>(Cond)) {
    // FIXME: Handle MemberExpr.
    return true;
  }

  return false;
}

/// Return whether at least one variable of `Cond` changed in `LoopStmt`.
static bool isAtLeastOneCondVarChanged(const FunctionDecl *Func,
                                       const Stmt *LoopStmt, const Stmt *Cond,
                                       ASTContext *Context) {
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
      return Var->getName();
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

void InfiniteLoopCheck::registerMatchers(MatchFinder *Finder) {
  const auto LoopCondition = allOf(
      hasCondition(
          expr(forFunction(functionDecl().bind("func"))).bind("condition")),
      unless(hasBody(hasDescendant(
          loopEndingStmt(forFunction(equalsBoundNode("func")))))));

  Finder->addMatcher(stmt(anyOf(whileStmt(LoopCondition), doStmt(LoopCondition),
                                forStmt(LoopCondition)))
                         .bind("loop-stmt"),
                     this);
}

void InfiniteLoopCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cond = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *LoopStmt = Result.Nodes.getNodeAs<Stmt>("loop-stmt");
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  if (isAtLeastOneCondVarChanged(Func, LoopStmt, Cond, Result.Context))
    return;

  std::string CondVarNames = getCondVarNames(Cond);
  if (CondVarNames.empty())
    return;

  diag(LoopStmt->getBeginLoc(),
       "this loop is infinite; none of its condition variables (%0)"
       " are updated in the loop body")
      << CondVarNames;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
