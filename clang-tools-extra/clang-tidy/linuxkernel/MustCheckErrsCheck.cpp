//===--- MustCheckErrsCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MustCheckErrsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::linuxkernel {

void MustCheckErrsCheck::registerMatchers(MatchFinder *Finder) {
  auto ErrFn =
      functionDecl(hasAnyName("ERR_PTR", "PTR_ERR", "IS_ERR", "IS_ERR_OR_NULL",
                              "ERR_CAST", "PTR_ERR_OR_ZERO"));
  auto NonCheckingStmts = stmt(anyOf(compoundStmt(), labelStmt()));
  Finder->addMatcher(
      callExpr(callee(ErrFn), hasParent(NonCheckingStmts)).bind("call"),
      this);

  auto ReturnToCheck = returnStmt(hasReturnValue(callExpr(callee(ErrFn))));
  auto ReturnsErrFn = functionDecl(hasDescendant(ReturnToCheck));
  Finder->addMatcher(callExpr(callee(ReturnsErrFn), hasParent(NonCheckingStmts))
                         .bind("transitive_call"),
                     this);
}

void MustCheckErrsCheck::check(const MatchFinder::MatchResult &Result) {
  const CallExpr *MatchedCallExpr = Result.Nodes.getNodeAs<CallExpr>("call");
  if (MatchedCallExpr) {
    diag(MatchedCallExpr->getExprLoc(), "result from function %0 is unused")
        << MatchedCallExpr->getDirectCallee();
  }

  const CallExpr *MatchedTransitiveCallExpr =
      Result.Nodes.getNodeAs<CallExpr>("transitive_call");
  if (MatchedTransitiveCallExpr) {
    diag(MatchedTransitiveCallExpr->getExprLoc(),
         "result from function %0 is unused but represents an error value")
        << MatchedTransitiveCallExpr->getDirectCallee();
  }
}

} // namespace clang::tidy::linuxkernel
