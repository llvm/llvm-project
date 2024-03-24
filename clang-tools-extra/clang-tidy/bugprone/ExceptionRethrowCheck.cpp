//===--- ExceptionRethrowCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionRethrowCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
AST_MATCHER(VarDecl, isExceptionVariable) { return Node.isExceptionVariable(); }
} // namespace

void ExceptionRethrowCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxThrowExpr(unless(isExpansionInSystemHeader()),
                   anyOf(unless(has(expr())),
                         has(declRefExpr(to(varDecl(isExceptionVariable()))))),
                   optionally(hasAncestor(cxxCatchStmt().bind("catch"))))
          .bind("throw"),
      this);
}

void ExceptionRethrowCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedThrow = Result.Nodes.getNodeAs<CXXThrowExpr>("throw");

  if (const Expr *ThrownObject = MatchedThrow->getSubExpr()) {
    diag(MatchedThrow->getThrowLoc(),
         "throwing a copy of the caught %0 exception, remove the argument to "
         "throw the original exception object")
        << ThrownObject->getType().getNonReferenceType();
    return;
  }

  const bool HasCatchAnsestor =
      Result.Nodes.getNodeAs<Stmt>("catch") != nullptr;
  if (!HasCatchAnsestor) {
    diag(MatchedThrow->getThrowLoc(),
         "empty 'throw' outside a catch block without an exception can trigger "
         "'std::terminate'");
  }
}

} // namespace clang::tidy::bugprone
