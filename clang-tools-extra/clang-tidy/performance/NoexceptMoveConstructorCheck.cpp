//===--- NoexceptMoveConstructorCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoexceptMoveConstructorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

// FixItHint - comment added to fix list.rst generation in add_new_check.py.
// Do not remove. Fixes are generated in base class.

namespace clang::tidy::performance {

void NoexceptMoveConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(unless(isDeleted()),
                    anyOf(cxxConstructorDecl(isMoveConstructor()),
                          isMoveAssignmentOperator()))
          .bind(BindFuncDeclName),
      this);
}

DiagnosticBuilder NoexceptMoveConstructorCheck::reportMissingNoexcept(
    const FunctionDecl *FuncDecl) {
  return diag(FuncDecl->getLocation(),
              "move %select{assignment operator|constructor}0s should "
              "be marked noexcept")
         << CXXConstructorDecl::classof(FuncDecl);
}

void NoexceptMoveConstructorCheck::reportNoexceptEvaluatedToFalse(
    const FunctionDecl *FuncDecl, const Expr *NoexceptExpr) {
  diag(NoexceptExpr->getExprLoc(),
       "noexcept specifier on the move %select{assignment "
       "operator|constructor}0 evaluates to 'false'")
      << CXXConstructorDecl::classof(FuncDecl);
}

} // namespace clang::tidy::performance
