//===--- NoexceptDestructorCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoexceptDestructorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

// FixItHint - comment added to fix list.rst generation in add_new_check.py.
// Do not remove. Fixes are generated in base class.

namespace clang::tidy::performance {

void NoexceptDestructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(unless(isDeleted()), cxxDestructorDecl())
                         .bind(BindFuncDeclName),
                     this);
}

DiagnosticBuilder
NoexceptDestructorCheck::reportMissingNoexcept(const FunctionDecl *FuncDecl) {
  return diag(FuncDecl->getLocation(), "destructors should "
                                       "be marked noexcept");
}

void NoexceptDestructorCheck::reportNoexceptEvaluatedToFalse(
    const FunctionDecl *FuncDecl, const Expr *NoexceptExpr) {
  diag(NoexceptExpr->getExprLoc(),
       "noexcept specifier on the destructor evaluates to 'false'");
}

} // namespace clang::tidy::performance
