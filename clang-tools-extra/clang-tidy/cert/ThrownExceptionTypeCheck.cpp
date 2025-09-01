//===--- ThrownExceptionTypeCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrownExceptionTypeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void ThrownExceptionTypeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxThrowExpr(has(ignoringParenImpCasts(
              cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                                   isCopyConstructor(), unless(isNoThrow()))))
                  .bind("expr"))))),
      this);
}

void ThrownExceptionTypeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
  diag(E->getExprLoc(),
       "thrown exception type is not nothrow copy constructible");
}

} // namespace clang::tidy::cert
