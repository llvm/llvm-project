//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRethrowCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void UseRethrowCheck::registerMatchers(MatchFinder *Finder) {
  auto CatchVar = varDecl(isExceptionVariable(), hasType(referenceType()));
  auto RefToVar = declRefExpr(to(CatchVar));

  Finder->addMatcher(
      cxxThrowExpr(has(expr(anyOf(
                       ignoringParenImpCasts(RefToVar),
                       cxxConstructExpr(
                           hasDeclaration(cxxConstructorDecl(anyOf(
                               isCopyConstructor(), isMoveConstructor()))),
                           hasArgument(0, ignoringParenImpCasts(RefToVar)))))))
          .bind("throw"),
      this);
}

void UseRethrowCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedThrow = Result.Nodes.getNodeAs<CXXThrowExpr>("throw");

  if (!MatchedThrow)
    return;

  diag(MatchedThrow->getThrowLoc(),
       "throwing a copy of the caught exception; use a bare 'throw' to rethrow "
       "the original exception object")
      << FixItHint::CreateReplacement(MatchedThrow->getSourceRange(), "throw");
}

} // namespace clang::tidy::readability
