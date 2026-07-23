//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ErrnoComparisonCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

void ErrnoComparisonCheck::registerMatchers(MatchFinder *Finder) {
  // Match a comparison that has an integer literal on one side.
  Finder->addMatcher(binaryOperator(isComparisonOperator(),
                                    hasEitherOperand(ignoringParenImpCasts(
                                        integerLiteral().bind("lit"))))
                         .bind("cmp"),
                     this);
}

void ErrnoComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cmp = Result.Nodes.getNodeAs<BinaryOperator>("cmp");
  const auto *Lit = Result.Nodes.getNodeAs<IntegerLiteral>("lit");

  // errno == 0 is the portable way to test for "no error".
  if (Lit->getValue() == 0)
    return;

  // A macro literal (e.g. EINVAL) is already the recommended form.
  if (Lit->getBeginLoc().isMacroID())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const auto IsErrno = [&](const Expr *E) {
    const SourceLocation Loc = E->getBeginLoc();
    return Loc.isMacroID() &&
           Lexer::getImmediateMacroName(Loc, SM, getLangOpts()) == "errno";
  };

  // Warn only when exactly one operand is errno.
  if (IsErrno(Cmp->getLHS()) == IsErrno(Cmp->getRHS()))
    return;

  diag(Cmp->getOperatorLoc(),
       "comparing 'errno' against a literal is not portable");
}

} // namespace clang::tidy::portability
