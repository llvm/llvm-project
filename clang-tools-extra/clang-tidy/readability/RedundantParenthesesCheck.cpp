//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantParenthesesCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER_P(ParenExpr, subExpr, ast_matchers::internal::Matcher<Expr>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
}

} // namespace

void RedundantParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      parenExpr(subExpr(anyOf(parenExpr(), integerLiteral(), floatLiteral(),
                              characterLiteral(), cxxBoolLiteral(),
                              stringLiteral(), declRefExpr())),
                unless(
                    // sizeof(...) is common used.
                    hasParent(unaryExprOrTypeTraitExpr())))
          .bind("dup"),
      this);
}

void RedundantParenthesesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *PE = Result.Nodes.getNodeAs<ParenExpr>("dup");
  assert(PE);
  const Expr *E = PE->getSubExpr();
  if (PE->getLParen().isMacroID() || PE->getRParen().isMacroID() ||
      E->getBeginLoc().isMacroID() || E->getEndLoc().isMacroID())
    return;
  diag(PE->getBeginLoc(), "redundant parentheses around expression")
      << FixItHint::CreateRemoval(PE->getLParen())
      << FixItHint::CreateRemoval(PE->getRParen());
}

} // namespace clang::tidy::readability
