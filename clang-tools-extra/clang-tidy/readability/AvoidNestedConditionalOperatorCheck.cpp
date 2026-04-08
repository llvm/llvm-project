//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNestedConditionalOperatorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/DiagnosticIDs.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void AvoidNestedConditionalOperatorCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      conditionalOperator(
          anyOf(
              hasCondition(ignoringParenCasts(
                  conditionalOperator().bind("nested-conditional-operator"))),
              hasTrueExpression(ignoringParenCasts(
                  conditionalOperator().bind("nested-conditional-operator"))),
              hasFalseExpression(ignoringParenCasts(
                  conditionalOperator().bind("nested-conditional-operator")))))
          .bind("conditional-operator"),
      this);
}

void AvoidNestedConditionalOperatorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *CO =
      Result.Nodes.getNodeAs<ConditionalOperator>("conditional-operator");
  const auto *NCO = Result.Nodes.getNodeAs<ConditionalOperator>(
      "nested-conditional-operator");
  assert(CO);
  assert(NCO);

  if (CO->getBeginLoc().isMacroID() || NCO->getBeginLoc().isMacroID())
    return;

  diag(NCO->getBeginLoc(),
       "conditional operator is used as sub-expression of parent conditional "
       "operator, refrain from using nested conditional operators");
  diag(CO->getBeginLoc(), "parent conditional operator here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::readability
