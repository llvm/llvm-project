//===--- AvoidNestedConditionalOperatorCheck.cpp - clang-tidy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNestedConditionalOperatorCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/DiagnosticIDs.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
constexpr const char *Description = "don't use nested conditional operator";
constexpr const char *OutSideConditionalOperatorNote =
    "outside conditional operator here";
} // namespace

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

  diag(NCO->getBeginLoc(), Description);
  diag(CO->getBeginLoc(), OutSideConditionalOperatorNote, DiagnosticIDs::Note);
}

} // namespace clang::tidy::readability
