//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This check detects ternary conditional (?:) expressions and replaces them
// with equivalent if/else statements to improve code readability.
//
// Example:
//
//   int x = cond ? 1 : 2;
//
// Becomes:
//
//   int x;
//   if (cond)
//     x = 1;
//   else
//     x = 2;
//
//===----------------------------------------------------------------------===//

#include "ConditionalToIfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h" // <-- ADD THIS INCLUDE
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void ConditionalToIfCheck::registerMatchers(MatchFinder *Finder) {
  // Match ternary conditional operators (?:)
  Finder->addMatcher(
      conditionalOperator(hasTrueExpression(expr().bind("trueExpr")),
                          hasFalseExpression(expr().bind("falseExpr")),
                          hasCondition(expr().bind("condExpr")))
          .bind("ternary"),
      this);
}

void ConditionalToIfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ternary = Result.Nodes.getNodeAs<ConditionalOperator>("ternary");
  if (!Ternary)
    return;

  const Expr *Cond = Result.Nodes.getNodeAs<Expr>("condExpr");
  const Expr *TrueExpr = Result.Nodes.getNodeAs<Expr>("trueExpr");
  const Expr *FalseExpr = Result.Nodes.getNodeAs<Expr>("falseExpr");

  if (!Cond || !TrueExpr || !FalseExpr)
    return;

  const SourceManager &SM = *Result.SourceManager;

  auto Diag =
      diag(Ternary->getBeginLoc(),
           "replace ternary operator with if/else statement for readability");

  // Extract source text for condition, true and false expressions
  const std::string CondStr = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                                 Cond->getSourceRange()),
                                             SM, Result.Context->getLangOpts())
                            .str();

  const std::string TrueStr =
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(TrueExpr->getSourceRange()), SM,
          Result.Context->getLangOpts())
          .str();

  const std::string FalseStr =
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(FalseExpr->getSourceRange()), SM,
          Result.Context->getLangOpts())
          .str();

  // Construct the replacement code
  const std::string Replacement =
      "{ if (" + CondStr + ") " + TrueStr + "; else " + FalseStr + "; }";

  Diag << FixItHint::CreateReplacement(Ternary->getSourceRange(), Replacement);
}

} // namespace clang::tidy::readability
