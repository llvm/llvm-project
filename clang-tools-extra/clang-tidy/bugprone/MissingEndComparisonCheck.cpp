//===--- MissingEndComparisonCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingEndComparisonCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void MissingEndComparisonCheck::registerMatchers(MatchFinder *Finder) {
  // List of standard algorithms that return an iterator and should be compared
  // to the end iterator.
  // Note: Algorithms returning pairs (like equal_range, mismatch) are excluded
  // because std::pair doesn't implicitly convert to bool, so they wouldn't
  // match CK_PointerToBoolean anyway.
  auto StandardIteratorAlgorithms = functionDecl(hasAnyName(
      "::std::find", "::std::find_if", "::std::find_if_not", "::std::search",
      "::std::search_n", "::std::find_end", "::std::find_first_of",
      "::std::lower_bound", "::std::upper_bound", "::std::partition_point",
      "::std::min_element", "::std::max_element", "::std::adjacent_find",
      "::std::is_sorted_until"));

  // Matcher 1: Implicit cast from pointer to boolean.
  // This catches cases where the algorithm returns a raw pointer (e.g.,
  // finding in a C-array) and it's used in a boolean context.
  Finder->addMatcher(
      implicitCastExpr(
          hasCastKind(CK_PointerToBoolean),
          hasSourceExpression(ignoringParenImpCasts(
              callExpr(callee(StandardIteratorAlgorithms)).bind("call"))))
          .bind("cast"),
      this);

  // Matcher 2: Explicit/Implicit conversion via operator bool.
  // This catches cases where the returned iterator has an explicit or implicit
  // conversion to bool (e.g., some custom iterators).
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxConversionDecl(returns(booleanType()))),
          on(ignoringParenImpCasts(
              callExpr(callee(StandardIteratorAlgorithms)).bind("call"))))
          .bind("cast"),
      this);
}

void MissingEndComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *Cast = Result.Nodes.getNodeAs<Expr>("cast");

  if (!Call || !Cast)
    return;

  // Most standard algorithms take the end iterator as the second argument.
  // Check if we have enough arguments.
  if (Call->getNumArgs() < 2)
    return;

  const Expr *EndIterArg = Call->getArg(1);
  
  // If the second argument is nullptr/NULL, the user might be intentionally 
  // checking against nullptr (though odd for std algorithms, it's possible 
  // for raw pointers).
  if (EndIterArg->isNullPointerConstant(*Result.Context, 
                                        Expr::NPC_ValueDependentIsNull)) {
      return;
  }

  auto Diag = diag(Cast->getBeginLoc(),
                   "result of standard algorithm used in boolean context; did "
                   "you mean to compare with the end iterator?");

  // Try to generate a fix-it.
  // We want to rewrite the expression 'E' to 'E != EndIter'.
  // 'Cast' is the boolean expression (e.g. the implicit cast or the bool conversion call).
  // However, simply appending '!= End' to the end of the Cast's range might be tricky
  // if there are precedence issues, but usually != has lower precedence than function calls
  // and higher than assignment/logic.
  
  // Get the source text of the end iterator argument.
  StringRef EndIterText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(EndIterArg->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());

  if (EndIterText.empty())
    return;

  // Check if the end iterator expression is safe to duplicate.
  // (Simple variable, member access, call to .end()).
  // For now, we'll be conservative. If it looks complex, skip the fix-it.
  // A simple heuristic: if it contains side effects or is too long, skip.
  // But we can just provide the hint.
  
  Diag << FixItHint::CreateInsertion(
      Lexer::getLocForEndOfToken(Cast->getEndLoc(), 0, *Result.SourceManager,
                                 Result.Context->getLangOpts()),
      (" != " + EndIterText).str());
}

} // namespace clang::tidy::bugprone