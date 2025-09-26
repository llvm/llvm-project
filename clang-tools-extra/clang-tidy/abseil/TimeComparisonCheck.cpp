//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TimeComparisonCheck.h"
#include "DurationRewriter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

void TimeComparisonCheck::registerMatchers(MatchFinder *Finder) {
  auto Matcher =
      expr(comparisonOperatorWithCallee(functionDecl(
               functionDecl(TimeConversionFunction()).bind("function_decl"))))
          .bind("binop");

  Finder->addMatcher(Matcher, this);
}

void TimeComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Binop = Result.Nodes.getNodeAs<BinaryOperator>("binop");

  std::optional<DurationScale> Scale = getScaleForTimeInverse(
      Result.Nodes.getNodeAs<FunctionDecl>("function_decl")->getName());
  if (!Scale)
    return;

  if (isInMacro(Result, Binop->getLHS()) || isInMacro(Result, Binop->getRHS()))
    return;

  // In most cases, we'll only need to rewrite one of the sides, but we also
  // want to handle the case of rewriting both sides. This is much simpler if
  // we unconditionally try and rewrite both, and let the rewriter determine
  // if nothing needs to be done.
  std::string LhsReplacement =
      rewriteExprFromNumberToTime(Result, *Scale, Binop->getLHS());
  std::string RhsReplacement =
      rewriteExprFromNumberToTime(Result, *Scale, Binop->getRHS());

  diag(Binop->getBeginLoc(), "perform comparison in the time domain")
      << FixItHint::CreateReplacement(Binop->getSourceRange(),
                                      (llvm::Twine(LhsReplacement) + " " +
                                       Binop->getOpcodeStr() + " " +
                                       RhsReplacement)
                                          .str());
}

} // namespace clang::tidy::abseil
