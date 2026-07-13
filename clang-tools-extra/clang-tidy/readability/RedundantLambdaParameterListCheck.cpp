//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantLambdaParameterListCheck.h"
#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(LambdaExpr, hasRedundantParens) {
  return Node.hasExplicitParameters() &&
         Node.getCallOperator()->getNumParams() == 0 &&
         !Node.getCallOperator()->getTrailingRequiresClause();
}

} // namespace

void RedundantLambdaParameterListCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasRedundantParens()).bind("lambda"), this);
}

void RedundantLambdaParameterListCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");

  const LangOptions &LangOpts = getLangOpts();

  const FunctionTypeLoc FTL = Lambda->getCallOperator()->getFunctionTypeLoc();
  const SourceLocation LParenLoc = FTL.getLParenLoc();
  const SourceLocation RParenLoc = FTL.getRParenLoc();

  if (LParenLoc.isInvalid() || RParenLoc.isInvalid())
    return;

  // Ensure parens are truly empty (reject "(void)")
  const std::optional<Token> FirstInParens =
      utils::lexer::findNextTokenSkippingComments(
          LParenLoc, *Result.SourceManager, LangOpts);

  if (!FirstInParens || FirstInParens->getLocation() != RParenLoc)
    return;

  const std::optional<Token> NextAfterParenTok =
      utils::lexer::findNextTokenSkippingComments(
          RParenLoc, *Result.SourceManager, LangOpts);

  if (!NextAfterParenTok || NextAfterParenTok->is(tok::l_square))
    return;

  if (!LangOpts.CPlusPlus23 && NextAfterParenTok->isNot(tok::l_brace))
    return;

  diag(LParenLoc, "redundant empty parameter list in lambda expression")
      << FixItHint::CreateRemoval(LParenLoc)
      << FixItHint::CreateRemoval(RParenLoc);
}

} // namespace clang::tidy::readability
