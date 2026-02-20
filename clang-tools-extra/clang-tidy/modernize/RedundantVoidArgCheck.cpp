//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantVoidArgCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void RedundantVoidArgCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionTypeLoc(unless(hasParent(functionDecl(isExternC())))).bind("fn"),
      this);
  Finder->addMatcher(lambdaExpr().bind("fn"), this);
}

void RedundantVoidArgCheck::check(const MatchFinder::MatchResult &Result) {
  const FunctionTypeLoc TL = [&] {
    if (const auto *TL = Result.Nodes.getNodeAs<FunctionTypeLoc>("fn"))
      return *TL;
    return Result.Nodes.getNodeAs<LambdaExpr>("fn")
        ->getCallOperator()
        ->getFunctionTypeLoc();
  }();

  if (TL.getNumParams() != 0)
    return;

  const std::optional<Token> Tok = utils::lexer::findNextTokenSkippingComments(
      Result.SourceManager->getSpellingLoc(TL.getLParenLoc()),
      *Result.SourceManager, getLangOpts());

  if (!Tok || Tok->isNot(tok::raw_identifier) ||
      Tok->getRawIdentifier() != "void")
    return;

  diag(Tok->getLocation(), "redundant void argument list")
      << FixItHint::CreateRemoval(Tok->getLocation());
}

} // namespace clang::tidy::modernize
