//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantLambdaParenthesesCheck.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(LambdaExpr, hasRedundantParens) {
  return (Node.hasExplicitParameters() || Node.isGenericLambda()) &&
         Node.getCallOperator()->getNumParams() == 0 &&
         (!Node.isGenericLambda() ||
          Finder->getASTContext().getLangOpts().CPlusPlus20);
}

} // namespace

void RedundantLambdaParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasRedundantParens()).bind("lambda"), this);
}

void RedundantLambdaParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");

  if (Lambda->getBeginLoc().isMacroID())
    return;

  const LangOptions &LangOpts = getLangOpts();

  const auto FTL = Lambda->getCallOperator()->getFunctionTypeLoc();
  const SourceLocation LParenLoc = FTL.getLParenLoc();
  const SourceLocation RParenLoc = FTL.getRParenLoc();

  if (LParenLoc.isInvalid() || RParenLoc.isInvalid())
    return;

  if (!LangOpts.CPlusPlus23) {
    std::optional<Token> RParen =
        Lexer::findNextToken(LParenLoc, *Result.SourceManager, LangOpts);
    if (!RParen || RParen->isNot(tok::r_paren))
      return;
    std::optional<Token> NextTok = Lexer::findNextToken(
        RParen->getLocation(), *Result.SourceManager, LangOpts);
    if (NextTok && NextTok->is(tok::raw_identifier)) {
      StringRef Id = NextTok->getRawIdentifier();
      if (Id == "constexpr" || Id == "consteval" || Id == "mutable" ||
          Id == "noexcept")
        return;
    }
    if (NextTok && NextTok->is(tok::arrow))
      return;
  }

  diag(LParenLoc, "redundant empty parameter list in lambda expression")
      << FixItHint::CreateRemoval({LParenLoc, RParenLoc});
}

} // namespace clang::tidy::readability
