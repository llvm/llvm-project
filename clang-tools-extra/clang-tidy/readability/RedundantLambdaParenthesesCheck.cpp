//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantLambdaParenthesesCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace {

AST_MATCHER(clang::LambdaExpr, hasNoParameters) {
  return Node.getCallOperator()->getNumParams() == 0;
}

AST_MATCHER(clang::LambdaExpr, hasExplicitOrGenericParameters) {
  return Node.hasExplicitParameters() || Node.isGenericLambda();
}

AST_MATCHER(clang::LambdaExpr, isGenericLambdaInCxx20OrLater) {
  return !Node.isGenericLambda() ||
         Finder->getASTContext().getLangOpts().CPlusPlus20;
}

} // namespace

namespace clang::tidy::readability {

void RedundantLambdaParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      lambdaExpr(hasExplicitOrGenericParameters(), hasNoParameters(),
                 isGenericLambdaInCxx20OrLater())
          .bind("lambda"),
      this);
}

void RedundantLambdaParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");


  if (Lambda->getBeginLoc().isMacroID())
    return;

  const LangOptions &LangOpts = getLangOpts();

  const SourceLocation ScanFrom =
      Lambda->isGenericLambda()
          ? Lexer::getLocForEndOfToken(
                Lambda->getTemplateParameterList()->getRAngleLoc(), 0,
                *Result.SourceManager, LangOpts)
          : Lexer::getLocForEndOfToken(
                Lambda->getIntroducerRange().getEnd(), 0,
                *Result.SourceManager, LangOpts);

  Token Tok;
  if (Lexer::getRawToken(ScanFrom, Tok, *Result.SourceManager, LangOpts,
                         /*IgnoreWhiteSpace=*/true))
    return;

  if (Tok.isNot(tok::l_paren))
    return;

  const SourceLocation LParenLoc = Tok.getLocation();
  const SourceLocation RParenLoc = Lexer::findLocationAfterToken(
      LParenLoc, tok::r_paren, *Result.SourceManager, LangOpts,
      /*SkipTrailingWhitespaceAndNewLine=*/false);

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

  const CharSourceRange ParenRange =
      CharSourceRange::getCharRange(LParenLoc, RParenLoc);

  diag(LParenLoc, "redundant empty parameter list in lambda expression")
      << FixItHint::CreateRemoval(ParenRange);
}

} // namespace clang::tidy::readability