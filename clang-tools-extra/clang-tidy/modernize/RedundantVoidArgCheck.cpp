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
using namespace clang::ast_matchers::internal;

namespace clang::tidy::modernize {

void RedundantVoidArgCheck::registerMatchers(MatchFinder *Finder) {
  const VariadicDynCastAllOfMatcher<TypeLoc, FunctionProtoTypeLoc>
      functionProtoTypeLoc; // NOLINT(readability-identifier-naming)
  Finder->addMatcher(traverse(TK_IgnoreUnlessSpelledInSource,
                              functionProtoTypeLoc(
                                  unless(hasParent(functionDecl(isExternC()))))
                                  .bind("fn")),
                     this);
  Finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource, lambdaExpr().bind("fn")), this);
}

void RedundantVoidArgCheck::check(const MatchFinder::MatchResult &Result) {
  const FunctionProtoTypeLoc TL = [&] {
    if (const auto *TL = Result.Nodes.getNodeAs<FunctionProtoTypeLoc>("fn"))
      return *TL;
    return Result.Nodes.getNodeAs<LambdaExpr>("fn")
        ->getCallOperator()
        ->getTypeSourceInfo()
        ->getTypeLoc()
        .getAs<FunctionProtoTypeLoc>();
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
