//===--- ExplicitMoveConstructorCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExplicitMoveConstructorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static SourceRange findExplicitToken(const CXXConstructorDecl *Ctor,
                                     const SourceManager &Source,
                                     const LangOptions &LangOpts) {
  SourceLocation CurrentLoc = Ctor->getBeginLoc();
  const SourceLocation EndLoc = Ctor->getEndLoc();
  Token Tok;

  do {
    const bool failed = Lexer::getRawToken(CurrentLoc, Tok, Source, LangOpts);

    if (failed)
      return {};

    if (Tok.is(tok::raw_identifier) && Tok.getRawIdentifier() == "explicit")
      return {Tok.getLocation(), Tok.getEndLoc()};

    CurrentLoc = Tok.getEndLoc();
  } while (Tok.isNot(tok::eof) && CurrentLoc < EndLoc);

  return {};
}

void ExplicitMoveConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          cxxRecordDecl(
              has(cxxConstructorDecl(isMoveConstructor(), isExplicit(),
                                     unless(isDeleted()))
                      .bind("move-ctor")),
              has(cxxConstructorDecl(isCopyConstructor(), unless(isDeleted()))
                      .bind("copy-ctor")),
              unless(isExpansionInSystemHeader()))),
      this);
}

void ExplicitMoveConstructorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MoveCtor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("move-ctor");
  const auto *CopyCtor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("copy-ctor");

  if (!MoveCtor || !CopyCtor)
    return;

  auto Diag =
      diag(MoveCtor->getLocation(),
           "copy constructor may be called instead of move constructor");
  const SourceRange ExplicitTokenRange =
      findExplicitToken(MoveCtor, *Result.SourceManager, getLangOpts());

  if (ExplicitTokenRange.isInvalid())
    return;

  Diag << FixItHint::CreateRemoval(
      CharSourceRange::getCharRange(ExplicitTokenRange));
}

} // namespace clang::tidy::performance
