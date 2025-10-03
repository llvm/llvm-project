//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StaticDefinitionInAnonymousNamespaceCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void StaticDefinitionInAnonymousNamespaceCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      namedDecl(anyOf(functionDecl(isDefinition(), isStaticStorageClass()),
                      varDecl(isDefinition(), isStaticStorageClass())),
                isInAnonymousNamespace())
          .bind("static-def"),
      this);
}

void StaticDefinitionInAnonymousNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Def = Result.Nodes.getNodeAs<NamedDecl>("static-def");
  // Skips all static definitions defined in Macro.
  if (Def->getLocation().isMacroID())
    return;

  // Skips all static definitions in function scope.
  const DeclContext *DC = Def->getDeclContext();
  if (DC->getDeclKind() != Decl::Namespace)
    return;

  auto Diag =
      diag(Def->getLocation(), "%0 is a static definition in "
                               "anonymous namespace; static is redundant here")
      << Def;
  Token Tok;
  SourceLocation Loc = Def->getSourceRange().getBegin();
  while (Loc < Def->getSourceRange().getEnd() &&
         !Lexer::getRawToken(Loc, Tok, *Result.SourceManager, getLangOpts(),
                             true)) {
    SourceRange TokenRange(Tok.getLocation(), Tok.getEndLoc());
    StringRef SourceText =
        Lexer::getSourceText(CharSourceRange::getTokenRange(TokenRange),
                             *Result.SourceManager, getLangOpts());
    if (SourceText == "static") {
      Diag << FixItHint::CreateRemoval(TokenRange);
      break;
    }
    Loc = Tok.getEndLoc();
  }
}

} // namespace clang::tidy::readability
