//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedAliasDeclsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void UnusedAliasDeclsCheck::registerMatchers(MatchFinder *Finder) {
  // We cannot do anything about headers (yet), as the alias declarations
  // used in one header could be used by some other translation unit.
  Finder->addMatcher(namespaceAliasDecl(isExpansionInMainFile()).bind("alias"),
                     this);
  Finder->addMatcher(nestedNameSpecifier().bind("nns"), this);
}

void UnusedAliasDeclsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *AliasDecl = Result.Nodes.getNodeAs<NamedDecl>("alias")) {
    FoundDecls[AliasDecl] = CharSourceRange::getCharRange(
        AliasDecl->getBeginLoc(),
        Lexer::findLocationAfterToken(
            AliasDecl->getEndLoc(), tok::semi, *Result.SourceManager,
            getLangOpts(),
            /*SkipTrailingWhitespaceAndNewLine=*/true));
    return;
  }

  if (const auto *NestedName =
          Result.Nodes.getNodeAs<NestedNameSpecifier>("nns");
      NestedName &&
      NestedName->getKind() == NestedNameSpecifier::Kind::Namespace)
    if (const auto *AliasDecl = dyn_cast<NamespaceAliasDecl>(
            NestedName->getAsNamespaceAndPrefix().Namespace))
      FoundDecls[AliasDecl] = CharSourceRange();
}

void UnusedAliasDeclsCheck::onEndOfTranslationUnit() {
  for (const auto &FoundDecl : FoundDecls) {
    if (!FoundDecl.second.isValid())
      continue;
    diag(FoundDecl.first->getLocation(), "namespace alias decl %0 is unused")
        << FoundDecl.first << FixItHint::CreateRemoval(FoundDecl.second);
  }
}

} // namespace clang::tidy::misc
