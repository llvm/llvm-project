//===--- UnusedAliasDeclsCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnusedAliasDeclsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void UnusedAliasDeclsCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++11; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus11)
    return;

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
          Result.Nodes.getNodeAs<NestedNameSpecifier>("nns")) {
    if (const auto *AliasDecl = NestedName->getAsNamespaceAlias()) {
      FoundDecls[AliasDecl] = CharSourceRange();
    }
  }
}

void UnusedAliasDeclsCheck::onEndOfTranslationUnit() {
  for (const auto &FoundDecl : FoundDecls) {
    if (!FoundDecl.second.isValid())
      continue;
    diag(FoundDecl.first->getLocation(), "namespace alias decl %0 is unused")
        << FoundDecl.first << FixItHint::CreateRemoval(FoundDecl.second);
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
