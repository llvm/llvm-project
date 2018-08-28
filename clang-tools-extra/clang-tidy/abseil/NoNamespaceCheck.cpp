//===--- NoNamespaceCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoNamespaceCheck.h"
#include "AbseilMatcher.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void NoNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      namespaceDecl(hasName("::absl"), unless(isInAbseilFile()))
          .bind("abslNamespace"),
      this);
}

void NoNamespaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *abslNamespaceDecl =
      Result.Nodes.getNodeAs<NamespaceDecl>("abslNamespace");

  diag(abslNamespaceDecl->getLocation(),
       "namespace 'absl' is reserved for implementation of the Abseil library "
       "and should not be opened in user code");
}

} // namespace abseil
} // namespace tidy
} // namespace clang
