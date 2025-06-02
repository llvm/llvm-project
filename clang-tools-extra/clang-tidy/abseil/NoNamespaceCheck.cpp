//===--- NoNamespaceCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoNamespaceCheck.h"
#include "AbseilMatcher.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

void NoNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      namespaceDecl(hasName("::absl"), unless(isInAbseilFile()))
          .bind("abslNamespace"),
      this);
}

void NoNamespaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AbslNamespaceDecl =
      Result.Nodes.getNodeAs<NamespaceDecl>("abslNamespace");

  diag(AbslNamespaceDecl->getLocation(),
       "namespace 'absl' is reserved for implementation of the Abseil library "
       "and should not be opened in user code");
}

} // namespace clang::tidy::abseil
