//===--- DontModifyStdNamespaceCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DontModifyStdNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void DontModifyStdNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      namespaceDecl(unless(isExpansionInSystemHeader()),
                    hasAnyName("std", "posix"),
                    has(decl(unless(anyOf(
                        functionDecl(isExplicitTemplateSpecialization()),
                        cxxRecordDecl(isExplicitTemplateSpecialization()))))))
          .bind("nmspc"),
      this);
}

void DontModifyStdNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *N = Result.Nodes.getNodeAs<NamespaceDecl>("nmspc");

  // Only consider top level namespaces.
  if (N->getParent() != Result.Context->getTranslationUnitDecl())
    return;

  diag(N->getLocation(),
       "modification of %0 namespace can result in undefined behavior")
      << N;
}

} // namespace cert
} // namespace tidy
} // namespace clang
