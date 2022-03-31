//===--- ImplementationInNamespaceCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplementationInNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace llvm_libc {

const static StringRef RequiredNamespace = "__llvm_libc";
void ImplementationInNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      decl(hasParent(translationUnitDecl()), unless(linkageSpecDecl()))
          .bind("child_of_translation_unit"),
      this);
}

void ImplementationInNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<Decl>("child_of_translation_unit");
  if (!Result.SourceManager->isInMainFile(MatchedDecl->getLocation()))
    return;

  if (const auto *NS = dyn_cast<NamespaceDecl>(MatchedDecl)) {
    if (NS->getName() != RequiredNamespace) {
      diag(NS->getLocation(), "'%0' needs to be the outermost namespace")
          << RequiredNamespace;
    }
    return;
  }
  diag(MatchedDecl->getLocation(),
       "declaration must be declared within the '%0' namespace")
      << RequiredNamespace;
}

} // namespace llvm_libc
} // namespace tidy
} // namespace clang
