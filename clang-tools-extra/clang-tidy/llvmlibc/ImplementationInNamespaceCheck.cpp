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

namespace clang::tidy::llvm_libc {

const static StringRef RequiredNamespaceStart = "__llvm_libc";
const static StringRef RequiredNamespaceMacroName = "LIBC_NAMESPACE";

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
    if (!Result.SourceManager->isMacroBodyExpansion(NS->getLocation()))
      diag(NS->getLocation(),
           "the outermost namespace should be the '%0' macro")
          << RequiredNamespaceMacroName;
    else if (!NS->getName().starts_with(RequiredNamespaceStart))
      diag(NS->getLocation(), "the outermost namespace should start with '%0'")
          << RequiredNamespaceStart;
    return;
  }
  diag(MatchedDecl->getLocation(),
       "declaration must be declared within a namespace starting with '%0'")
      << RequiredNamespaceStart;
}

} // namespace clang::tidy::llvm_libc
