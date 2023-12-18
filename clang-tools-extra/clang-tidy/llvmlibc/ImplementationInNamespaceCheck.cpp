//===--- ImplementationInNamespaceCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplementationInNamespaceCheck.h"
#include "NamespaceConstants.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_libc {

void ImplementationInNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      translationUnitDecl(
          forEach(decl(isExpansionInMainFile(), unless(linkageSpecDecl()),
                       // anonymous namespaces generate usingDirective
                       unless(usingDirectiveDecl(isImplicit())))
                      .bind("child_of_translation_unit"))),
      this);
}

void ImplementationInNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<Decl>("child_of_translation_unit");
  const auto *NS = dyn_cast<NamespaceDecl>(MatchedDecl);
  if (NS == nullptr || NS->isAnonymousNamespace()) {
    diag(MatchedDecl->getLocation(),
         "declaration must be enclosed within the '%0' namespace")
        << RequiredNamespaceMacroName;
    return;
  }
  if (Result.SourceManager->isMacroBodyExpansion(NS->getLocation()) == false) {
    diag(NS->getLocation(), "the outermost namespace should be the '%0' macro")
        << RequiredNamespaceMacroName;
    return;
  }
  if (NS->getName().starts_with(RequiredNamespaceStart) == false) {
    diag(NS->getLocation(), "the '%0' macro should start with '%1'")
        << RequiredNamespaceMacroName << RequiredNamespaceStart;
    return;
  }
}

} // namespace clang::tidy::llvm_libc
