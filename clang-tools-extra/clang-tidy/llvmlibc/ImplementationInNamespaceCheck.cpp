//===--- ImplementationInNamespaceCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplementationInNamespaceCheck.h"
#include "NamespaceConstants.h"
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

  // LLVM libc declarations should be inside of a non-anonymous namespace.
  if (NS == nullptr || NS->isAnonymousNamespace()) {
    diag(MatchedDecl->getLocation(),
         "declaration must be enclosed within the '%0' namespace")
        << RequiredNamespaceDeclMacroName;
    return;
  }

  // Enforce that the namespace is the result of macro expansion
  if (Result.SourceManager->isMacroBodyExpansion(NS->getLocation()) == false) {
    diag(NS->getLocation(), "the outermost namespace should be the '%0' macro")
        << RequiredNamespaceDeclMacroName;
    return;
  }

  // We want the macro to have [[gnu::visibility("hidden")]] as a prefix, but
  // visibility is just an attribute in the AST construct, so we check that
  // instead.
  if (NS->getVisibility() != Visibility::HiddenVisibility) {
    diag(NS->getLocation(), "the '%0' macro should start with '%1'")
        << RequiredNamespaceDeclMacroName << RequiredNamespaceDeclStart;
    return;
  }

  // Lastly, make sure the namespace name actually has the __llvm_libc prefix
  if (NS->getName().starts_with(RequiredNamespaceRefStart) == false) {
    diag(NS->getLocation(), "the '%0' macro expansion should start with '%1'")
        << RequiredNamespaceDeclMacroName << RequiredNamespaceRefStart;
    return;
  }
}

} // namespace clang::tidy::llvm_libc
