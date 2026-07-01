//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "empty_namespaces.hpp"

namespace libcpp {
empty_namespaces::empty_namespaces(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void empty_namespaces::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(namespaceDecl(unless(hasDescendant(decl()))).bind("empty_namespace"), this);
}

void empty_namespaces::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* ns = result.Nodes.getNodeAs<clang::NamespaceDecl>("empty_namespace")) {
    diag(ns->getBeginLoc(), "Empty namespaces should be avoided. Move any checks around the namespace instead.");
  }
}
} // namespace libcpp
