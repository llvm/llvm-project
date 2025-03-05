//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"

#include "nodebug_on_aliases.hpp"
#include "utilities.hpp"

namespace libcpp {
namespace {
AST_MATCHER(clang::NamedDecl, isPretty) { return !is_ugly_name(Node.getName()); }
} // namespace

nodebug_on_aliases::nodebug_on_aliases(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void nodebug_on_aliases::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(
      typeAliasDecl(unless(anyOf(isPretty(), hasAttr(clang::attr::NoDebug), hasAncestor(functionDecl()))))
          .bind("nodebug_on_internal_aliases"),
      this);
}

void nodebug_on_aliases::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* alias = result.Nodes.getNodeAs<clang::TypeAliasDecl>("nodebug_on_internal_aliases")) {
    diag(alias->getBeginLoc(), "Internal aliases should always be marked _LIBCPP_NODEBUG");
  }
}
} // namespace libcpp
