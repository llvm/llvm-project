//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "hide_from_abi.hpp"

namespace libcpp {
hide_from_abi::hide_from_abi(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void hide_from_abi::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(
      functionDecl(
          unless(anyOf(
              // These functions can't be marked `[[gnu::always_inline]]` for various reasons,
              // so we can't mark them `_LIBCPP_HIDE_FROM_ABI`. These functions are ignored in
              // all namespaces. Checking the qualified name is a lot harder and these names
              // should result in very few (if any) false-negatives. This is also just a
              // temporary work-around until we can mark functions as HIDE_FROM_ABI without
              // having to add `[[gnu::always_inline]]` with GCC.
              hasAnyName("__introsort",
                         "__inplace_merge",
                         "__libcpp_snprintf_l",
                         "__libcpp_asprintf_l",
                         "__libcpp_sscanf_l",
                         "__tree_sub_invariant",
                         "__stable_sort_move",
                         "__stable_sort",
                         "__stable_partition",
                         "__lock_first",
                         "__stable_partition_impl"),
              hasAttr(clang::attr::Visibility),
              hasAttr(clang::attr::AbiTag),
              cxxMethodDecl(), // We have explicitly instantiated classes and some of their methods don't have these attributes
              isDeleted(),
              isConsteval())),
          isDefinition())
          .bind("missing_hide_from_abi"),
      this);
}

void hide_from_abi::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs<clang::FunctionDecl>("missing_hide_from_abi"); call != nullptr) {
    diag(call->getBeginLoc(), "_LIBCPP_HIDE_FROM_ABI is missing");
  }
}
} // namespace libcpp
