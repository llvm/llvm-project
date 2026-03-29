//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qualify_declval.hpp"

namespace {
AST_MATCHER(clang::UnresolvedLookupExpr, requiresADL) { return Node.requiresADL(); }
AST_MATCHER(clang::UnresolvedLookupExpr, isDeclval) { return Node.getName().getAsString() == "declval"; }
} // namespace

namespace libcpp {
qualify_declval::qualify_declval(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void qualify_declval::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(callExpr(has(unresolvedLookupExpr(requiresADL(), isDeclval()))).bind("ADLcall"), this);
}

void qualify_declval::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs<clang::CallExpr>("ADLcall"); call != nullptr) {
    diag(call->getBeginLoc(), "declval should be qualified to get better error messages")
        << clang::FixItHint::CreateInsertion(call->getBeginLoc(), "std::");
  }
}
} // namespace libcpp
