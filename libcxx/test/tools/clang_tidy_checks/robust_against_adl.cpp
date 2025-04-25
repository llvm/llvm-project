//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "robust_against_adl.hpp"

#include <algorithm>

namespace {
AST_MATCHER(clang::UnresolvedLookupExpr, requiresADL) { return Node.requiresADL(); }

AST_MATCHER(clang::CallExpr, isOperator) { return llvm::isa<clang::CXXOperatorCallExpr>(Node); }

AST_MATCHER(clang::UnresolvedLookupExpr, isCustomizationPoint) {
  return std::ranges::any_of(
      std::array{"swap", "make_error_code", "make_error_condition", "begin", "end", "size", "rend", "rbegin"},
      [&](const char* func) { return Node.getName().getAsString() == func; });
}

AST_MATCHER(clang::CXXMethodDecl, isStatic) { return Node.isStatic(); }

} // namespace

namespace libcpp {
robust_against_adl_check::robust_against_adl_check(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void robust_against_adl_check::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(
      callExpr(unless(isOperator()),
               unless(argumentCountIs(0)),
               has(unresolvedLookupExpr(requiresADL(), unless(isCustomizationPoint()))),
               unless(callee(cxxMethodDecl(isStatic()))))
          .bind("ADLcall"),
      this);
}

void robust_against_adl_check::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs<clang::CallExpr>("ADLcall"); call != nullptr) {
    diag(call->getBeginLoc(), "ADL lookup");
  }
}
} // namespace libcpp
