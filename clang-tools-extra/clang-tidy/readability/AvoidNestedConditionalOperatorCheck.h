//===--- AvoidNestedConditionalOperatorCheck.h - clang-tidy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_NESTED_CONDITIONAL_OPERATOR_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_NESTED_CONDITIONAL_OPERATOR_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Identifies instances of nested conditional operators in the code.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/avoid-nested-conditional-operator.html
class AvoidNestedConditionalOperatorCheck : public ClangTidyCheck {
public:
  AvoidNestedConditionalOperatorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_NESTED_CONDITIONAL_OPERATOR_CHECK_H
