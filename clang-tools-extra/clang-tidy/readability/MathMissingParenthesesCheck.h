//===--- MathMissingParenthesesCheck.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MATHMISSINGPARENTHESESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MATHMISSINGPARENTHESESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Check for mising parantheses in mathematical expressions that involve
/// operators of different priorities.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/math-missing-parentheses.html
class MathMissingParenthesesCheck : public ClangTidyCheck {
public:
  MathMissingParenthesesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MATHMISSINGPARENTHESESCHECK_H
