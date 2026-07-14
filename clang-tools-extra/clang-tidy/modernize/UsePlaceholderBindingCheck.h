//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEPLACEHOLDERBINDINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEPLACEHOLDERBINDINGCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds structured bindings where a binding is only used to suppress an
/// "unused variable" warning via a ``(void)name;`` statement, and suggests
/// replacing the binding with a C++26 placeholder (``_``) instead.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-placeholder-binding.html
class UsePlaceholderBindingCheck : public ClangTidyCheck {
public:
  UsePlaceholderBindingCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus26;
  }
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEPLACEHOLDERBINDINGCHECK_H
