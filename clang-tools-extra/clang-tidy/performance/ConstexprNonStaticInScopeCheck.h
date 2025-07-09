//===--- ConstexprNonStaticInScopeCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTEXPRNONSTATICINSCOPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTEXPRNONSTATICINSCOPECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/constexpr-non-static-in-scope.html
class ConstexprNonStaticInScopeCheck : public ClangTidyCheck {
public:
  ConstexprNonStaticInScopeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        WarnInConstexprFuncCpp23(Options.get("WarnInConstexprFuncCpp23", /*Default=*/true)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
private:
  const bool WarnInConstexprFuncCpp23;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTEXPRNONSTATICINSCOPECHECK_H
