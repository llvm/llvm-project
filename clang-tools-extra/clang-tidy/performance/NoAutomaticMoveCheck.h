//===--- NoAutomaticMoveCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOAUTOMATICMOVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOAUTOMATICMOVECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Finds local variables that cannot be automatically moved due to constness.
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/no-automatic-move.html
class NoAutomaticMoveCheck : public ClangTidyCheck {
public:
  NoAutomaticMoveCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<StringRef> AllowedTypes;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOAUTOMATICMOVECHECK_H
