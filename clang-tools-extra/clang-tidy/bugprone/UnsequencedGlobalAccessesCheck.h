//===--- UnsequencedGlobalAccessesCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSEQUENCEDGLOBALACCESSES\
CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSEQUENCEDGLOBALACCESSES\
CHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds conflicting accesses on global variables.
class UnsequencedGlobalAccessesCheck : public ClangTidyCheck {
public:
  UnsequencedGlobalAccessesCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  bool HandleMutableFunctionParametersAsWrites;
};

} // namespace clang::tidy::bugprone

#endif
