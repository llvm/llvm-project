//===--- UninitializedThreadLocalCheck.cpp - Clang tidy tool --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINITIALIZEDTHREADLOCALCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINITIALIZEDTHREADLOCALCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::runtime {

// Finds accesses to thread_local variables that might occur prior to
// initialization.
class UninitializedThreadLocalCheck : public ClangTidyCheck {
public:
  UninitializedThreadLocalCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
};

} // namespace clang::tidy::runtime

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINITIALIZEDTHREADLOCALCHECK_H
