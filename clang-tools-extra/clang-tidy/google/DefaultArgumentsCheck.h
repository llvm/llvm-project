//===--- DefaultArgumentsCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google {

/// Checks that default parameters are not given for virtual methods.
///
/// See https://google.github.io/styleguide/cppguide.html#Default_Arguments
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/default-arguments.html
class DefaultArgumentsCheck : public ClangTidyCheck {
public:
  DefaultArgumentsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::google

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H
