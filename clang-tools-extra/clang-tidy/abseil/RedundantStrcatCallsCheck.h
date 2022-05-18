//===--- RedundantStrcatCallsCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace abseil {

/// Flags redundant calls to absl::StrCat when the result is being passed to 
///	another call of absl::StrCat/absl::StrAppend. Also suggests a fix to 
///	collapse the calls.
/// Example:
///   StrCat(1, StrCat(2, 3))  ==>  StrCat(1, 2, 3)
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/redundant-strcat-calls.html
class RedundantStrcatCallsCheck : public ClangTidyCheck {
public:
  RedundantStrcatCallsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H
