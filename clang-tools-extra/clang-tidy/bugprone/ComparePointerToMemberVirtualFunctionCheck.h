//=--- ComparePointerToMemberVirtualFunctionCheck.h - clang-tidy --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COMPAREPOINTERTOMEMBERVIRTUALFUNCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COMPAREPOINTERTOMEMBERVIRTUALFUNCTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detects unspecified behavior about equality comparison between pointer to
/// member virtual function and anything other than null-pointer-constant.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/compare-pointer-to-member-virtual-function.html
class ComparePointerToMemberVirtualFunctionCheck : public ClangTidyCheck {
public:
  ComparePointerToMemberVirtualFunctionCheck(StringRef Name,
                                             ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_COMPAREPOINTERTOMEMBERVIRTUALFUNCTIONCHECK_H
