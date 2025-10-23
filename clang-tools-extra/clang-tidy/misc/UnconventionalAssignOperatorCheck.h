//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSIGNOPERATORSIGNATURECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSIGNOPERATORSIGNATURECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Finds declarations of assignment operators with the wrong return and/or
/// argument types and definitions with good return type but wrong return
/// statements.
///
///   * The return type must be `Class&`.
///   * Works with move-assign and assign by value.
///   * Private and deleted operators are ignored.
///   * The operator must always return ``*this``.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/unconventional-assign-operator.html
class UnconventionalAssignOperatorCheck : public ClangTidyCheck {
public:
  UnconventionalAssignOperatorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSIGNOPERATORSIGNATURECHECK_H
