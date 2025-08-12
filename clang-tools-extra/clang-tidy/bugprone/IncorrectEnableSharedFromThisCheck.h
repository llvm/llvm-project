//===--- IncorrectEnableSharedFromThisCheck.h - clang-tidy ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTENABLESHAREDFROMTHISCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTENABLESHAREDFROMTHISCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detect classes or structs that do not publicly inherit from
/// ``std::enable_shared_from_this``, because unintended behavior will
/// otherwise occur when calling ``shared_from_this``.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/incorrect-enable-shared-from-this.html
class IncorrectEnableSharedFromThisCheck : public ClangTidyCheck {
public:
  IncorrectEnableSharedFromThisCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTENABLESHAREDFROMTHISCHECK_H
