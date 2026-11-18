//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INVALIDREGEXPATTERNCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INVALIDREGEXPATTERNCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Detects malformed regex patterns used in ``llvm::Regex``.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/regex.html
class InvalidRegexPatternCheck : public ClangTidyCheck {
public:
  InvalidRegexPatternCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INVALIDREGEXPATTERNCHECK_H
