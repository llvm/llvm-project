//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTPARENTHESESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTPARENTHESESCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Basic/LangOptions.h"

namespace clang::tidy::readability {

/// Detect redundant parentheses.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/redundant-parentheses.html
class RedundantParenthesesCheck : public ClangTidyCheck {
public:
  RedundantParenthesesCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus | LangOpts.C99;
  }

private:
  const std::vector<StringRef> AllowedDecls;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTPARENTHESESCHECK_H
