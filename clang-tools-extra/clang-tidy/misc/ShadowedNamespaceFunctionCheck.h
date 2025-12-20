//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SHADOWEDNAMESPACEFUNCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SHADOWEDNAMESPACEFUNCTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Detects free functions in global namespace that shadow functions from other
/// namespaces.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/shadowed-namespace-function.html
class ShadowedNamespaceFunctionCheck : public ClangTidyCheck {
public:
  ShadowedNamespaceFunctionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreTemplated(
            Options.get("IgnoreTemplated", false)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  const bool IgnoreTemplated;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SHADOWEDNAMESPACEFUNCTIONCHECK_H
