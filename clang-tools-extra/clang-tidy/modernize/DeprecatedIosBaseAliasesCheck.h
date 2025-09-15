//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// This check warns the uses of the deprecated member types of
/// ``std::ios_base`` and replaces those that have a non-deprecated equivalent.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/deprecated-ios-base-aliases.html
class DeprecatedIosBaseAliasesCheck : public ClangTidyCheck {
public:
  DeprecatedIosBaseAliasesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H
