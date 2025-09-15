//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UPGRADEGOOGLETESTCASECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UPGRADEGOOGLETESTCASECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google {

/// Finds uses of deprecated Googletest APIs with names containing "case" and
/// replaces them with equivalent names containing "suite".
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/google/upgrade-googletest-case.html
class UpgradeGoogletestCaseCheck : public ClangTidyCheck {
public:
  UpgradeGoogletestCaseCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::DenseSet<SourceLocation> MatchedTemplateLocations;
};

} // namespace clang::tidy::google

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UPGRADEGOOGLETESTCASECHECK_H
