//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFETOALLOWEXCEPTIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFETOALLOWEXCEPTIONSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds functions where throwing exceptions is unsafe but the function is
/// still marked as potentially throwing.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/unsafe-to-allow-exceptions.html
class UnsafeToAllowExceptionsCheck : public ClangTidyCheck {
public:
  UnsafeToAllowExceptionsCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && LangOpts.CXXExceptions;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::vector<llvm::StringRef> CheckedSwapFunctions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFETOALLOWEXCEPTIONSCHECK_H
