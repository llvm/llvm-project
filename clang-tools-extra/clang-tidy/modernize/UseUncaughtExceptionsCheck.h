//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// This check will warn on calls to std::uncaught_exception and replace them
/// with calls to std::uncaught_exceptions, since std::uncaught_exception was
/// deprecated in C++17. In case of macro ID there will be only a warning
/// without fixits.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-uncaught-exceptions.html
class UseUncaughtExceptionsCheck : public ClangTidyCheck {
public:
  UseUncaughtExceptionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H
