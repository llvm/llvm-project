//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NOSUSPENDWITHLOCKCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NOSUSPENDWITHLOCKCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Flag coroutines that suspend while any lock guard is alive.
/// This check implements CppCoreGuideline CP.52.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/no-suspend-with-lock.html
class NoSuspendWithLockCheck : public ClangTidyCheck {
public:
  NoSuspendWithLockCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        LockGuards(Options.get("LockGuards",
                               "::std::unique_lock;::std::scoped_lock;::"
                               "std::shared_lock;::std::lock_guard")) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20;
  }

private:
  /// Semicolon-separated list of fully qualified names of lock guard template
  /// types. Defaults to
  /// `::std::unique_lock;::std::scoped_lock;::std::shared_lock;::std::lock_guard`.
  const StringRef LockGuards;
};

} // namespace clang::tidy::cppcoreguidelines

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NOSUSPENDWITHLOCKCHECK_H
