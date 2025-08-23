//===--- UniqueptrDeleteReleaseCheck.h - clang-tidy--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Flags statements of the form ``delete <unique_ptr expr>.release();`` and
/// replaces them with: ``<unique_ptr expr> = nullptr;``
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/uniqueptr-delete-release.html
class UniqueptrDeleteReleaseCheck : public ClangTidyCheck {
public:
  UniqueptrDeleteReleaseCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool PreferResetCall;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H
