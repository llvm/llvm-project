//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_REDUNDANTCASTINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_REDUNDANTCASTINGCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Detect redundant uses of LLVM's cast and dyn_cast functions.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/redundant-casting.html
class RedundantCastingCheck : public ClangTidyCheck {
public:
  RedundantCastingCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    // Casts can be redundant for some instantiations but not others.
    // Only emit warnings in templates in the uninstantated versions.
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_REDUNDANTCASTINGCHECK_H
