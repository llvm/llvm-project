//===--- NoexceptSwapCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/ExceptionSpecAnalyzer.h"

namespace clang::tidy::performance {

/// The check flags swap functions not marked with `noexcept` or marked
/// with `noexcept(expr)` where `expr` evaluates to `false`
/// (but is not a `false` literal itself).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/noexcept-swap.html
class NoexceptSwapCheck : public ClangTidyCheck {
public:
  NoexceptSwapCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11 && LangOpts.CXXExceptions;
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  utils::ExceptionSpecAnalyzer SpecAnalyzer;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H
