//===--- StaticAssertCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STATICASSERTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STATICASSERTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang::tidy::misc {

/// Replaces `assert()` with `static_assert()` if the condition is evaluatable
/// at compile time.
///
/// The condition of `static_assert()` is evaluated at compile time which is
/// safer and more efficient.
class StaticAssertCheck : public ClangTidyCheck {
public:
  StaticAssertCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11 || LangOpts.C11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  SourceLocation getLastParenLoc(const ASTContext *ASTCtx,
                                 SourceLocation AssertLoc);
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STATICASSERTCHECK_H
