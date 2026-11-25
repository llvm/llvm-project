//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTZEROINITIALIZERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTZEROINITIALIZERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds explicit zero initializers of arrays that can be replaced with empty
/// braces, e.g. ``char a[12] = {0};`` becomes ``char a[12] = {};``.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/redundant-zero-initializer.html
class RedundantZeroInitializerCheck : public ClangTidyCheck {
public:
  RedundantZeroInitializerCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    // Empty braces are only valid in C++ and, for C, since C23.
    return LangOpts.CPlusPlus || LangOpts.C23;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTZEROINITIALIZERCHECK_H
