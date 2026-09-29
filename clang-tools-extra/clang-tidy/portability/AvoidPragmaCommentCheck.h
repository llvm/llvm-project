//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMACOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMACOMMENTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::portability {

/// Finds uses of ``#pragma comment`` and for ``lib`` or ``linker`` comments
/// suggests using the build system for improved portability.
///
/// Only the "lib" pragma comment type is implemented on Linux, the rest are
/// Windows-only and should be caught by "-Wunknown-pragmas" on Linux.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/portability/avoid-pragma-comment.html
class AvoidPragmaCommentCheck : public ClangTidyCheck {
public:
  AvoidPragmaCommentCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus || LangOpts.C99;
  }
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPRAGMACOMMENTCHECK_H
