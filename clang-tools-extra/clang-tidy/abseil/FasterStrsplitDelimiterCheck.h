//===--- FasterStrsplitDelimiterCheck.h - clang-tidy-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::abseil {

/// Finds instances of absl::StrSplit() or absl::MaxSplits() where the delimiter
/// is a single character string literal and replaces it with a character.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/faster-strsplit-delimiter.html
class FasterStrsplitDelimiterCheck : public ClangTidyCheck {
public:
  FasterStrsplitDelimiterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H
