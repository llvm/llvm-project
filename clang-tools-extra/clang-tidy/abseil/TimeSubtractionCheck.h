//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_TIMESUBTRACTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_TIMESUBTRACTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::abseil {

/// Finds and fixes `absl::Time` subtraction expressions to do subtraction
/// in the time domain instead of the numeric domain.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/time-subtraction.html
class TimeSubtractionCheck : public ClangTidyCheck {
public:
  TimeSubtractionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void emitDiagnostic(const Expr *Node, llvm::StringRef Replacement);
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_TIMESUBTRACTIONCHECK_H
