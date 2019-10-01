//===--- SuspiciousMissingCommaCheck.h - clang-tidy--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSMISSINGCOMMACHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSMISSINGCOMMACHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// This check finds string literals which are probably concatenated
/// accidentally.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-suspicious-missing-comma.html
class SuspiciousMissingCommaCheck : public ClangTidyCheck {
public:
  SuspiciousMissingCommaCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // Minimal size of a string literals array to be considered by the checker.
  const unsigned SizeThreshold;
  // Maximal threshold ratio of suspicious string literals to be considered.
  const double RatioThreshold;
  // Maximal number of concatenated tokens.
  const unsigned MaxConcatenatedTokens;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSMISSINGCOMMACHECK_H
