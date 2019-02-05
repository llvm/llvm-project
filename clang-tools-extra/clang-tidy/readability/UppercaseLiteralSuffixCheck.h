//===--- UppercaseLiteralSuffixCheck.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UPPERCASELITERALSUFFIXCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UPPERCASELITERALSUFFIXCHECK_H

#include "../ClangTidy.h"
#include "../utils/OptionsUtils.h"

namespace clang {
namespace tidy {
namespace readability {

/// Detects when the integral literal or floating point literal has
/// non-uppercase suffix, and suggests to make the suffix uppercase.
/// Alternatively, a list of destination suffixes can be provided.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-uppercase-literal-suffix.html
class UppercaseLiteralSuffixCheck : public ClangTidyCheck {
public:
  UppercaseLiteralSuffixCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  template <typename LiteralType>
  bool checkBoundMatch(const ast_matchers::MatchFinder::MatchResult &Result);

  const std::vector<std::string> NewSuffixes;
  const bool IgnoreMacros;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UPPERCASELITERALSUFFIXCHECK_H
