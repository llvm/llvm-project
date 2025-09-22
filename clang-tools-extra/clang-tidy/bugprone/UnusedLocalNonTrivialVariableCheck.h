//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDLOCALNONTRIVIALVARIABLECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDLOCALNONTRIVIALVARIABLECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Warns when a local non trivial variable is unused within a function. By
/// default std::.*mutex and std::future are included.
///
/// The check supports these options:
///   - 'IncludeTypes': a semicolon-separated list of regular expressions
///                     matching types to ensure must be used.
///   - 'ExcludeTypes': a semicolon-separated list of regular expressions
///                     matching types that are excluded from the
///                     'IncludeTypes' matches.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/unused-local-non-trivial-variable.html
class UnusedLocalNonTrivialVariableCheck : public ClangTidyCheck {
public:
  UnusedLocalNonTrivialVariableCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;

private:
  const std::vector<StringRef> IncludeTypes;
  const std::vector<StringRef> ExcludeTypes;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDLOCALNONTRIVIALVARIABLECHECK_H
