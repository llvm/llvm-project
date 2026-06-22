//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNHANDLEDCODEPATHSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNHANDLEDCODEPATHSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Find occasions where not all codepaths are explicitly covered in code.
/// This includes 'switch' without a 'default'-branch and 'if'-'else if'-chains
/// without a final 'else'-branch.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/unhandled-code-paths.html
class UnhandledCodePathsCheck : public ClangTidyCheck {
public:
  UnhandledCodePathsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        WarnOnMissingElse(Options.get("WarnOnMissingElse", false)) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void handleSwitchWithDefault(const SwitchStmt *Switch, std::size_t CaseCount);
  void handleSwitchWithoutDefault(
      const SwitchStmt *Switch, std::size_t CaseCount,
      const ast_matchers::MatchFinder::MatchResult &Result);
  /// This option can be configured to warn on missing 'else' branches in an
  /// 'if-else if' chain. The default is false because this option might be
  /// noisy on some code bases.
  const bool WarnOnMissingElse;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNHANDLEDCODEPATHSCHECK_H
