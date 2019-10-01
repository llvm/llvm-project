//===--- MultiwayPathsCoveredCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_MULTIWAY_PATHS_COVERED_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_MULTIWAY_PATHS_COVERED_H

#include "../ClangTidy.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <iostream>

namespace clang {
namespace tidy {
namespace hicpp {

/// Find occasions where not all codepaths are explicitly covered in code.
/// This includes 'switch' without a 'default'-branch and 'if'-'else if'-chains
/// without a final 'else'-branch.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp-multiway-paths-covered.html
class MultiwayPathsCoveredCheck : public ClangTidyCheck {
public:
  MultiwayPathsCoveredCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        WarnOnMissingElse(Options.get("WarnOnMissingElse", 0)) {}
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

} // namespace hicpp
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_MULTIWAY_PATHS_COVERED_H
