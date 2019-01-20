//===--- MoveConstArgCheck.h - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace performance {

/// Find casts of calculation results to bigger type. Typically from int to
///
/// There is one option:
///
///   - `CheckTriviallyCopyableMove`: Whether to check for trivially-copyable
//      types as their objects are not moved but copied. Enabled by default.
class MoveConstArgCheck : public ClangTidyCheck {
public:
  MoveConstArgCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        CheckTriviallyCopyableMove(
            Options.get("CheckTriviallyCopyableMove", true)) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckTriviallyCopyableMove;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
