//===--- MoveConstArgCheck.h - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseSet.h"

namespace clang::tidy::performance {

/// Find casts of calculation results to bigger type. Typically from int to
///
/// The options are
///
///   - `CheckTriviallyCopyableMove`: Whether to check for trivially-copyable
//      types as their objects are not moved but copied. Enabled by default.
//    - `CheckMoveToConstRef`: Whether to check if a `std::move()` is passed
//      as a const reference argument.
class MoveConstArgCheck : public ClangTidyCheck {
public:
  MoveConstArgCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context), CheckTriviallyCopyableMove(Options.get(
                                           "CheckTriviallyCopyableMove", true)),
        CheckMoveToConstRef(Options.get("CheckMoveToConstRef", true)) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckTriviallyCopyableMove;
  const bool CheckMoveToConstRef;
  llvm::DenseSet<const CallExpr *> AlreadyCheckedMoves;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
