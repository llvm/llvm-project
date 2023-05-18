//===--- MisplacedWideningCastCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACEDWIDENINGCASTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACEDWIDENINGCASTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Find casts of calculation results to bigger type. Typically from int to
/// long. If the intention of the cast is to avoid loss of precision then
/// the cast is misplaced, and there can be loss of precision. Otherwise
/// such cast is ineffective.
///
/// There is one option:
///
///   - `CheckImplicitCasts`: Whether to check implicit casts as well which may
//      be the most common case. Enabled by default.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/misplaced-widening-cast.html
class MisplacedWideningCastCheck : public ClangTidyCheck {
public:
  MisplacedWideningCastCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckImplicitCasts;
};

} // namespace clang::tidy::bugprone

#endif
