//===--- MisplacedWideningCastCheck.h - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACEDWIDENINGCASTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACEDWIDENINGCASTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

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
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-misplaced-widening-cast.html
class MisplacedWideningCastCheck : public ClangTidyCheck {
public:
  MisplacedWideningCastCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckImplicitCasts;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif
