//===--- TooSmallLoopVariableCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TOOSMALLLOOPVARIABLECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TOOSMALLLOOPVARIABLECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// This check gives a warning if a loop variable has a too small type which
/// might not be able to represent all values which are part of the whole range
/// in which the loop iterates.
/// If the loop variable's type is too small we might end up in an infinite
/// loop. Example:
/// \code
///   long size = 294967296l;
///   for (short i = 0; i < size; ++i) {} { ... }
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/too-small-loop-variable.html
class TooSmallLoopVariableCheck : public ClangTidyCheck {
public:
  TooSmallLoopVariableCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned MagnitudeBitsUpperLimit;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TOOSMALLLOOPVARIABLECHECK_H
