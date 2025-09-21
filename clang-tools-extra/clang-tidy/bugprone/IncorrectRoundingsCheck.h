//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTROUNDINGSCHECK_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTROUNDINGSCHECK_H_

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Checks the usage of patterns known to produce incorrect rounding.
/// Programmers often use
///   (int)(double_expression + 0.5)
/// to round the double expression to an integer. The problem with this
///  1. It is unnecessarily slow.
///  2. It is incorrect. The number 0.499999975 (smallest representable float
///     number below 0.5) rounds to 1.0. Even worse behavior for negative
///     numbers where both -0.5f and -1.4f both round to 0.0.
class IncorrectRoundingsCheck : public ClangTidyCheck {
public:
  IncorrectRoundingsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTROUNDINGSCHECK_H_
