//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_ALLOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_ALLOC_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds cases where an integer is added to or subracted from the result of a
/// memory allocation function instead of its argument.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/misplaced-pointer-arithmetic-in-alloc.html
class MisplacedPointerArithmeticInAllocCheck : public ClangTidyCheck {
public:
  MisplacedPointerArithmeticInAllocCheck(StringRef Name,
                                         ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_ALLOC_H
