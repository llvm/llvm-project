//===--- SuspiciousPointerArithmeticsUsingSizeofCheck.h - clang-tidy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSPOINTERARITHMETICSUSINGSIZEOFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSPOINTERARITHMETICSUSINGSIZEOFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds suspicious pointer arithmetic calculations where the pointer is
/// offset by an alignof(), offsetof(), or sizeof() expression.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-pointer-arithmetics-using-sizeof.html
class SuspiciousPointerArithmeticsUsingSizeofCheck : public ClangTidyCheck {
public:
  SuspiciousPointerArithmeticsUsingSizeofCheck(StringRef Name,
                                               ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSPOINTERARITHMETICSUSINGSIZEOFCHECK_H
