//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H

#include "../ClangTidyCheck.h"
#include "NoexceptFunctionBaseCheck.h"

namespace clang::tidy::performance {

/// The check flags swap functions not marked with `noexcept` or marked
/// with `noexcept(expr)` where `expr` evaluates to `false`
/// (but is not a `false` literal itself).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/noexcept-swap.html
class NoexceptSwapCheck : public NoexceptFunctionBaseCheck {
public:
  using NoexceptFunctionBaseCheck::NoexceptFunctionBaseCheck;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

private:
  DiagnosticBuilder
  reportMissingNoexcept(const FunctionDecl *FuncDecl) final override;
  void reportNoexceptEvaluatedToFalse(const FunctionDecl *FuncDecl,
                                      const Expr *NoexceptExpr) final override;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTSWAPCHECK_H
