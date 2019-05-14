//===--- ImplicitConversionInLoopCheck.h - clang-tidy------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CONVERSION_IN_LOOP_CHECK_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CONVERSION_IN_LOOP_CHECK_H_

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace performance {

// Checks that in a for range loop, if the provided type is a reference, then
// the underlying type is the one returned by the iterator (i.e. that there
// isn't any implicit conversion).
class ImplicitConversionInLoopCheck : public ClangTidyCheck {
public:
  ImplicitConversionInLoopCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void ReportAndFix(const ASTContext *Context, const VarDecl *VD,
                    const Expr *OperatorCall);
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CONVERSION_IN_LOOP_CHECK_H_
