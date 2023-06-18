//===--- NoexceptFunctionCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTFUNCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTFUNCTIONCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/ExceptionSpecAnalyzer.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::performance {

/// Generic check which checks if the bound function decl is
/// marked with `noexcept` or `noexcept(expr)` where `expr` evaluates to
/// `false`.
class NoexceptFunctionBaseCheck : public ClangTidyCheck {
public:
  NoexceptFunctionBaseCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11 && LangOpts.CXXExceptions;
  }
  void
  check(const ast_matchers::MatchFinder::MatchResult &Result) final override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

protected:
  virtual DiagnosticBuilder
  reportMissingNoexcept(const FunctionDecl *FuncDecl) = 0;
  virtual void reportNoexceptEvaluatedToFalse(const FunctionDecl *FuncDecl,
                                              const Expr *NoexceptExpr) = 0;

  static constexpr StringRef BindFuncDeclName = "FuncDecl";

private:
  utils::ExceptionSpecAnalyzer SpecAnalyzer;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_NOEXCEPTFUNCTIONCHECK_H
