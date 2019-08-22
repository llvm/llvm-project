//===--- ImplicitBoolConversionCheck.h - clang-tidy--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CONVERSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CONVERSION_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks for use of implicit bool conversions in expressions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-implicit-bool-conversion.html
class ImplicitBoolConversionCheck : public ClangTidyCheck {
public:
  ImplicitBoolConversionCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void handleCastToBool(const ImplicitCastExpr *CastExpression,
                        const Stmt *ParentStatement, ASTContext &Context);
  void handleCastFromBool(const ImplicitCastExpr *CastExpression,
                          const ImplicitCastExpr *FurtherImplicitCastExpression,
                          ASTContext &Context);

  const bool AllowIntegerConditions;
  const bool AllowPointerConditions;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CONVERSION_H
