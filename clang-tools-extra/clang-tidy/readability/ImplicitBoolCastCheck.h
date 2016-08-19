//===--- ImplicitBoolCastCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CAST_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Checks for use of implicit bool casts in expressions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-implicit-bool-cast.html
class ImplicitBoolCastCheck : public ClangTidyCheck {
public:
  ImplicitBoolCastCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void handleCastToBool(const ImplicitCastExpr *CastExpression,
                        const Stmt *ParentStatement, ASTContext &Context);
  void handleCastFromBool(const ImplicitCastExpr *CastExpression,
                          const ImplicitCastExpr *FurtherImplicitCastExpression,
                          ASTContext &Context);

  bool AllowConditionalIntegerCasts;
  bool AllowConditionalPointerCasts;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IMPLICIT_BOOL_CAST_H
