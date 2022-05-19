//===--- SimplifyBooleanExpr.h clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Looks for boolean expressions involving boolean constants and simplifies
/// them to use the appropriate boolean expression directly.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-simplify-boolean-expr.html
class SimplifyBooleanExprCheck : public ClangTidyCheck {
public:
  SimplifyBooleanExprCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  llvm::Optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  class Visitor;

  void reportBinOp(const ASTContext &Context, const BinaryOperator *Op);

  void replaceWithThenStatement(const ASTContext &Context,
                                const IfStmt *IfStatement,
                                const Expr *BoolLiteral);

  void replaceWithElseStatement(const ASTContext &Context,
                                const IfStmt *IfStatement,
                                const Expr *BoolLiteral);

  void replaceWithCondition(const ASTContext &Context,
                            const ConditionalOperator *Ternary, bool Negated);

  void replaceWithReturnCondition(const ASTContext &Context, const IfStmt *If,
                                  const Expr *BoolLiteral, bool Negated);

  void replaceWithAssignment(const ASTContext &Context, const IfStmt *If,
                             const Expr *Var, SourceLocation Loc, bool Negated);

  void replaceCompoundReturnWithCondition(const ASTContext &Context,
                                          const ReturnStmt *Ret, bool Negated,
                                          const IfStmt *If,
                                          const Expr *ThenReturn);

  void issueDiag(const ASTContext &Result, SourceLocation Loc,
                 StringRef Description, SourceRange ReplacementRange,
                 StringRef Replacement);

  const bool ChainedConditionalReturn;
  const bool ChainedConditionalAssignment;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
