//===--- SimplifyBooleanExpr.h clang-tidy -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H

#include "../ClangTidy.h"

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

private:
  void matchBoolBinOpExpr(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef OperatorName, StringRef BooleanId);

  void matchExprBinOpBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef OperatorName, StringRef BooleanId);

  void matchBoolCompOpExpr(ast_matchers::MatchFinder *Finder, bool Value,
                           StringRef OperatorName, StringRef BooleanId);

  void matchExprCompOpBool(ast_matchers::MatchFinder *Finder, bool Value,
                           StringRef OperatorName, StringRef BooleanId);

  void matchBoolCondition(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef BooleanId);

  void matchTernaryResult(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef TernaryId);

  void matchIfReturnsBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef Id);

  void matchIfAssignsBool(ast_matchers::MatchFinder *Finder, bool Value,
                          StringRef Id);

  void matchCompoundIfReturnsBool(ast_matchers::MatchFinder *Finder, bool Value,
                                  StringRef Id);

  void
  replaceWithExpression(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CXXBoolLiteralExpr *BoolLiteral, bool UseLHS,
                        bool Negated = false);

  void
  replaceWithThenStatement(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CXXBoolLiteralExpr *BoolLiteral);

  void
  replaceWithElseStatement(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CXXBoolLiteralExpr *FalseConditionRemoved);

  void
  replaceWithCondition(const ast_matchers::MatchFinder::MatchResult &Result,
                       const ConditionalOperator *Ternary,
                       bool Negated = false);

  void replaceWithReturnCondition(
      const ast_matchers::MatchFinder::MatchResult &Result, const IfStmt *If,
      bool Negated = false);

  void
  replaceWithAssignment(const ast_matchers::MatchFinder::MatchResult &Result,
                        const IfStmt *If, bool Negated = false);

  void replaceCompoundReturnWithCondition(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const CompoundStmt *Compound, bool Negated = false);

  void issueDiag(const ast_matchers::MatchFinder::MatchResult &Result,
                 SourceLocation Loc, StringRef Description,
                 SourceRange ReplacementRange, StringRef Replacement);

  const bool ChainedConditionalReturn;
  const bool ChainedConditionalAssignment;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SIMPLIFY_BOOLEAN_EXPR_H
