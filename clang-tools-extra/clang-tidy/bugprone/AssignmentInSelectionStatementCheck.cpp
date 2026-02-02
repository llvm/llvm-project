//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssignmentInSelectionStatementCheck.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

class ConditionValueCanPropagateFrom
    : public ConstStmtVisitor<ConditionValueCanPropagateFrom, void> {
public:
  llvm::SmallVector<const Expr *, 2> ExprToProcess;

  void VisitBinaryOperator(const BinaryOperator *BO) {
    if (BO->isCommaOp())
      ExprToProcess.push_back(BO->getRHS()->IgnoreParenImpCasts());
  }
  void VisitConditionalOperator(const ConditionalOperator *CO) {
    ExprToProcess.push_back(CO->getFalseExpr()->IgnoreParenImpCasts());
    ExprToProcess.push_back(CO->getTrueExpr()->IgnoreParenImpCasts());
  }
};

AST_MATCHER_P(Expr, conditionValueCanPropagateFrom,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  bool Found = false;
  ConditionValueCanPropagateFrom Visitor;
  Visitor.Visit(&Node); // Do not match Node itself.
  while (!Visitor.ExprToProcess.empty()) {
    const Expr *E = Visitor.ExprToProcess.pop_back_val();
    ast_matchers::internal::BoundNodesTreeBuilder Result;
    if (InnerMatcher.matches(*E, Finder, &Result)) {
      Found = true;
      Builder->addMatch(Result);
    }
    Visitor.Visit(E);
  }
  return Found;
}

} // namespace

namespace clang::tidy::bugprone {

void AssignmentInSelectionStatementCheck::registerMatchers(
    MatchFinder *Finder) {
  auto AssignOp = binaryOperation(hasOperatorName("=")).bind("assignment");

  auto CondExprWithAssign = expr(
      anyOf(ignoringImpCasts(AssignOp),
            ignoringParenImpCasts(conditionValueCanPropagateFrom(AssignOp))));
  auto OpCondExprWithAssign = expr(ignoringParenImpCasts(
      anyOf(AssignOp, conditionValueCanPropagateFrom(AssignOp))));

  // In these cases "single primary expression" is possible.
  // A single assignment within a 'ParenExpr' is allowed (but not if mixed with
  // other operators).
  auto FoundControlStmt = mapAnyOf(ifStmt, whileStmt, doStmt, forStmt)
                              .with(hasCondition(CondExprWithAssign));
  // In these cases "single primary expression" is not possible because the
  // assignment is already part of a bigger expression.
  auto FoundConditionalOperator =
      conditionalOperator(hasCondition(OpCondExprWithAssign));
  auto FoundLogicalOp = binaryOperator(
      hasAnyOperatorName("&&", "||"),
      eachOf(hasLHS(OpCondExprWithAssign), hasRHS(OpCondExprWithAssign)));

  auto FoundSelectionStmt =
      stmt(anyOf(FoundControlStmt, FoundConditionalOperator, FoundLogicalOp))
          .bind("parent");

  Finder->addMatcher(FoundSelectionStmt, this);
}

void AssignmentInSelectionStatementCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FoundAssignment =
      Result.Nodes.getNodeAs<BinaryOperator>("assignment");
  if (!FoundAssignment)
    return;
  const auto *ParentStmt = Result.Nodes.getNodeAs<Stmt>("parent");
  const char *CondStr = nullptr;
  switch (ParentStmt->getStmtClass()) {
  case Stmt::IfStmtClass:
    CondStr = "condition of 'if' statement";
    break;
  case Stmt::WhileStmtClass:
    CondStr = "condition of 'while' statement";
    break;
  case Stmt::DoStmtClass:
    CondStr = "condition of 'do' statement";
    break;
  case Stmt::ForStmtClass:
    CondStr = "condition of 'for' statement";
    break;
  case Stmt::ConditionalOperatorClass:
    CondStr = "condition of conditional operator";
    break;
  case Stmt::BinaryOperatorClass:
    CondStr = "operand of a logical operator";
    break;
  default:
    llvm_unreachable("unexpected statement class, should not match");
  };
  diag(FoundAssignment->getOperatorLoc(),
       "Assignment within %0 may indicate programmer error")
      << FoundAssignment->getSourceRange() << CondStr;
}

} // namespace clang::tidy::bugprone
