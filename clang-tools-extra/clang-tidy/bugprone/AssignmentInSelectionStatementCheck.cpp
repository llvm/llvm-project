//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssignmentInSelectionStatementCheck.h"
#include "clang/AST/IgnoreExpr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/TypeSwitch.h"

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

// Ignore implicit casts (including C++ conversion member calls) but not parens.
AST_MATCHER_P(Expr, ignoringImplicitAsWritten,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  auto IgnoreImplicitMemberCallSingleStep = [](Expr *E) {
    if (auto *C = dyn_cast<CXXMemberCallExpr>(E)) {
      Expr *ExprNode = C->getImplicitObjectArgument();
      if (ExprNode->getSourceRange() == E->getSourceRange())
        return ExprNode;
      if (auto *PE = dyn_cast<ParenExpr>(ExprNode)) {
        if (PE->getSourceRange() == C->getSourceRange())
          return cast<Expr>(PE);
      }
      ExprNode = ExprNode->IgnoreParenImpCasts();
      if (ExprNode->getSourceRange() == E->getSourceRange())
        return ExprNode;
    }
    return E;
  };

  const Expr *IgnoreE = IgnoreExprNodes(&Node, IgnoreImplicitSingleStep,
                                        IgnoreImplicitCastsExtraSingleStep,
                                        IgnoreImplicitMemberCallSingleStep);

  return InnerMatcher.matches(*IgnoreE, Finder, Builder);
}

} // namespace

namespace clang::tidy::bugprone {

void AssignmentInSelectionStatementCheck::registerMatchers(
    MatchFinder *Finder) {
  auto AssignOpNoParens = ignoringImplicitAsWritten(
      binaryOperation(hasOperatorName("=")).bind("assignment"));
  auto AssignOpMaybeParens = ignoringParenImpCasts(
      binaryOperation(hasOperatorName("=")).bind("assignment"));
  auto AssignOpFromEmbeddedExpr = expr(ignoringParenImpCasts(
      conditionValueCanPropagateFrom(AssignOpMaybeParens)));

  auto CondExprWithAssign = anyOf(AssignOpNoParens, AssignOpFromEmbeddedExpr);
  auto OpCondExprWithAssign =
      anyOf(AssignOpMaybeParens, AssignOpFromEmbeddedExpr);

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
  const auto *FoundAssignment = Result.Nodes.getNodeAs<Stmt>("assignment");
  assert(FoundAssignment);

  const auto *ParentStmt = Result.Nodes.getNodeAs<Stmt>("parent");
  StringRef CondStr =
      llvm::TypeSwitch<const Stmt *, const char *>(ParentStmt)
          .Case<IfStmt>(
              [](const IfStmt *) { return "condition of 'if' statement"; })
          .Case<WhileStmt, DoStmt, ForStmt>(
              [](const Stmt *) { return "condition of a loop"; })
          .Case<ConditionalOperator>([](const ConditionalOperator *) {
            return "condition of a ternary operator";
          })
          .Case<BinaryOperator>([](const BinaryOperator *) {
            return "operand of a logical operator";
          })
          .DefaultUnreachable();

  SourceLocation OpLoc =
      llvm::TypeSwitch<const Stmt *, SourceLocation>(FoundAssignment)
          .Case<BinaryOperator, CXXOperatorCallExpr>(
              [](const auto *Op) { return Op->getOperatorLoc(); })
          .Default(FoundAssignment->getBeginLoc());
  diag(OpLoc, "assignment within %0 may indicate programmer error")
      << FoundAssignment->getSourceRange() << CondStr;
  diag(OpLoc, "if it should be an assignment, move it out of the condition",
       DiagnosticIDs::Note);
  diag(OpLoc, "if it is meant to be an equality check, change '=' to '=='",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
