//===--- IncDecInConditionsCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncDecInConditionsCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

AST_MATCHER(BinaryOperator, isLogicalOperator) { return Node.isLogicalOp(); }

AST_MATCHER(UnaryOperator, isUnaryPrePostOperator) {
  return Node.isPrefix() || Node.isPostfix();
}

AST_MATCHER(CXXOperatorCallExpr, isPrePostOperator) {
  return Node.getOperator() == OO_PlusPlus ||
         Node.getOperator() == OO_MinusMinus;
}

void IncDecInConditionsCheck::registerMatchers(MatchFinder *Finder) {
  auto OperatorMatcher = expr(
      anyOf(binaryOperator(anyOf(isComparisonOperator(), isLogicalOperator())),
            cxxOperatorCallExpr(isComparisonOperator())));

  auto IsInUnevaluatedContext =
      expr(anyOf(hasAncestor(expr(matchers::hasUnevaluatedContext())),
                 hasAncestor(typeLoc())));

  Finder->addMatcher(
      expr(
          OperatorMatcher, unless(isExpansionInSystemHeader()),
          unless(hasAncestor(OperatorMatcher)), expr().bind("parent"),

          forEachDescendant(
              expr(anyOf(unaryOperator(isUnaryPrePostOperator(),
                                       hasUnaryOperand(expr().bind("operand"))),
                         cxxOperatorCallExpr(
                             isPrePostOperator(),
                             hasUnaryOperand(expr().bind("operand")))),
                   unless(IsInUnevaluatedContext),
                   hasAncestor(
                       expr(equalsBoundNode("parent"),
                            hasDescendant(
                                expr(unless(equalsBoundNode("operand")),
                                     matchers::isStatementIdenticalToBoundNode(
                                         "operand"),
                                     unless(IsInUnevaluatedContext))
                                    .bind("second")))))
                  .bind("operator"))),
      this);
}

void IncDecInConditionsCheck::check(const MatchFinder::MatchResult &Result) {

  SourceLocation ExprLoc;
  bool IsIncrementOp = false;

  if (const auto *MatchedDecl =
          Result.Nodes.getNodeAs<CXXOperatorCallExpr>("operator")) {
    ExprLoc = MatchedDecl->getExprLoc();
    IsIncrementOp = (MatchedDecl->getOperator() == OO_PlusPlus);
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<UnaryOperator>("operator")) {
    ExprLoc = MatchedDecl->getExprLoc();
    IsIncrementOp = MatchedDecl->isIncrementOp();
  } else
    return;

  diag(ExprLoc,
       "%select{decrementing|incrementing}0 and referencing a variable in a "
       "complex condition can cause unintended side-effects due to C++'s order "
       "of evaluation, consider moving the modification outside of the "
       "condition to avoid misunderstandings")
      << IsIncrementOp;
  diag(Result.Nodes.getNodeAs<Expr>("second")->getExprLoc(),
       "variable is referenced here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
