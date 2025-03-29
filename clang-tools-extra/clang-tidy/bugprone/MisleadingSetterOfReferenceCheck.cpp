//===--- MisleadingSetterOfReferenceCheck.cpp - clang-tidy-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisleadingSetterOfReferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void MisleadingSetterOfReferenceCheck::registerMatchers(MatchFinder *Finder) {
  auto RefField =
      fieldDecl(unless(isPublic()),
                hasType(referenceType(pointee(equalsBoundNode("type")))))
          .bind("member");
  auto AssignLHS =
      memberExpr(hasObjectExpression(cxxThisExpr()), member(RefField));
  auto DerefOperand = expr(ignoringParenImpCasts(
      declRefExpr(to(parmVarDecl(equalsBoundNode("parm"))))));
  auto AssignRHS = expr(ignoringParenImpCasts(
      unaryOperator(hasOperatorName("*"), hasUnaryOperand(DerefOperand))));

  auto BinaryOpAssign = binaryOperator(hasOperatorName("="), hasLHS(AssignLHS),
                                       hasRHS(AssignRHS));
  auto CXXOperatorCallAssign = cxxOperatorCallExpr(
      hasOverloadedOperatorName("="), hasLHS(AssignLHS), hasRHS(AssignRHS));

  auto SetBody =
      compoundStmt(statementCountIs(1),
                   anyOf(has(BinaryOpAssign), has(CXXOperatorCallAssign)));
  auto BadSetFunction =
      cxxMethodDecl(parameterCountIs(1), isPublic(),
                    hasAnyParameter(parmVarDecl(hasType(pointerType(pointee(
                                                    qualType().bind("type")))))
                                        .bind("parm")),
                    hasBody(SetBody))
          .bind("bad-set-function");
  Finder->addMatcher(BadSetFunction, this);
}

void MisleadingSetterOfReferenceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Found = Result.Nodes.getNodeAs<CXXMethodDecl>("bad-set-function");
  const auto *Member = Result.Nodes.getNodeAs<FieldDecl>("member");

  diag(Found->getBeginLoc(),
       "function \'%0\' can be mistakenly used in order to change the "
       "reference \'%1\' instead of the value of it")
      << Found->getName() << Member->getName();
}

} // namespace clang::tidy::bugprone
