//===--- UnconventionalAssignOperatorCheck.cpp - clang-tidy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnconventionalAssignOperatorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void UnconventionalAssignOperatorCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  const auto HasGoodReturnType =
      cxxMethodDecl(returns(hasCanonicalType(lValueReferenceType(pointee(
          unless(isConstQualified()),
          anyOf(autoType(), hasDeclaration(equalsBoundNode("class"))))))));

  const auto IsSelf = qualType(hasCanonicalType(
      anyOf(hasDeclaration(equalsBoundNode("class")),
            referenceType(pointee(hasDeclaration(equalsBoundNode("class")))))));
  const auto IsAssign =
      cxxMethodDecl(unless(anyOf(isDeleted(), isPrivate(), isImplicit())),
                    hasName("operator="), ofClass(recordDecl().bind("class")))
          .bind("method");
  const auto IsSelfAssign =
      cxxMethodDecl(IsAssign, hasParameter(0, parmVarDecl(hasType(IsSelf))))
          .bind("method");

  Finder->addMatcher(
      cxxMethodDecl(IsAssign, unless(HasGoodReturnType)).bind("ReturnType"),
      this);

  const auto BadSelf = qualType(hasCanonicalType(referenceType(
      anyOf(lValueReferenceType(pointee(unless(isConstQualified()))),
            rValueReferenceType(pointee(isConstQualified()))))));

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign,
                    hasParameter(0, parmVarDecl(hasType(BadSelf))))
          .bind("ArgumentType"),
      this);

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign, anyOf(isConst(), isVirtual())).bind("cv"),
      this);

  const auto IsBadReturnStatement = returnStmt(unless(has(ignoringParenImpCasts(
      anyOf(unaryOperator(hasOperatorName("*"), hasUnaryOperand(cxxThisExpr())),
            cxxOperatorCallExpr(argumentCountIs(1),
                                callee(unresolvedLookupExpr()),
                                hasArgument(0, cxxThisExpr())),
            cxxOperatorCallExpr(
                hasOverloadedOperatorName("="),
                hasArgument(
                    0, unaryOperator(hasOperatorName("*"),
                                     hasUnaryOperand(cxxThisExpr())))))))));
  const auto IsGoodAssign = cxxMethodDecl(IsAssign, HasGoodReturnType);

  Finder->addMatcher(returnStmt(IsBadReturnStatement, forFunction(IsGoodAssign))
                         .bind("returnStmt"),
                     this);
}

void UnconventionalAssignOperatorCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *RetStmt = Result.Nodes.getNodeAs<ReturnStmt>("returnStmt")) {
    diag(RetStmt->getBeginLoc(), "operator=() should always return '*this'");
  } else {
    const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
    if (Result.Nodes.getNodeAs<CXXMethodDecl>("ReturnType"))
      diag(Method->getBeginLoc(), "operator=() should return '%0&'")
          << Method->getParent()->getName();
    if (Result.Nodes.getNodeAs<CXXMethodDecl>("ArgumentType"))
      diag(Method->getBeginLoc(),
           "operator=() should take '%0 const&'%select{|, '%0&&'}1 or '%0'")
          << Method->getParent()->getName() << getLangOpts().CPlusPlus11;
    if (Result.Nodes.getNodeAs<CXXMethodDecl>("cv"))
      diag(Method->getBeginLoc(),
           "operator=() should not be marked '%select{const|virtual}0'")
          << !Method->isConst();
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
