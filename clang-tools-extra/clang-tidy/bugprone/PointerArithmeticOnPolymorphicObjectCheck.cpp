//===--- PointerArithmeticOnPolymorphicObjectCheck.cpp - clang-tidy--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerArithmeticOnPolymorphicObjectCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void PointerArithmeticOnPolymorphicObjectCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto PointerExprWithVirtualMethod =
      expr(hasType(pointerType(pointee(hasDeclaration(
               cxxRecordDecl(hasMethod(isVirtualAsWritten())))))))
          .bind("pointer");

  const auto ArraySubscript =
      arraySubscriptExpr(hasBase(PointerExprWithVirtualMethod));

  const auto BinaryOperators =
      binaryOperator(hasAnyOperatorName("+", "-", "+=", "-="),
                     hasEitherOperand(PointerExprWithVirtualMethod));

  const auto UnaryOperators =
      unaryOperator(hasAnyOperatorName("++", "--"),
                    hasUnaryOperand(PointerExprWithVirtualMethod));

  Finder->addMatcher(
      expr(anyOf(ArraySubscript, BinaryOperators, UnaryOperators)), this);
}

void PointerArithmeticOnPolymorphicObjectCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PointerExpr = Result.Nodes.getNodeAs<Expr>("pointer");
  const CXXRecordDecl *PointeeType =
      PointerExpr->getType()->getPointeeType()->getAsCXXRecordDecl();

  diag(PointerExpr->getBeginLoc(),
       "pointer arithmetic on polymorphic class '%0' that declares a virtual "
       "function, undefined behavior if the pointee is a different class")
      << PointeeType->getName();
}

} // namespace clang::tidy::bugprone
