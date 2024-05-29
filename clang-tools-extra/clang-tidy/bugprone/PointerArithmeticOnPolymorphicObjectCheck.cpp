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

PointerArithmeticOnPolymorphicObjectCheck::
    PointerArithmeticOnPolymorphicObjectCheck(StringRef Name,
                                              ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MatchInheritedVirtualFunctions(
          Options.get("MatchInheritedVirtualFunctions", false)) {}

void PointerArithmeticOnPolymorphicObjectCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MatchInheritedVirtualFunctions", true);
}

void PointerArithmeticOnPolymorphicObjectCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto PolymorphicPointerExpr =
      expr(hasType(hasCanonicalType(
               pointerType(pointee(hasCanonicalType(hasDeclaration(
                   cxxRecordDecl(cxxRecordDecl(hasMethod(isVirtual()))))))))))
          .bind("pointer");

  const auto PointerExprWithVirtualMethod =
      expr(hasType(hasCanonicalType(pointerType(
               pointee(hasCanonicalType(hasDeclaration(cxxRecordDecl(
                   hasMethod(anyOf(isVirtualAsWritten(), isPure()))))))))))
          .bind("pointer");

  const auto SelectedPointerExpr = MatchInheritedVirtualFunctions
                                       ? PolymorphicPointerExpr
                                       : PointerExprWithVirtualMethod;

  const auto ArraySubscript = arraySubscriptExpr(hasBase(SelectedPointerExpr));

  const auto BinaryOperators =
      binaryOperator(hasAnyOperatorName("+", "-", "+=", "-="),
                     hasEitherOperand(SelectedPointerExpr));

  const auto UnaryOperators = unaryOperator(
      hasAnyOperatorName("++", "--"), hasUnaryOperand(SelectedPointerExpr));

  Finder->addMatcher(
      expr(anyOf(ArraySubscript, BinaryOperators, UnaryOperators)), this);
}

void PointerArithmeticOnPolymorphicObjectCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PointerExpr = Result.Nodes.getNodeAs<Expr>("pointer");
  const CXXRecordDecl *PointeeType =
      PointerExpr->getType()->getPointeeType()->getAsCXXRecordDecl();

  diag(PointerExpr->getBeginLoc(),
       "pointer arithmetic on polymorphic class '%0', which can result in "
       "undefined behavior if the pointee is a different class")
      << PointeeType->getName();
}

} // namespace clang::tidy::bugprone
