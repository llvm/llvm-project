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

namespace {
AST_MATCHER(CXXRecordDecl, isAbstract) { return Node.isAbstract(); }
AST_MATCHER(CXXRecordDecl, isPolymorphic) { return Node.isPolymorphic(); }
} // namespace

PointerArithmeticOnPolymorphicObjectCheck::
    PointerArithmeticOnPolymorphicObjectCheck(StringRef Name,
                                              ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreInheritedVirtualFunctions(
          Options.get("IgnoreInheritedVirtualFunctions", false)) {}

void PointerArithmeticOnPolymorphicObjectCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreInheritedVirtualFunctions",
                IgnoreInheritedVirtualFunctions);
}

void PointerArithmeticOnPolymorphicObjectCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto PolymorphicPointerExpr =
      expr(hasType(hasCanonicalType(pointerType(pointee(hasCanonicalType(
               hasDeclaration(cxxRecordDecl(unless(isFinal()), isPolymorphic())
                                  .bind("pointee"))))))))
          .bind("pointer");

  const auto PointerExprWithVirtualMethod =
      expr(hasType(hasCanonicalType(
               pointerType(pointee(hasCanonicalType(hasDeclaration(
                   cxxRecordDecl(
                       unless(isFinal()),
                       anyOf(hasMethod(isVirtualAsWritten()), isAbstract()))
                       .bind("pointee"))))))))
          .bind("pointer");

  const auto SelectedPointerExpr = IgnoreInheritedVirtualFunctions
                                       ? PointerExprWithVirtualMethod
                                       : PolymorphicPointerExpr;

  const auto ArraySubscript = arraySubscriptExpr(hasBase(SelectedPointerExpr));

  const auto BinaryOperators =
      binaryOperator(hasAnyOperatorName("+", "-", "+=", "-="),
                     hasEitherOperand(SelectedPointerExpr));

  const auto UnaryOperators = unaryOperator(
      hasAnyOperatorName("++", "--"), hasUnaryOperand(SelectedPointerExpr));

  Finder->addMatcher(ArraySubscript, this);
  Finder->addMatcher(BinaryOperators, this);
  Finder->addMatcher(UnaryOperators, this);
}

void PointerArithmeticOnPolymorphicObjectCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PointerExpr = Result.Nodes.getNodeAs<Expr>("pointer");
  const auto *PointeeDecl = Result.Nodes.getNodeAs<CXXRecordDecl>("pointee");

  diag(PointerExpr->getBeginLoc(),
       "pointer arithmetic on polymorphic object of type %0 can result in "
       "undefined behavior if the dynamic type differs from the pointer type")
      << PointeeDecl << PointerExpr->getSourceRange();
}

} // namespace clang::tidy::bugprone
