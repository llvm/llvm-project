//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoexceptSwapCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

// FixItHint - comment added to fix list.rst generation in add_new_check.py.
// Do not remove. Fixes are generated in base class.

namespace clang::tidy::performance {

void NoexceptSwapCheck::registerMatchers(MatchFinder *Finder) {

  // Match non-const method with single argument that is non-const reference to
  // a class type that owns method and return void.
  // Matches: void Class::swap(Class&)
  auto MethodMatcher = cxxMethodDecl(
      parameterCountIs(1U), unless(isConst()), returns(voidType()),
      hasParameter(0, hasType(qualType(hasCanonicalType(
                          qualType(unless(isConstQualified()),
                                   references(namedDecl().bind("class"))))))),
      ofClass(equalsBoundNode("class")));

  // Match function with 2 arguments, both are non-const references to same type
  // and return void.
  // Matches: void swap(Type&, Type&)
  auto FunctionMatcher = allOf(
      unless(cxxMethodDecl()), parameterCountIs(2U), returns(voidType()),
      hasParameter(
          0, hasType(qualType(hasCanonicalType(
                 qualType(unless(isConstQualified()), references(qualType()))
                     .bind("type"))))),
      hasParameter(1, hasType(qualType(hasCanonicalType(
                          qualType(equalsBoundNode("type")))))));
  Finder->addMatcher(functionDecl(unless(isDeleted()),
                                  hasAnyName("swap", "iter_swap"),
                                  anyOf(MethodMatcher, FunctionMatcher))
                         .bind(BindFuncDeclName),
                     this);
}

DiagnosticBuilder
NoexceptSwapCheck::reportMissingNoexcept(const FunctionDecl *FuncDecl) {
  return diag(FuncDecl->getLocation(), "swap functions should "
                                       "be marked noexcept");
}

void NoexceptSwapCheck::reportNoexceptEvaluatedToFalse(
    const FunctionDecl *FuncDecl, const Expr *NoexceptExpr) {
  diag(NoexceptExpr->getExprLoc(),
       "noexcept specifier on swap function evaluates to 'false'");
}

} // namespace clang::tidy::performance
