//===--- ExceptionRethrowCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionRethrowCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void ExceptionRethrowCheck::registerMatchers(MatchFinder *Finder) {

  auto RefToExceptionVariable = declRefExpr(to(varDecl(isExceptionVariable())));
  auto StdMoveCall =
      callExpr(argumentCountIs(1), callee(functionDecl(hasName("::std::move"))),
               hasArgument(0, RefToExceptionVariable));
  auto CopyOrMoveConstruction = cxxConstructExpr(
      argumentCountIs(1),
      traverse(TK_AsIs, hasDeclaration(cxxConstructorDecl(
                            anyOf(isCopyConstructor(), isMoveConstructor())))),
      hasArgument(0, anyOf(RefToExceptionVariable, StdMoveCall)));

  auto HasEmptyThrowExprDescendant =
      hasDescendant(cxxThrowExpr(equalsBoundNode("empty-throw")));

  Finder->addMatcher(
      cxxThrowExpr(
          unless(isExpansionInSystemHeader()), unless(has(expr())),
          expr().bind("empty-throw"),
          anyOf(unless(hasAncestor(cxxCatchStmt())),
                hasAncestor(cxxCatchStmt(anyOf(
                    hasDescendant(functionDecl(HasEmptyThrowExprDescendant)),
                    hasDescendant(lambdaExpr(HasEmptyThrowExprDescendant))))))),
      this);
  Finder->addMatcher(
      cxxThrowExpr(unless(isExpansionInSystemHeader()),
                   has(expr(anyOf(RefToExceptionVariable, StdMoveCall,
                                  CopyOrMoveConstruction))))
          .bind("throw"),
      this);
}

void ExceptionRethrowCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedThrow =
          Result.Nodes.getNodeAs<CXXThrowExpr>("throw")) {
    const Expr *ThrownObject = MatchedThrow->getSubExpr();
    diag(MatchedThrow->getThrowLoc(),
         "throwing a copy of the caught %0 exception, remove the argument to "
         "throw the original exception object")
        << ThrownObject->getType().getNonReferenceType();
    return;
  }

  if (const auto *MatchedEmptyThrow =
          Result.Nodes.getNodeAs<CXXThrowExpr>("empty-throw")) {
    diag(MatchedEmptyThrow->getThrowLoc(),
         "empty 'throw' outside a catch block with no operand triggers "
         "'std::terminate()'");
  }
}

} // namespace clang::tidy::bugprone
