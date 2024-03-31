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
  auto StdMoveCall = callExpr(argumentCountIs(1), callee(functionDecl(hasName("::std::move"))), hasArgument(0, RefToExceptionVariable));
  auto CopyOrMoveConstruction = cxxConstructExpr(argumentCountIs(1), hasDeclaration(cxxConstructorDecl(anyOf(isCopyConstructor(), isMoveConstructor()))), 	hasArgument(0, anyOf(RefToExceptionVariable, StdMoveCall)));
  auto FunctionCast = cxxFunctionalCastExpr(	hasSourceExpression(anyOf(RefToExceptionVariable, StdMoveCall)));

  Finder->addMatcher(
      cxxThrowExpr(unless(isExpansionInSystemHeader()),
                   anyOf(unless(has(expr())),
                         has(RefToExceptionVariable),
                         has(StdMoveCall),
                         has(CopyOrMoveConstruction),
                         has(FunctionCast)),
                   optionally(hasAncestor(cxxCatchStmt().bind("catch"))))
          .bind("throw"),
      this);
}

void ExceptionRethrowCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedThrow = Result.Nodes.getNodeAs<CXXThrowExpr>("throw");

  if (const Expr *ThrownObject = MatchedThrow->getSubExpr()) {
    diag(MatchedThrow->getThrowLoc(),
         "throwing a copy of the caught %0 exception, remove the argument to "
         "throw the original exception object")
        << ThrownObject->getType().getNonReferenceType();
    return;
  }

  const bool HasCatchAncestor =
      Result.Nodes.getNodeAs<Stmt>("catch") != nullptr;
  if (!HasCatchAncestor) {
    diag(MatchedThrow->getThrowLoc(),
         "empty 'throw' outside a catch block with no operand triggers 'std::terminate()'");
  }
}

} // namespace clang::tidy::bugprone
