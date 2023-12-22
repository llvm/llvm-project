//===--- ReturnExpressionInVoidFunctionCheck.cpp - clang-tidy -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnExpressionInVoidFunctionCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void ReturnExpressionInVoidFunctionCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(hasReturnValue(hasType(voidType()))).bind("void_return"),
      this);
}

void ReturnExpressionInVoidFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VoidReturn = Result.Nodes.getNodeAs<ReturnStmt>("void_return");
  diag(VoidReturn->getBeginLoc(),
       "return statement in void function should not return a value");
}

} // namespace clang::tidy::readability
