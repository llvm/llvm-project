//===--- AvoidReturnWithVoidValueCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidReturnWithVoidValueCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void AvoidReturnWithVoidValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(hasReturnValue(hasType(voidType()))).bind("void_return"),
      this);
}

void AvoidReturnWithVoidValueCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VoidReturn = Result.Nodes.getNodeAs<ReturnStmt>("void_return");
  if (IgnoreMacros && VoidReturn->getBeginLoc().isMacroID())
    return;
  diag(VoidReturn->getBeginLoc(), "return statement within a void function "
                                  "should not have a specified return value");
}

} // namespace clang::tidy::readability
