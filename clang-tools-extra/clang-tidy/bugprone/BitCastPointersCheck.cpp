//===--- BitCastPointersCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitCastPointersCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void BitCastPointersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(allOf(hasName("::std::bit_cast"),
                                         hasAnyTemplateArgument(refersToType(
                                             qualType(isAnyPointer())))))))
          .bind("x"),
      this);
}

void BitCastPointersCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("x"))
    diag(MatchedDecl->getBeginLoc(), "do not use std::bit_cast on pointers; use reinterpret_cast instead");
}

} // namespace clang::tidy::bugprone
