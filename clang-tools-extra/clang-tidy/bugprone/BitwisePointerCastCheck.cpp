//===--- BitwisePointerCastCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitwisePointerCastCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void BitwisePointerCastCheck::registerMatchers(MatchFinder *Finder) {
  if (getLangOpts().CPlusPlus20) {
    auto IsPointerType = refersToType(qualType(isAnyPointer()));
    Finder->addMatcher(callExpr(hasDeclaration(functionDecl(allOf(
                                    hasName("::std::bit_cast"),
                                    hasTemplateArgument(0, IsPointerType),
                                    hasTemplateArgument(1, IsPointerType)))))
                           .bind("bit_cast"),
                       this);
  }

  auto IsDoublePointerType =
      hasType(qualType(pointsTo(qualType(isAnyPointer()))));
  Finder->addMatcher(callExpr(hasArgument(0, IsDoublePointerType),
                              hasArgument(1, IsDoublePointerType),
                              hasDeclaration(functionDecl(hasName("::memcpy"))))
                         .bind("memcpy"),
                     this);
}

void BitwisePointerCastCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("bit_cast"))
    diag(Call->getBeginLoc(),
         "do not use 'std::bit_cast' to cast between pointers")
        << Call->getSourceRange();
  else if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("memcpy"))
    diag(Call->getBeginLoc(), "do not use 'memcpy' to cast between pointers")
        << Call->getSourceRange();
}

} // namespace clang::tidy::bugprone
