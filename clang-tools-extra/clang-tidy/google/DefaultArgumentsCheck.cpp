//===--- DefaultArgumentsCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultArgumentsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google {

void DefaultArgumentsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(anyOf(isOverride(), isVirtual()),
                    hasAnyParameter(parmVarDecl(hasInitializer(expr()))))
          .bind("Decl"),
      this);
}

void DefaultArgumentsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("Decl");
  diag(MatchedDecl->getLocation(),
       "default arguments on virtual or override methods are prohibited");
}

} // namespace clang::tidy::google
