//===--- UseDigitSeparatorCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDigitSeparatorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseDigitSeparatorCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME: Add matchers.
//  Finder->addMatcher(functionDecl().bind("x"), this);
  Finder->addMatcher(integerLiteral().bind("integerLiteral"), this);
}

void UseDigitSeparatorCheck::check(const MatchFinder::MatchResult &Result) {
  // FIXME: Add callback implementation.
//  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("x");
//  if (!MatchedDecl->getIdentifier() || MatchedDecl->getName().startswith("awesome_"))
//    return;
//  diag(MatchedDecl->getLocation(), "function %0 is insufficiently awesome")
//      << MatchedDecl
//      << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
//  diag(MatchedDecl->getLocation(), "insert 'awesome'", DiagnosticIDs::Note);
  const auto *MatchedInteger = Result.Nodes.getNodeAs<IntegerLiteral>("integerLiteral");
  const auto IntegerValue = MatchedInteger->getValue();
  diag(MatchedInteger->getLocation(), "integer warning %0") << toString(IntegerValue, 10, true)
        << FixItHint::CreateInsertion(MatchedInteger->getLocation(), "this is integer");
  diag(MatchedInteger->getLocation(), "integer", DiagnosticIDs::Note);
}

} // namespace clang::tidy::modernize
