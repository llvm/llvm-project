//===--- AvoidConstOrRefDataMembersCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidConstOrRefDataMembersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void AvoidConstOrRefDataMembersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      fieldDecl(hasType(hasCanonicalType(referenceType()))).bind("ref"), this);
  Finder->addMatcher(
      fieldDecl(hasType(qualType(isConstQualified()))).bind("const"), this);
}

void AvoidConstOrRefDataMembersCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<FieldDecl>("ref"))
    diag(MatchedDecl->getLocation(), "member %0 of type %1 is a reference")
        << MatchedDecl << MatchedDecl->getType();
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<FieldDecl>("const"))
    diag(MatchedDecl->getLocation(), "member %0 of type %1 is const qualified")
        << MatchedDecl << MatchedDecl->getType();
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
