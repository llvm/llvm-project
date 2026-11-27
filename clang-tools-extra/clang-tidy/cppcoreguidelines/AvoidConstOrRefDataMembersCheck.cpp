//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidConstOrRefDataMembersCheck.h"
#include "../utils/TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {

AST_MATCHER(FieldDecl, isMemberOfLambda) {
  return Node.getParent()->isLambda();
}

AST_MATCHER(CXXRecordDecl, isCopyableOrMovable) {
  return utils::type_traits::isCopyConstructible(Node) ||
         utils::type_traits::isMoveConstructible(Node) ||
         utils::type_traits::isCopyAssignable(Node) ||
         utils::type_traits::isMoveAssignable(Node);
}

} // namespace

void AvoidConstOrRefDataMembersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      fieldDecl(
          unless(isMemberOfLambda()),
          anyOf(
              fieldDecl(hasType(hasCanonicalType(referenceType()))).bind("ref"),
              fieldDecl(hasType(qualType(isConstQualified()))).bind("const")),
          hasDeclContext(cxxRecordDecl(isCopyableOrMovable()))),
      this);
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

} // namespace clang::tidy::cppcoreguidelines
