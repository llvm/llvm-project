//===----------------------------------------------------------------------===//
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

namespace clang::tidy::cppcoreguidelines {

static bool isCopyConstructible(const CXXRecordDecl &Node) {
  if (Node.needsOverloadResolutionForCopyConstructor() &&
      Node.needsImplicitCopyConstructor()) {
    // unresolved
    for (const CXXBaseSpecifier &BS : Node.bases()) {
      const CXXRecordDecl *BRD = BS.getType()->getAsCXXRecordDecl();
      if (BRD != nullptr && !isCopyConstructible(*BRD))
        return false;
    }
  }
  if (Node.hasSimpleCopyConstructor())
    return true;
  for (const CXXConstructorDecl *Ctor : Node.ctors())
    if (Ctor->isCopyConstructor())
      return !Ctor->isDeleted();
  return false;
}

static bool isMoveConstructible(const CXXRecordDecl &Node) {
  if (Node.needsOverloadResolutionForMoveConstructor() &&
      Node.needsImplicitMoveConstructor()) {
    // unresolved
    for (const CXXBaseSpecifier &BS : Node.bases()) {
      const CXXRecordDecl *BRD = BS.getType()->getAsCXXRecordDecl();
      if (BRD != nullptr && !isMoveConstructible(*BRD))
        return false;
    }
  }
  if (Node.hasSimpleMoveConstructor())
    return true;
  for (const CXXConstructorDecl *Ctor : Node.ctors())
    if (Ctor->isMoveConstructor())
      return !Ctor->isDeleted();
  return false;
}

static bool isCopyAssignable(const CXXRecordDecl &Node) {
  if (Node.needsOverloadResolutionForCopyAssignment() &&
      Node.needsImplicitCopyAssignment()) {
    // unresolved
    for (const CXXBaseSpecifier &BS : Node.bases()) {
      const CXXRecordDecl *BRD = BS.getType()->getAsCXXRecordDecl();
      if (BRD != nullptr && !isCopyAssignable(*BRD))
        return false;
    }
  }
  if (Node.hasSimpleCopyAssignment())
    return true;
  for (const CXXMethodDecl *Method : Node.methods())
    if (Method->isCopyAssignmentOperator())
      return !Method->isDeleted();
  return false;
}

static bool isMoveAssignable(const CXXRecordDecl &Node) {
  if (Node.needsOverloadResolutionForMoveAssignment() &&
      Node.needsImplicitMoveAssignment()) {
    // unresolved
    for (const CXXBaseSpecifier &BS : Node.bases()) {
      const CXXRecordDecl *BRD = BS.getType()->getAsCXXRecordDecl();
      if (BRD != nullptr && !isMoveAssignable(*BRD))
        return false;
    }
  }
  if (Node.hasSimpleMoveAssignment())
    return true;
  for (const CXXMethodDecl *Method : Node.methods())
    if (Method->isMoveAssignmentOperator())
      return !Method->isDeleted();
  return false;
}

namespace {

AST_MATCHER(FieldDecl, isMemberOfLambda) {
  return Node.getParent()->isLambda();
}

AST_MATCHER(CXXRecordDecl, isCopyableOrMovable) {
  return isCopyConstructible(Node) || isMoveConstructible(Node) ||
         isCopyAssignable(Node) || isMoveAssignable(Node);
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
