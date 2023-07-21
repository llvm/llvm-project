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

namespace clang::tidy::cppcoreguidelines {
namespace {

AST_MATCHER(FieldDecl, isMemberOfLambda) {
  return Node.getParent()->isLambda();
}

struct MemberFunctionInfo {
  bool Declared{};
  bool Deleted{};
};

struct MemberFunctionPairInfo {
  MemberFunctionInfo Copy{};
  MemberFunctionInfo Move{};
};

MemberFunctionPairInfo getConstructorsInfo(CXXRecordDecl const &Node) {
  MemberFunctionPairInfo Constructors{};

  for (CXXConstructorDecl const *Ctor : Node.ctors()) {
    if (Ctor->isCopyConstructor()) {
      Constructors.Copy.Declared = true;
      if (Ctor->isDeleted())
        Constructors.Copy.Deleted = true;
    }
    if (Ctor->isMoveConstructor()) {
      Constructors.Move.Declared = true;
      if (Ctor->isDeleted())
        Constructors.Move.Deleted = true;
    }
  }

  return Constructors;
}

MemberFunctionPairInfo getAssignmentsInfo(CXXRecordDecl const &Node) {
  MemberFunctionPairInfo Assignments{};

  for (CXXMethodDecl const *Method : Node.methods()) {
    if (Method->isCopyAssignmentOperator()) {
      Assignments.Copy.Declared = true;
      if (Method->isDeleted())
        Assignments.Copy.Deleted = true;
    }

    if (Method->isMoveAssignmentOperator()) {
      Assignments.Move.Declared = true;
      if (Method->isDeleted())
        Assignments.Move.Deleted = true;
    }
  }

  return Assignments;
}

AST_MATCHER(CXXRecordDecl, isCopyableOrMovable) {
  MemberFunctionPairInfo Constructors = getConstructorsInfo(Node);
  MemberFunctionPairInfo Assignments = getAssignmentsInfo(Node);

  if (Node.hasSimpleCopyConstructor() ||
      (Constructors.Copy.Declared && !Constructors.Copy.Deleted))
    return true;
  if (Node.hasSimpleMoveConstructor() ||
      (Constructors.Move.Declared && !Constructors.Move.Deleted))
    return true;
  if (Node.hasSimpleCopyAssignment() ||
      (Assignments.Copy.Declared && !Assignments.Copy.Deleted))
    return true;
  if (Node.hasSimpleMoveAssignment() ||
      (Assignments.Move.Declared && !Assignments.Move.Deleted))
    return true;

  return false;
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
