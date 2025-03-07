//===--- CaptureThisByFieldCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CaptureThisByFieldCheck.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(CXXRecordDecl, correctHandleCaptureThisLambda) {
  // unresolved
  if (Node.needsOverloadResolutionForCopyConstructor() &&
      Node.needsImplicitCopyConstructor())
    return false;
  if (Node.needsOverloadResolutionForMoveConstructor() &&
      Node.needsImplicitMoveConstructor())
    return false;
  if (Node.needsOverloadResolutionForCopyAssignment() &&
      Node.needsImplicitCopyAssignment())
    return false;
  if (Node.needsOverloadResolutionForMoveAssignment() &&
      Node.needsImplicitMoveAssignment())
    return false;
  // default but not deleted
  if (Node.hasSimpleCopyConstructor())
    return false;
  if (Node.hasSimpleMoveConstructor())
    return false;
  if (Node.hasSimpleCopyAssignment())
    return false;
  if (Node.hasSimpleMoveAssignment())
    return false;

  for (CXXConstructorDecl const *C : Node.ctors()) {
    if (C->isCopyOrMoveConstructor() && C->isDefaulted() && !C->isDeleted())
      return false;
  }
  for (CXXMethodDecl const *M : Node.methods()) {
    if (M->isCopyAssignmentOperator() && M->isDefaulted() && !M->isDeleted())
      return false;
    if (M->isMoveAssignmentOperator() && M->isDefaulted() && !M->isDeleted())
      return false;
  }
  // FIXME: find ways to identifier correct handle capture this lambda
  return true;
}

} // namespace

void CaptureThisByFieldCheck::registerMatchers(MatchFinder *Finder) {
  auto IsStdFunctionField =
      fieldDecl(hasType(cxxRecordDecl(hasName("::std::function"))))
          .bind("field");
  auto CaptureThis = lambdaCapture(anyOf(
      // [this]
      capturesThis(),
      // [self = this]
      capturesVar(varDecl(hasInitializer(cxxThisExpr())))));
  auto IsInitWithLambda = cxxConstructExpr(hasArgument(
      0,
      lambdaExpr(hasAnyCapture(CaptureThis.bind("capture"))).bind("lambda")));
  Finder->addMatcher(
      cxxRecordDecl(
          has(cxxConstructorDecl(
              unless(isCopyConstructor()), unless(isMoveConstructor()),
              hasAnyConstructorInitializer(cxxCtorInitializer(
                  isMemberInitializer(), forField(IsStdFunctionField),
                  withInitializer(IsInitWithLambda))))),
          unless(correctHandleCaptureThisLambda())),
      this);
}

void CaptureThisByFieldCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Capture = Result.Nodes.getNodeAs<LambdaCapture>("capture");
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field");
  diag(Lambda->getBeginLoc(),
       "using lambda expressions to capture this and storing it in class "
       "member will cause potential variable lifetime issue when the class "
       "instance is moved or copied")
      << Capture->getLocation();
  diag(Field->getLocation(),
       "'std::function' that stores captured this and becomes invalid during "
       "copying or moving",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
