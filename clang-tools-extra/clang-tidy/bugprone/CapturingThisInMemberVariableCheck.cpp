//===--- CapturingThisInMemberVariableCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CapturingThisInMemberVariableCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
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
    if (M->isCopyAssignmentOperator())
      llvm::errs() << M->isDeleted() << "\n";
    if (M->isCopyAssignmentOperator() && M->isDefaulted() && !M->isDeleted())
      return false;
    if (M->isMoveAssignmentOperator() && M->isDefaulted() && !M->isDeleted())
      return false;
  }
  // FIXME: find ways to identifier correct handle capture this lambda
  return true;
}

} // namespace

constexpr const char *DefaultFunctionWrapperTypes =
    "::std::function;::std::move_only_function;::boost::function";

CapturingThisInMemberVariableCheck::CapturingThisInMemberVariableCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      FunctionWrapperTypes(utils::options::parseStringList(
          Options.get("FunctionWrapperTypes", DefaultFunctionWrapperTypes))) {}
void CapturingThisInMemberVariableCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionWrapperTypes",
                utils::options::serializeStringList(FunctionWrapperTypes));
}

void CapturingThisInMemberVariableCheck::registerMatchers(MatchFinder *Finder) {
  auto IsStdFunctionField =
      fieldDecl(hasType(cxxRecordDecl(
                    matchers::matchesAnyListedName(FunctionWrapperTypes))))
          .bind("field");
  auto CaptureThis = lambdaCapture(anyOf(
      // [this]
      capturesThis(),
      // [self = this]
      capturesVar(varDecl(hasInitializer(cxxThisExpr())))));
  auto IsLambdaCapturingThis =
      lambdaExpr(hasAnyCapture(CaptureThis.bind("capture"))).bind("lambda");
  auto IsInitWithLambda =
      anyOf(IsLambdaCapturingThis,
            cxxConstructExpr(hasArgument(0, IsLambdaCapturingThis)));
  Finder->addMatcher(
      cxxRecordDecl(
          anyOf(has(cxxConstructorDecl(
                    unless(isCopyConstructor()), unless(isMoveConstructor()),
                    hasAnyConstructorInitializer(cxxCtorInitializer(
                        isMemberInitializer(), forField(IsStdFunctionField),
                        withInitializer(IsInitWithLambda))))),
                has(fieldDecl(IsStdFunctionField,
                              hasInClassInitializer(IsInitWithLambda)))),
          unless(correctHandleCaptureThisLambda())),
      this);
}

void CapturingThisInMemberVariableCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Capture = Result.Nodes.getNodeAs<LambdaCapture>("capture");
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field");
  diag(Lambda->getBeginLoc(),
       "'this' captured by a lambda and stored in a class member variable; "
       "disable implicit class copying/moving to prevent potential "
       "use-after-free")
      << Capture->getLocation();
  diag(Field->getLocation(),
       "class member of type '%0' that stores captured 'this'",
       DiagnosticIDs::Note)
      << Field->getType().getAsString();
}

} // namespace clang::tidy::bugprone
