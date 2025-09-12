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

  for (const CXXConstructorDecl *C : Node.ctors()) {
    if (C->isCopyOrMoveConstructor() && C->isDefaulted() && !C->isDeleted())
      return false;
  }
  for (const CXXMethodDecl *M : Node.methods()) {
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
constexpr const char *DefaultBindFunctions =
    "::std::bind;::boost::bind;::std::bind_front;::std::bind_back;"
    "::boost::compat::bind_front;::boost::compat::bind_back";

CapturingThisInMemberVariableCheck::CapturingThisInMemberVariableCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      FunctionWrapperTypes(utils::options::parseStringList(
          Options.get("FunctionWrapperTypes", DefaultFunctionWrapperTypes))),
      BindFunctions(utils::options::parseStringList(
          Options.get("BindFunctions", DefaultBindFunctions))) {}
void CapturingThisInMemberVariableCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionWrapperTypes",
                utils::options::serializeStringList(FunctionWrapperTypes));
  Options.store(Opts, "BindFunctions",
                utils::options::serializeStringList(BindFunctions));
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
      lambdaExpr(hasAnyCapture(CaptureThis)).bind("lambda");

  auto IsBindCapturingThis =
      callExpr(
          callee(functionDecl(matchers::matchesAnyListedName(BindFunctions))
                     .bind("callee")),
          hasAnyArgument(cxxThisExpr()))
          .bind("bind");

  auto IsInitWithLambdaOrBind =
      anyOf(IsLambdaCapturingThis, IsBindCapturingThis,
            cxxConstructExpr(hasArgument(
                0, anyOf(IsLambdaCapturingThis, IsBindCapturingThis))));

  Finder->addMatcher(
      cxxRecordDecl(
          anyOf(has(cxxConstructorDecl(
                    unless(isCopyConstructor()), unless(isMoveConstructor()),
                    hasAnyConstructorInitializer(cxxCtorInitializer(
                        isMemberInitializer(), forField(IsStdFunctionField),
                        withInitializer(IsInitWithLambdaOrBind))))),
                has(fieldDecl(IsStdFunctionField,
                              hasInClassInitializer(IsInitWithLambdaOrBind)))),
          unless(correctHandleCaptureThisLambda())),
      this);
}
void CapturingThisInMemberVariableCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda")) {
    diag(Lambda->getBeginLoc(),
         "'this' captured by a lambda and stored in a class member variable; "
         "disable implicit class copying/moving to prevent potential "
         "use-after-free");
  } else if (const auto *Bind = Result.Nodes.getNodeAs<CallExpr>("bind")) {
    const auto *Callee = Result.Nodes.getNodeAs<FunctionDecl>("callee");
    assert(Callee);
    diag(Bind->getBeginLoc(),
         "'this' captured by a '%0' call and stored in a class member "
         "variable; disable implicit class copying/moving to prevent potential "
         "use-after-free")
        << Callee->getQualifiedNameAsString();
  }

  const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field");
  assert(Field);

  diag(Field->getLocation(),
       "class member of type '%0' that stores captured 'this'",
       DiagnosticIDs::Note)
      << Field->getType().getAsString();
}

} // namespace clang::tidy::bugprone
