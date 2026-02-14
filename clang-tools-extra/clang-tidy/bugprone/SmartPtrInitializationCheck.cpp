//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

// TODO: all types must be in config
// TODO: boost::shared_ptr and boost::unique_ptr
// TODO: reset and release must be in config

} // namespace

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  auto ReleaseCallMatcher =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));

  auto UniquePtrWithCustomDeleter = classTemplateSpecializationDecl(
    hasName("std::unique_ptr"), templateArgumentCountIs(2),
    hasTemplateArgument(1, refersToType(unless(hasDeclaration(cxxRecordDecl(
                               hasName("std::default_delete")))))));

  // Matcher for smart pointer constructors
  // Exclude constructors with custom deleters:
  // - shared_ptr with 2+ arguments (second is deleter)
  // - unique_ptr with 2+ template args where second is not default_delete
  auto HasCustomDeleter = anyOf(
      allOf(hasDeclaration(
                cxxConstructorDecl(ofClass(hasName("std::shared_ptr")))),
            hasArgument(1, anything())),
      hasDeclaration(cxxConstructorDecl(ofClass(UniquePtrWithCustomDeleter))));

  auto smartPtrConstructorMatcher =
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(
              ofClass(hasAnyName("std::shared_ptr", "std::unique_ptr")),
              unless(anyOf(isCopyConstructor(), isMoveConstructor())))),
          hasArgument(0,
                      expr(unless(nullPointerConstant())).bind("pointer-arg")),
          unless(HasCustomDeleter), unless(hasArgument(0, cxxNewExpr())),
          unless(hasArgument(0, ReleaseCallMatcher)))
          .bind("constructor");

  // Matcher for reset() calls
  // Exclude reset() calls with custom deleters:
  // - shared_ptr with 2+ arguments (second is deleter)
  // - unique_ptr with custom deleter type (2+ template args where second is not
  // default_delete)
  auto HasCustomDeleterInReset =
      anyOf(allOf(on(hasType(cxxRecordDecl(hasName("std::shared_ptr")))),
                  hasArgument(1, anything())),
            on(hasType(qualType(hasDeclaration(UniquePtrWithCustomDeleter)))));

  auto resetCallMatcher =
      cxxMemberCallExpr(
          on(hasType(
              cxxRecordDecl(hasAnyName("std::shared_ptr", "std::unique_ptr")))),
          callee(cxxMethodDecl(hasName("reset"))),
          hasArgument(0,
                      expr(unless(nullPointerConstant())).bind("pointer-arg")),
          unless(HasCustomDeleterInReset), unless(hasArgument(0, cxxNewExpr())),
          unless(hasArgument(0, ReleaseCallMatcher)))
          .bind("reset-call");

  Finder->addMatcher(smartPtrConstructorMatcher, this);
  Finder->addMatcher(resetCallMatcher, this);
}

void SmartPtrInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *pointerArg = Result.Nodes.getNodeAs<Expr>("pointer-arg");
  const auto *constructor =
      Result.Nodes.getNodeAs<CXXConstructExpr>("constructor");
  const auto *ResetCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset-call");
  assert(pointerArg);

  const SourceLocation loc = pointerArg->getBeginLoc();
  const CXXMethodDecl *MD =
      constructor ? constructor->getConstructor()
                  : (ResetCall ? ResetCall->getMethodDecl() : nullptr);
  if (!MD)
    return;

  const auto *record = MD->getParent();
  if (!record)
    return;

  const std::string typeName = record->getQualifiedNameAsString();
  diag(loc, "passing a raw pointer '%0' to %1%2 may cause double deletion")
      << getPointerDescription(pointerArg, *Result.Context) << typeName
      << (constructor ? " constructor" : "::reset()");
}

std::string
SmartPtrInitializationCheck::getPointerDescription(const Expr *PointerExpr,
                                                   ASTContext &Context) {
  std::string Desc;
  llvm::raw_string_ostream OS(Desc);

  // Try to get a readable representation of the expression
  PrintingPolicy Policy(Context.getLangOpts());
  Policy.SuppressSpecifiers = false;
  Policy.SuppressTagKeyword = true;

  PointerExpr->printPretty(OS, nullptr, Policy);
  return OS.str();
}

} // namespace clang::tidy::bugprone
