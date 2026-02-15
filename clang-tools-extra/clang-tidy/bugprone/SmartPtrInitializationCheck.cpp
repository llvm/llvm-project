//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

const auto DefaultSharedPointers = "::std::shared_ptr;::boost::shared_ptr";
const auto DefaultUniquePointers = "::std::unique_ptr";
const auto DefaultDefaultDeleters = "::std::default_delete";

} // namespace

SmartPtrInitializationCheck::SmartPtrInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointers(utils::options::parseStringList(
          Options.get("SharedPointers", DefaultSharedPointers))),
      UniquePointers(utils::options::parseStringList(
          Options.get("UniquePointers", DefaultUniquePointers))),
      DefaultDeleters(utils::options::parseStringList(
          Options.get("DefaultDeleters", DefaultDefaultDeleters))) {}

void SmartPtrInitializationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SharedPointers",
                utils::options::serializeStringList(SharedPointers));
  Options.store(Opts, "UniquePointers",
                utils::options::serializeStringList(UniquePointers));
  Options.store(Opts, "DefaultDeleters",
                utils::options::serializeStringList(DefaultDeleters));
}

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  auto ReleaseCallMatcher =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));

  // Build matchers for the smart pointer types
  auto SharedPtrMatcher = hasAnyName(SharedPointers);
  auto UniquePtrMatcher = hasAnyName(UniquePointers);
  auto AllSmartPtrMatcher = anyOf(SharedPtrMatcher, UniquePtrMatcher);

  // Matcher for unique_ptr types with custom deleters
  auto DefaultDeleterMatcher = hasAnyName(DefaultDeleters);
  auto UniquePtrWithCustomDeleter = classTemplateSpecializationDecl(
      UniquePtrMatcher, templateArgumentCountIs(2),
      hasTemplateArgument(1, refersToType(unless(hasDeclaration(cxxRecordDecl(
                                 DefaultDeleterMatcher))))));

  // Matcher for smart pointer constructors
  // Exclude constructors with custom deleters:
  // - shared_ptr with 2+ arguments (second is deleter)
  // - unique_ptr with 2+ template args where second is not default_delete
  auto HasCustomDeleter = anyOf(
      allOf(hasDeclaration(cxxConstructorDecl(
                ofClass(SharedPtrMatcher))),
            hasArgument(1, anything())),
      hasDeclaration(cxxConstructorDecl(ofClass(UniquePtrWithCustomDeleter))));

  auto smartPtrConstructorMatcher =
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(
              ofClass(AllSmartPtrMatcher),
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
      anyOf(allOf(on(hasType(cxxRecordDecl(SharedPtrMatcher))),
                  hasArgument(1, anything())),
            on(hasType(qualType(hasDeclaration(UniquePtrWithCustomDeleter)))));

  auto resetCallMatcher =
      cxxMemberCallExpr(
          on(hasType(cxxRecordDecl(AllSmartPtrMatcher))),
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
