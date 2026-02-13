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

// Helper function to check if a smart pointer record type has a custom deleter
// based on the record type and number of arguments in the call/constructor
static bool hasCustomDeleterForRecord(const CXXRecordDecl *Record,
                                      unsigned NumArgs) {
  if (!Record)
    return false;

  const std::string typeName = Record->getQualifiedNameAsString();
  if (typeName == "std::shared_ptr") {
    // Check if the second argument is a deleter
    if (NumArgs >= 2)
      return true;
  } else if (typeName == "std::unique_ptr") {
    // Check if the second template argument is a deleter
    const auto *templateSpec =
        dyn_cast<ClassTemplateSpecializationDecl>(Record);
    if (!templateSpec)
      return false;

    const auto &templateArgs = templateSpec->getTemplateArgs();
    // unique_ptr has at least 1 template argument (the pointer type)
    // If it has 2, the second one is the deleter type
    if (templateArgs.size() >= 2) {
      const auto &deleterArg = templateArgs[1];
      // The deleter must be a type
      if (deleterArg.getKind() == TemplateArgument::Type) {
        QualType deleterType = deleterArg.getAsType();
        if (auto *deleterRecord = deleterType->getAsCXXRecordDecl()) {
          const std::string DeleterTypeName =
              deleterRecord->getQualifiedNameAsString();
          if (DeleterTypeName != "std::default_delete")
            return true;
        }
      }
    }
  }

  return false;
}

// TODO: all types must be in config
// TODO: boost::shared_ptr and boost::unique_ptr
// TODO: reset and release must be in config
AST_MATCHER(Stmt, hasCustomDeleter) {
  const auto *constructExpr = dyn_cast<CXXConstructExpr>(&Node);
  if (constructExpr) {
    const auto *record = constructExpr->getConstructor()->getParent();
    return hasCustomDeleterForRecord(record, constructExpr->getNumArgs());
  }

  const auto *callExpr = dyn_cast<CallExpr>(&Node);
  if (callExpr) {
    const auto *memberCall = dyn_cast<CXXMemberCallExpr>(callExpr);
    if (!memberCall)
      return false;

    const auto *method = memberCall->getMethodDecl();
    if (!method || method->getName() != "reset")
      return false;

    const auto *record = method->getParent();
    return hasCustomDeleterForRecord(record, callExpr->getNumArgs());
  }

  return false;
}

} // namespace

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  auto ReleaseCallMatcher =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));
  // Matcher for smart pointer constructors
  auto smartPtrConstructorMatcher =
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(
              ofClass(hasAnyName("std::shared_ptr", "std::unique_ptr")),
              unless(anyOf(isCopyConstructor(), isMoveConstructor())))),
          hasArgument(0, expr(unless(nullPointerConstant())).bind("pointer-arg")),
          unless(hasCustomDeleter()), unless(hasArgument(0, cxxNewExpr())),
          unless(hasArgument(0, ReleaseCallMatcher)))
          .bind("constructor");

  // Matcher for reset() calls
  auto resetCallMatcher =
      cxxMemberCallExpr(on(hasType(cxxRecordDecl(
                            hasAnyName("std::shared_ptr", "std::unique_ptr")))),
                        callee(cxxMethodDecl(hasName("reset"))),
                        hasArgument(0, expr(unless(nullPointerConstant())).bind("pointer-arg")),
                        unless(hasCustomDeleter()),
                        unless(hasArgument(0, cxxNewExpr())),
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

  if (MD) {
    const auto *record = MD->getParent();
    if (record) {
      const std::string typeName = record->getQualifiedNameAsString();
      diag(loc,
            "passing a raw pointer '%0' to %1%2 may cause double deletion")
          << getPointerDescription(pointerArg, *Result.Context) << typeName
          << (constructor ? " constructor" : "::reset()");
    }
  }
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
