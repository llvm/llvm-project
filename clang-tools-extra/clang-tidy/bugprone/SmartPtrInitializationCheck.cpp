//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(Expr, isNewExpression) {
  return isa<CXXNewExpr>(Node);
}

AST_MATCHER(Expr, isReleaseCall) {
  const auto *call = dyn_cast<CallExpr>(&Node);
  if (!call)
    return false;

  const auto *method = dyn_cast<CXXMemberCallExpr>(call);
  if (!method)
    return false;

  const auto *member = method->getMethodDecl();
  if (!member)
    return false;

  return member->getName() == "release";
}

AST_MATCHER(Expr, isMakeUniqueOrSharedCall) {
  const auto *call = dyn_cast<CallExpr>(&Node);
  if (!call)
    return false;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return false;

  StringRef name = callee->getName();
  if (name != "make_unique" && name != "make_shared")
    return false;

  // Check if it's in std namespace by checking the qualified name
  std::string qualifiedName = callee->getQualifiedNameAsString();
  return qualifiedName == "std::make_unique" || qualifiedName == "std::make_shared";
}

AST_MATCHER(CXXConstructExpr, hasCustomDeleter) {
  if (Node.getNumArgs() < 2)
    return false;

  // Check if the second argument is a deleter
  const Expr *deleterArg = Node.getArg(1);
  if (!deleterArg)
    return false;

  // Check if this is a smart pointer construction with custom deleter
  const auto *record = Node.getConstructor()->getParent();
  if (!record)
    return false;

  std::string typeName = record->getQualifiedNameAsString();
  return typeName == "std::shared_ptr" || typeName == "std::unique_ptr";
}

AST_MATCHER(CallExpr, isResetWithCustomDeleter) {
  const auto *memberCall = dyn_cast<CXXMemberCallExpr>(&Node);
  if (!memberCall)
    return false;

  const auto *method = memberCall->getMethodDecl();
  if (!method || method->getName() != "reset")
    return false;

  if (Node.getNumArgs() < 2)
    return false;

  const auto *record = method->getParent();
  if (!record)
    return false;

  std::string typeName = record->getQualifiedNameAsString();
  return typeName == "std::shared_ptr" || typeName == "std::unique_ptr";
}

} // namespace

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for smart pointer constructors
  auto smartPtrConstructorMatcher = cxxConstructExpr(
      hasDeclaration(
          cxxConstructorDecl(ofClass(hasAnyName("std::shared_ptr", 
                                               "std::unique_ptr")))),
      hasArgument(0, expr().bind("pointer-arg")),
      unless(hasCustomDeleter()),
      unless(hasArgument(0, isNewExpression())),
      unless(hasArgument(0, isReleaseCall())),
      unless(hasArgument(0, isMakeUniqueOrSharedCall()))
  ).bind("constructor");

  // Matcher for reset() calls
  auto resetCallMatcher = cxxMemberCallExpr(
      on(hasType(cxxRecordDecl(hasAnyName("std::shared_ptr", "std::unique_ptr")))),
      callee(cxxMethodDecl(hasName("reset"))),
      hasArgument(0, expr().bind("pointer-arg")),
      unless(isResetWithCustomDeleter()),
      unless(hasArgument(0, isNewExpression())),
      unless(hasArgument(0, isReleaseCall())),
      unless(hasArgument(0, isMakeUniqueOrSharedCall()))
  ).bind("reset-call");

  Finder->addMatcher(smartPtrConstructorMatcher, this);
  Finder->addMatcher(resetCallMatcher, this);
}

void SmartPtrInitializationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *pointerArg = Result.Nodes.getNodeAs<Expr>("pointer-arg");
  if (!pointerArg)
    return;

  // Skip if the pointer is a null pointer
  if (pointerArg->isNullPointerConstant(*Result.Context,
                                        Expr::NPC_ValueDependentIsNotNull))
    return;

  // Check if the expression is a call to a function returning a pointer
  bool isFunctionReturn = false;
  if (const auto *call = dyn_cast<CallExpr>(pointerArg)) {
    if (call->getDirectCallee()) {
      isFunctionReturn = true;
    }
  }

  // Check if it's taking address of something
  bool isAddressOf = isa<UnaryOperator>(pointerArg) &&
                     cast<UnaryOperator>(pointerArg)->getOpcode() == UO_AddrOf;

  // Check if it's getting pointer from reference
  const Expr *innerExpr = pointerArg->IgnoreParenCasts();
  if (const auto *unaryOp = dyn_cast<UnaryOperator>(innerExpr)) {
    if (unaryOp->getOpcode() == UO_AddrOf) {
      isAddressOf = true;
    }
  }

  // Also check for member expressions that might return references
  if (const auto *memberExpr = dyn_cast<MemberExpr>(innerExpr)) {
    if (memberExpr->isArrow()) {
      // arrow operator returns pointer, not reference
      isAddressOf = false;
    }
  }

  if (isFunctionReturn || isAddressOf) {
    std::string message;
    const SourceLocation loc = pointerArg->getBeginLoc();

    if (const auto *constructor = 
            Result.Nodes.getNodeAs<CXXConstructExpr>("constructor")) {
      const auto *decl = constructor->getConstructor();
      if (decl) {
        const auto *record = decl->getParent();
        if (record) {
          std::string typeName = record->getQualifiedNameAsString();
          message = "passing a raw pointer '" + 
                    getPointerDescription(pointerArg, *Result.Context) +
                    "' to " + typeName + 
                    " constructor may cause double deletion";
        }
      }
    } else if (const auto *resetCall = 
                   Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset-call")) {
      const auto *method = resetCall->getMethodDecl();
      if (method) {
        const auto *record = method->getParent();
        if (record) {
          std::string typeName = record->getQualifiedNameAsString();
          message = "passing a raw pointer '" + 
                    getPointerDescription(pointerArg, *Result.Context) +
                    "' to " + typeName + 
                    "::reset() may cause double deletion";
        }
      }
    }

    if (!message.empty()) {
      diag(loc, message);
    }
  }
}

std::string SmartPtrInitializationCheck::getPointerDescription(
    const Expr *PointerExpr, ASTContext &Context) {
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
