//===--- AvoidNSObjectNewCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNSObjectNewCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/FormatVariadic.h"
#include <map>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::google::objc {

static bool isMessageExpressionInsideMacro(const ObjCMessageExpr *Expr) {
  SourceLocation ReceiverLocation = Expr->getReceiverRange().getBegin();
  if (ReceiverLocation.isMacroID())
    return true;

  SourceLocation SelectorLocation = Expr->getSelectorStartLoc();
  if (SelectorLocation.isMacroID())
    return true;

  return false;
}

// Walk up the class hierarchy looking for an -init method, returning true
// if one is found and has not been marked unavailable.
static bool isInitMethodAvailable(const ObjCInterfaceDecl *ClassDecl) {
  while (ClassDecl != nullptr) {
    for (const auto *MethodDecl : ClassDecl->instance_methods()) {
      if (MethodDecl->getSelector().getAsString() == "init")
        return !MethodDecl->isUnavailable();
    }
    ClassDecl = ClassDecl->getSuperClass();
  }

  // No -init method found in the class hierarchy. This should occur only rarely
  // in Objective-C code, and only really applies to classes not derived from
  // NSObject.
  return false;
}

// Returns the string for the Objective-C message receiver. Keeps any generics
// included in the receiver class type, which are stripped if the class type is
// used. While the generics arguments will not make any difference to the
// returned code at this time, the style guide allows them and they should be
// left in any fix-it hint.
static StringRef getReceiverString(SourceRange ReceiverRange,
                                   const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(ReceiverRange), SM, LangOpts);
  return Lexer::getSourceText(CharRange, SM, LangOpts);
}

static FixItHint getCallFixItHint(const ObjCMessageExpr *Expr,
                                  const SourceManager &SM,
                                  const LangOptions &LangOpts) {
  // Check whether the messaged class has a known factory method to use instead
  // of -init.
  StringRef Receiver =
      getReceiverString(Expr->getReceiverRange(), SM, LangOpts);
  // Some classes should use standard factory methods instead of alloc/init.
  std::map<StringRef, StringRef> ClassToFactoryMethodMap = {{"NSDate", "date"},
                                                            {"NSNull", "null"}};
  auto FoundClassFactory = ClassToFactoryMethodMap.find(Receiver);
  if (FoundClassFactory != ClassToFactoryMethodMap.end()) {
    StringRef ClassName = FoundClassFactory->first;
    StringRef FactorySelector = FoundClassFactory->second;
    std::string NewCall =
        std::string(llvm::formatv("[{0} {1}]", ClassName, FactorySelector));
    return FixItHint::CreateReplacement(Expr->getSourceRange(), NewCall);
  }

  if (isInitMethodAvailable(Expr->getReceiverInterface())) {
    std::string NewCall =
        std::string(llvm::formatv("[[{0} alloc] init]", Receiver));
    return FixItHint::CreateReplacement(Expr->getSourceRange(), NewCall);
  }

  return {}; // No known replacement available.
}

void AvoidNSObjectNewCheck::registerMatchers(MatchFinder *Finder) {
  // Add two matchers, to catch calls to +new and implementations of +new.
  Finder->addMatcher(
      objcMessageExpr(isClassMessage(), hasSelector("new")).bind("new_call"),
      this);
  Finder->addMatcher(
      objcMethodDecl(isClassMethod(), isDefinition(), hasName("new"))
          .bind("new_override"),
      this);
}

void AvoidNSObjectNewCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *CallExpr =
          Result.Nodes.getNodeAs<ObjCMessageExpr>("new_call")) {
    // Don't warn if the call expression originates from a macro expansion.
    if (isMessageExpressionInsideMacro(CallExpr))
      return;

    diag(CallExpr->getExprLoc(), "do not create objects with +new")
        << getCallFixItHint(CallExpr, *Result.SourceManager,
                            Result.Context->getLangOpts());
  }

  if (const auto *DeclExpr =
          Result.Nodes.getNodeAs<ObjCMethodDecl>("new_override")) {
    diag(DeclExpr->getBeginLoc(), "classes should not override +new");
  }
}

} // namespace clang::tidy::google::objc
