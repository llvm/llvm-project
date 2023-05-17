//===--- AvoidThrowingObjCExceptionCheck.cpp - clang-tidy------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidThrowingObjCExceptionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google::objc {

void AvoidThrowingObjCExceptionCheck::registerMatchers(MatchFinder *Finder) {

  Finder->addMatcher(objcThrowStmt().bind("throwStmt"), this);
  Finder->addMatcher(
      objcMessageExpr(anyOf(hasSelector("raise:format:"),
                            hasSelector("raise:format:arguments:")),
                      hasReceiverType(asString("NSException")))
          .bind("raiseException"),
      this);
}

void AvoidThrowingObjCExceptionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedStmt =
      Result.Nodes.getNodeAs<ObjCAtThrowStmt>("throwStmt");
  const auto *MatchedExpr =
      Result.Nodes.getNodeAs<ObjCMessageExpr>("raiseException");
  auto SourceLoc = MatchedStmt == nullptr ? MatchedExpr->getSelectorStartLoc()
                                          : MatchedStmt->getThrowLoc();

  // Early return on invalid locations.
  if (SourceLoc.isInvalid())
    return;

  // If the match location was in a macro, check if the macro was in a system
  // header.
  if (SourceLoc.isMacroID()) {
    SourceManager &SM = *Result.SourceManager;
    auto MacroLoc = SM.getImmediateMacroCallerLoc(SourceLoc);

    // Matches in system header macros should be ignored.
    if (SM.isInSystemHeader(MacroLoc))
      return;
  }

  diag(SourceLoc,
       "pass in NSError ** instead of throwing exception to indicate "
       "Objective-C errors");
}

} // namespace clang::tidy::google::objc
