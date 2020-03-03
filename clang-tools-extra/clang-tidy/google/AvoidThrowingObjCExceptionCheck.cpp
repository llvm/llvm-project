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

namespace clang {
namespace tidy {
namespace google {
namespace objc {

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
  diag(SourceLoc,
       "pass in NSError ** instead of throwing exception to indicate "
       "Objective-C errors");
}

}  // namespace objc
}  // namespace google
}  // namespace tidy
}  // namespace clang
