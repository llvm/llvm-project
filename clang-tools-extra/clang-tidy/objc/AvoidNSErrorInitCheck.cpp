//===--- AvoidNSErrorInitCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNSErrorInitCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

void AvoidNSErrorInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(objcMessageExpr(hasSelector("init"),
                                     hasReceiverType(asString("NSError *")))
                         .bind("nserrorInit"),
                     this);
}

void AvoidNSErrorInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr =
      Result.Nodes.getNodeAs<ObjCMessageExpr>("nserrorInit");
  diag(MatchedExpr->getBeginLoc(),
       "use errorWithDomain:code:userInfo: or initWithDomain:code:userInfo: to "
       "create a new NSError");
}

} // namespace clang::tidy::objc
