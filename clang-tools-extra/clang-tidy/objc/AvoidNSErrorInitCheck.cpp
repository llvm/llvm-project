//===--- AvoidNSErrorInitCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidNSErrorInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

void AvoidNSErrorInitCheck::registerMatchers(MatchFinder *Finder) {
  // this check should only be applied to ObjC sources.
  if (!getLangOpts().ObjC1 && !getLangOpts().ObjC2) {
    return;
  }
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

}  // namespace objc
}  // namespace tidy
}  // namespace clang
