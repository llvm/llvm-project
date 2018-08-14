//===--- AvoidSpinlockCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidSpinlockCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

void AvoidSpinlockCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee((functionDecl(hasAnyName(
                   "OSSpinlockLock", "OSSpinlockUnlock", "OSSpinlockTry")))))
          .bind("spinlock"),
      this);
}

void AvoidSpinlockCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CallExpr>("spinlock");
  diag(MatchedExpr->getBeginLoc(),
       "use os_unfair_lock_lock() or dispatch queue APIs instead of the "
       "deprecated OSSpinLock");
}

}  // namespace objc
}  // namespace tidy
}  // namespace clang
