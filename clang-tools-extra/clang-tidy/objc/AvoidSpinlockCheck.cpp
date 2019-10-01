//===--- AvoidSpinlockCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
