//===--- NoEscapeCheck.cpp - clang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoEscapeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void NoEscapeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::dispatch_async"))),
                              argumentCountIs(2),
                              hasArgument(1, blockExpr().bind("arg-block"))),
                     this);
  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::dispatch_after"))),
                              argumentCountIs(3),
                              hasArgument(2, blockExpr().bind("arg-block"))),
                     this);
}

void NoEscapeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedEscapingBlock =
      Result.Nodes.getNodeAs<BlockExpr>("arg-block");
  const BlockDecl *EscapingBlockDecl = MatchedEscapingBlock->getBlockDecl();
  for (const BlockDecl::Capture &CapturedVar : EscapingBlockDecl->captures()) {
    const VarDecl *Var = CapturedVar.getVariable();
    if (Var && Var->hasAttr<NoEscapeAttr>()) {
      // FIXME: Add a method to get the location of the use of a CapturedVar so
      // that we can diagnose the use of the pointer instead of the block.
      diag(MatchedEscapingBlock->getBeginLoc(),
           "pointer %0 with attribute 'noescape' is captured by an "
           "asynchronously-executed block")
          << Var;
      diag(Var->getBeginLoc(), "the 'noescape' attribute is declared here.",
           DiagnosticIDs::Note);
    }
  }
}

} // namespace clang::tidy::bugprone
