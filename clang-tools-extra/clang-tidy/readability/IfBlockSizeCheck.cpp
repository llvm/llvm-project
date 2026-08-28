//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IfBlockSizeCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void IfBlockSizeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ifStmt().bind("if"), this);
}

void IfBlockSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto& SrcMgr = Result.SourceManager;

  const auto *IfBlk = Result.Nodes.getNodeAs<IfStmt>("if");
  const unsigned FirstLine = SrcMgr->getSpellingLineNumber(IfBlk->getBeginLoc());
  const unsigned LastLine = [&](){
    if (const auto *ElseBlk = IfBlk->getElse())
      return SrcMgr->getSpellingLineNumber(ElseBlk->getBeginLoc());
    return SrcMgr->getSpellingLineNumber(IfBlk->getEndLoc());
  }();
  const unsigned LineCount = LastLine - FirstLine  + 1;

  if (LineCount <= LineCountThreshold)
    return;

  diag(IfBlk->getBeginLoc(), "if block spans %0 lines of code, which exceeds the threshold of %1 lines")
      << LineCount
      << LineCountThreshold;
}

} // namespace clang::tidy::readability
