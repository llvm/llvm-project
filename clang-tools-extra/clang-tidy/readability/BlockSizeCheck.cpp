//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BlockSizeCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

const unsigned DefaultIfLineCountThreshold = 20;
const unsigned DefaultForLineCountThreshold = 30;
const unsigned DefaultWhileLineCountThreshold = 30;

BlockSizeCheck::BlockSizeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IfLineCountThreshold(
          Options.get("IfLineCountThreshold", DefaultIfLineCountThreshold)),
      ForLineCountThreshold(
          Options.get("ForLineCountThreshold", DefaultForLineCountThreshold)),
      WhileLineCountThreshold(Options.get("WhileLineCountThreshold",
                                          DefaultWhileLineCountThreshold)) {}

void BlockSizeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IfLineCountThreshold", IfLineCountThreshold);
  Options.store(Opts, "ForLineCountThreshold", ForLineCountThreshold);
  Options.store(Opts, "WhileLineCountThreshold", WhileLineCountThreshold);
}

void BlockSizeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ifStmt().bind("if"), this);
  Finder->addMatcher(forStmt().bind("for"), this);
  Finder->addMatcher(whileStmt().bind("while"), this);
}

void BlockSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &SrcMgr = Result.SourceManager;

  if (const auto *IfBlk = Result.Nodes.getNodeAs<IfStmt>("if")) {
    const auto *ElseBlk = IfBlk->getElse();

    const unsigned FirstLine =
        SrcMgr->getSpellingLineNumber(IfBlk->getBeginLoc());
    const unsigned LastLine = [&]() {
      if (ElseBlk != nullptr)
        return SrcMgr->getSpellingLineNumber(ElseBlk->getBeginLoc());
      return SrcMgr->getSpellingLineNumber(IfBlk->getEndLoc());
    }();
    const unsigned LineCount = LastLine - FirstLine + 1;

    if (LineCount > IfLineCountThreshold)
      diag(IfBlk->getBeginLoc(),
           "if block spans %0 lines of code, which exceeds "
           "the threshold of %1 lines")
          << LineCount << IfLineCountThreshold;

    if (ElseBlk != nullptr &&
        isa<CompoundStmt>(ElseBlk)) { // i.e. is not an else if
      const unsigned ElseLastLine =
          SrcMgr->getSpellingLineNumber(IfBlk->getEndLoc());
      const unsigned ElseLineCount = ElseLastLine - LastLine + 1;

      if (ElseLineCount > IfLineCountThreshold)
        diag(ElseBlk->getBeginLoc(),
             "else block spans %0 lines of code, which exceeds "
             "the threshold of %1 lines")
            << ElseLineCount << IfLineCountThreshold;
    }

    return;
  }

  if (const auto *ForLoop = Result.Nodes.getNodeAs<ForStmt>("for")) {
    const unsigned FirstLine =
        SrcMgr->getSpellingLineNumber(ForLoop->getBeginLoc());
    const unsigned LastLine =
        SrcMgr->getSpellingLineNumber(ForLoop->getEndLoc());
    const unsigned LineCount = LastLine - FirstLine + 1;

    if (LineCount > ForLineCountThreshold) {
      diag(ForLoop->getBeginLoc(), "for loop spans %0 lines of code, which "
                                   "exceeds the threshold of %1 lines")
          << LineCount << ForLineCountThreshold;
    }

    return;
  }

  if (const auto *WhileLoop = Result.Nodes.getNodeAs<WhileStmt>("while")) {
    const unsigned FirstLine =
        SrcMgr->getSpellingLineNumber(WhileLoop->getBeginLoc());
    const unsigned LastLine =
        SrcMgr->getSpellingLineNumber(WhileLoop->getEndLoc());
    const unsigned LineCount = LastLine - FirstLine + 1;

    if (LineCount > WhileLineCountThreshold) {
      diag(WhileLoop->getBeginLoc(), "while loop spans %0 lines of code, which "
                                     "exceeds the threshold of %1 lines")
          << LineCount << WhileLineCountThreshold;
    }

    return;
  }
}

} // namespace clang::tidy::readability
