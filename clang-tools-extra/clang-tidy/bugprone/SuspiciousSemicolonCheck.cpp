//===--- SuspiciousSemicolonCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousSemicolonCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void SuspiciousSemicolonCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      stmt(anyOf(ifStmt(hasThen(nullStmt().bind("semi")),
                        unless(hasElse(stmt()))),
                 forStmt(hasBody(nullStmt().bind("semi"))),
                 cxxForRangeStmt(hasBody(nullStmt().bind("semi"))),
                 whileStmt(hasBody(nullStmt().bind("semi")))))
          .bind("stmt"),
      this);
}

void SuspiciousSemicolonCheck::check(const MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *Semicolon = Result.Nodes.getNodeAs<NullStmt>("semi");
  SourceLocation LocStart = Semicolon->getBeginLoc();

  if (LocStart.isMacroID())
    return;

  ASTContext &Ctxt = *Result.Context;
  auto Token = utils::lexer::getPreviousToken(LocStart, Ctxt.getSourceManager(),
                                              Ctxt.getLangOpts());
  auto &SM = *Result.SourceManager;
  unsigned SemicolonLine = SM.getSpellingLineNumber(LocStart);

  const auto *Statement = Result.Nodes.getNodeAs<Stmt>("stmt");
  const bool IsIfStmt = isa<IfStmt>(Statement);

  if (!IsIfStmt &&
      SM.getSpellingLineNumber(Token.getLocation()) != SemicolonLine)
    return;

  SourceLocation LocEnd = Semicolon->getEndLoc();
  FileID FID = SM.getFileID(LocEnd);
  llvm::MemoryBuffer *Buffer = SM.getBuffer(FID, LocEnd);
  Lexer Lexer(SM.getLocForStartOfFile(FID), Ctxt.getLangOpts(),
              Buffer->getBufferStart(), SM.getCharacterData(LocEnd) + 1,
              Buffer->getBufferEnd());
  if (Lexer.LexFromRawLexer(Token))
    return;

  unsigned BaseIndent = SM.getSpellingColumnNumber(Statement->getBeginLoc());
  unsigned NewTokenIndent = SM.getSpellingColumnNumber(Token.getLocation());
  unsigned NewTokenLine = SM.getSpellingLineNumber(Token.getLocation());

  if (!IsIfStmt && NewTokenIndent <= BaseIndent &&
      Token.getKind() != tok::l_brace && NewTokenLine != SemicolonLine)
    return;

  diag(LocStart, "potentially unintended semicolon")
      << FixItHint::CreateRemoval(SourceRange(LocStart, LocEnd));
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
