//===--- SuspiciousSemicolonCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SuspiciousSemicolonCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

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
  SourceLocation LocStart = Semicolon->getLocStart();

  if (LocStart.isMacroID())
    return;

  ASTContext &Ctxt = *Result.Context;
  auto Token = utils::lexer::getPreviousToken(Ctxt, LocStart);
  auto &SM = *Result.SourceManager;
  unsigned SemicolonLine = SM.getSpellingLineNumber(LocStart);

  const auto *Statement = Result.Nodes.getNodeAs<Stmt>("stmt");
  const bool IsIfStmt = isa<IfStmt>(Statement);

  if (!IsIfStmt &&
      SM.getSpellingLineNumber(Token.getLocation()) != SemicolonLine)
    return;

  SourceLocation LocEnd = Semicolon->getLocEnd();
  FileID FID = SM.getFileID(LocEnd);
  llvm::MemoryBuffer *Buffer = SM.getBuffer(FID, LocEnd);
  Lexer Lexer(SM.getLocForStartOfFile(FID), Ctxt.getLangOpts(),
              Buffer->getBufferStart(), SM.getCharacterData(LocEnd) + 1,
              Buffer->getBufferEnd());
  if (Lexer.LexFromRawLexer(Token))
    return;

  unsigned BaseIndent = SM.getSpellingColumnNumber(Statement->getLocStart());
  unsigned NewTokenIndent = SM.getSpellingColumnNumber(Token.getLocation());
  unsigned NewTokenLine = SM.getSpellingLineNumber(Token.getLocation());

  if (!IsIfStmt && NewTokenIndent <= BaseIndent &&
      Token.getKind() != tok::l_brace && NewTokenLine != SemicolonLine)
    return;

  diag(LocStart, "potentially unintended semicolon")
      << FixItHint::CreateRemoval(SourceRange(LocStart, LocEnd));
}

} // namespace misc
} // namespace tidy
} // namespace clang
