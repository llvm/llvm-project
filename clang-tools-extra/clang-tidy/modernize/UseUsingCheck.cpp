//===--- UseUsingCheck.cpp - clang-tidy------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseUsingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void UseUsingCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;
  Finder->addMatcher(typedefDecl().bind("typedef"), this);
}

// Checks if 'typedef' keyword can be removed - we do it only if
// it is the only declaration in a declaration chain.
static bool CheckRemoval(SourceManager &SM, const SourceLocation &LocStart,
                         const SourceLocation &LocEnd, ASTContext &Context) {
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(LocStart);
  StringRef File = SM.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer DeclLexer(SM.getLocForStartOfFile(LocInfo.first), Context.getLangOpts(),
                  File.begin(), TokenBegin, File.end());

  Token Tok;
  int ParenLevel = 0;

  while (!DeclLexer.LexFromRawLexer(Tok)) {
    if (SM.isBeforeInTranslationUnit(LocEnd, Tok.getLocation()))
      break;
    if (Tok.getKind() == tok::TokenKind::l_paren)
      ParenLevel++;
    if (Tok.getKind() == tok::TokenKind::r_paren)
      ParenLevel--;
    if (Tok.getKind() == tok::TokenKind::semi)
      break;
    // if there is comma and we are not between open parenthesis then it is
    // two or more declatarions in this chain
    if (ParenLevel == 0 && Tok.getKind() == tok::TokenKind::comma)
      return false;

    if (Tok.is(tok::TokenKind::raw_identifier)) {
      if (Tok.getRawIdentifier() == "typedef")
        return true;
    }
  }

  return false;
}

void UseUsingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<TypedefDecl>("typedef");
  if (MatchedDecl->getLocation().isInvalid())
    return;

  auto &Context = *Result.Context;
  auto &SM = *Result.SourceManager;

  auto Diag =
      diag(MatchedDecl->getLocStart(), "use 'using' instead of 'typedef'");

  if (MatchedDecl->getLocStart().isMacroID())
    return;

  if (CheckRemoval(SM, MatchedDecl->getLocStart(), MatchedDecl->getLocEnd(),
                   Context)) {
    Diag << FixItHint::CreateReplacement(
        MatchedDecl->getSourceRange(),
        "using " + MatchedDecl->getNameAsString() + " = " +
            MatchedDecl->getUnderlyingType().getAsString(getLangOpts()));
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
