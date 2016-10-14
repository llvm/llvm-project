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
                         const SourceLocation &LocEnd, ASTContext &Context,
                         SourceRange &ResultRange) {
  FileID FID = SM.getFileID(LocEnd);
  llvm::MemoryBuffer *Buffer = SM.getBuffer(FID, LocEnd);
  Lexer DeclLexer(SM.getLocForStartOfFile(FID), Context.getLangOpts(),
                  Buffer->getBufferStart(), SM.getCharacterData(LocStart),
                  Buffer->getBufferEnd());
  Token DeclToken;
  bool result = false;
  int parenthesisLevel = 0;

  while (!DeclLexer.LexFromRawLexer(DeclToken)) {
    if (DeclToken.getKind() == tok::TokenKind::l_paren)
      parenthesisLevel++;
    if (DeclToken.getKind() == tok::TokenKind::r_paren)
      parenthesisLevel--;
    if (DeclToken.getKind() == tok::TokenKind::semi)
      break;
    // if there is comma and we are not between open parenthesis then it is
    // two or more declatarions in this chain
    if (parenthesisLevel == 0 && DeclToken.getKind() == tok::TokenKind::comma)
      return false;

    if (DeclToken.isOneOf(tok::TokenKind::identifier,
                          tok::TokenKind::raw_identifier)) {
      auto TokenStr = DeclToken.getRawIdentifier().str();

      if (TokenStr == "typedef") {
        ResultRange =
            SourceRange(DeclToken.getLocation(), DeclToken.getEndLoc());
        result = true;
      }
    }
  }
  // assert if there was keyword 'typedef' in declaration
  assert(result && "No typedef found");

  return result;
}

void UseUsingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<TypedefDecl>("typedef");
  if (MatchedDecl->getLocation().isInvalid())
    return;

  auto &Context = *Result.Context;
  auto &SM = *Result.SourceManager;

  auto Diag =
      diag(MatchedDecl->getLocStart(), "use 'using' instead of 'typedef'");
  if (MatchedDecl->getLocStart().isMacroID()) {
    return;
  }
  SourceRange RemovalRange;
  if (CheckRemoval(SM, MatchedDecl->getLocStart(), MatchedDecl->getLocEnd(),
                   Context, RemovalRange)) {
    Diag << FixItHint::CreateReplacement(
        MatchedDecl->getSourceRange(),
        "using " + MatchedDecl->getNameAsString() + " = " +
            MatchedDecl->getUnderlyingType().getAsString(getLangOpts()));
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
