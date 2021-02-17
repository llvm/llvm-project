//===--- AvoidConstParamsInDecls.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidConstParamsInDecls.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/Optional.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {
namespace {

SourceRange getTypeRange(const ParmVarDecl &Param) {
  if (Param.getIdentifier() != nullptr)
    return SourceRange(Param.getBeginLoc(),
                       Param.getEndLoc().getLocWithOffset(-1));
  return Param.getSourceRange();
}

} // namespace

void AvoidConstParamsInDecls::registerMatchers(MatchFinder *Finder) {
  const auto ConstParamDecl =
      parmVarDecl(hasType(qualType(isConstQualified()))).bind("param");
  Finder->addMatcher(
      functionDecl(unless(isDefinition()),
                   has(typeLoc(forEach(ConstParamDecl))))
          .bind("func"),
      this);
}

// Re-lex the tokens to get precise location of last 'const'
static llvm::Optional<Token> constTok(CharSourceRange Range,
                                      const MatchFinder::MatchResult &Result) {
  const SourceManager &Sources = *Result.SourceManager;
  std::pair<FileID, unsigned> LocInfo =
      Sources.getDecomposedLoc(Range.getBegin());
  StringRef File = Sources.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer RawLexer(Sources.getLocForStartOfFile(LocInfo.first),
                 Result.Context->getLangOpts(), File.begin(), TokenBegin,
                 File.end());
  Token Tok;
  llvm::Optional<Token> ConstTok;
  while (!RawLexer.LexFromRawLexer(Tok)) {
    if (Sources.isBeforeInTranslationUnit(Range.getEnd(), Tok.getLocation()))
      break;
    if (Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = Result.Context->Idents.get(StringRef(
          Sources.getCharacterData(Tok.getLocation()), Tok.getLength()));
      Tok.setIdentifierInfo(&Info);
      Tok.setKind(Info.getTokenID());
    }
    if (Tok.is(tok::kw_const))
      ConstTok = Tok;
  }
  return ConstTok;
}

void AvoidConstParamsInDecls::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  if (!Param->getType().isLocalConstQualified())
    return;

  auto Diag = diag(Param->getBeginLoc(),
                   "parameter %0 is const-qualified in the function "
                   "declaration; const-qualification of parameters only has an "
                   "effect in function definitions");
  if (Param->getName().empty()) {
    for (unsigned int I = 0; I < Func->getNumParams(); ++I) {
      if (Param == Func->getParamDecl(I)) {
        Diag << (I + 1);
        break;
      }
    }
  } else {
    Diag << Param;
  }

  if (Param->getBeginLoc().isMacroID() != Param->getEndLoc().isMacroID()) {
    // Do not offer a suggestion if the part of the variable declaration comes
    // from a macro.
    return;
  }

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(getTypeRange(*Param)),
      *Result.SourceManager, getLangOpts());

  if (!FileRange.isValid())
    return;

  auto Tok = constTok(FileRange, Result);
  if (!Tok)
    return;
  Diag << FixItHint::CreateRemoval(
      CharSourceRange::getTokenRange(Tok->getLocation(), Tok->getLocation()));
}

} // namespace readability
} // namespace tidy
} // namespace clang
