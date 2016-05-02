//===--- AvoidConstParamsInDecls.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidConstParamsInDecls.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {
namespace {

SourceRange getTypeRange(const ParmVarDecl &Param) {
  if (Param.getIdentifier() != nullptr)
    return SourceRange(Param.getLocStart(),
                       Param.getLocEnd().getLocWithOffset(-1));
  return Param.getSourceRange();
}

} // namespace

void AvoidConstParamsInDecls::registerMatchers(MatchFinder *Finder) {
  const auto ConstParamDecl =
      parmVarDecl(hasType(qualType(isConstQualified()))).bind("param");
  Finder->addMatcher(functionDecl(unless(isDefinition()),
                                  has(typeLoc(forEach(ConstParamDecl))))
                         .bind("func"),
                     this);
}

// Re-lex the tokens to get precise location of last 'const'
static Token ConstTok(CharSourceRange Range,
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
  Token ConstTok;
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

  auto Diag = diag(Param->getLocStart(),
                   "parameter %0 is const-qualified in the function "
                   "declaration; const-qualification of parameters only has an "
                   "effect in function definitions");
  if (Param->getName().empty()) {
    for (unsigned int i = 0; i < Func->getNumParams(); ++i) {
      if (Param == Func->getParamDecl(i)) {
        Diag << (i + 1);
        break;
      }
    }
  } else {
    Diag << Param;
  }

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(getTypeRange(*Param)),
      *Result.SourceManager, Result.Context->getLangOpts());

  if (!FileRange.isValid())
    return;

  Token Tok = ConstTok(FileRange, Result);
  Diag << FixItHint::CreateRemoval(
      CharSourceRange::getTokenRange(Tok.getLocation(), Tok.getLocation()));
}

} // namespace readability
} // namespace tidy
} // namespace clang
