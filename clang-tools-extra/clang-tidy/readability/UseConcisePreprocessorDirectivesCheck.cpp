//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConcisePreprocessorDirectivesCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

#include <array>

namespace clang::tidy::readability {

using utils::lexer::getTokenName;

namespace {

class IfPreprocessorCallbacks final : public PPCallbacks {
public:
  IfPreprocessorCallbacks(ClangTidyCheck &Check, const Preprocessor &PP)
      : Check(Check), PP(PP) {}

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind) override {
    impl(Loc, ConditionRange, {"ifdef", "ifndef"});
  }

  void Elif(SourceLocation Loc, SourceRange ConditionRange, ConditionValueKind,
            SourceLocation) override {
    if (PP.getLangOpts().C23 || PP.getLangOpts().CPlusPlus23)
      impl(Loc, ConditionRange, {"elifdef", "elifndef"});
  }

private:
  void impl(SourceLocation DirectiveLoc, SourceRange ConditionRange,
            const std::array<StringRef, 2> &Replacements) {
    const std::vector<Token> Tokens = utils::lexer::getRawTokens(
        CharSourceRange::getTokenRange(ConditionRange), PP.getSourceManager(),
        PP.getLangOpts());
    bool Inverted = false; // The inverted form of #*def is #*ndef.
    std::size_t ParensNestingDepth = 0;
    std::size_t Index = 0;
    while (Index < Tokens.size()) {
      const Token &Tok = Tokens[Index];
      if (Tok.is(tok::TokenKind::exclaim) ||
          (PP.getLangOpts().CPlusPlus &&
           Tok.is(tok::TokenKind::raw_identifier) &&
           getTokenName(Tok) == "not")) {
        Inverted = !Inverted;
        ++Index;
      } else if (Tok.is(tok::TokenKind::l_paren)) {
        ++ParensNestingDepth;
        ++Index;
      } else {
        break;
      }
    }

    if (Index >= Tokens.size() ||
        Tokens[Index].isNot(tok::TokenKind::raw_identifier) ||
        getTokenName(Tokens[Index]) != "defined")
      return;
    ++Index;

    if (Index < Tokens.size() && Tokens[Index].is(tok::TokenKind::l_paren)) {
      ++ParensNestingDepth;
      ++Index;
    }

    if (Index >= Tokens.size() ||
        Tokens[Index].isNot(tok::TokenKind::raw_identifier))
      return;
    const StringRef Macro = getTokenName(Tokens[Index++]);

    while (Index < Tokens.size()) {
      if (Tokens[Index++].isNot(tok::TokenKind::r_paren) ||
          ParensNestingDepth == 0)
        return;
      --ParensNestingDepth;
    }

    if (ParensNestingDepth != 0)
      return;

    Check.diag(
        DirectiveLoc,
        "preprocessor condition can be written more concisely using '#%0'")
        << FixItHint::CreateReplacement(
               CharSourceRange::getCharRange(DirectiveLoc,
                                             ConditionRange.getBegin()),
               (Replacements[Inverted].str() + " "))
        << FixItHint::CreateReplacement(ConditionRange, Macro)
        << Replacements[Inverted];
  }

  ClangTidyCheck &Check;
  const Preprocessor &PP;
};

} // namespace

void UseConcisePreprocessorDirectivesCheck::registerPPCallbacks(
    const SourceManager &, Preprocessor *PP, Preprocessor *) {
  PP->addPPCallbacks(std::make_unique<IfPreprocessorCallbacks>(*this, *PP));
}

} // namespace clang::tidy::readability
