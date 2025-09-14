//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConcisePreprocessorDirectivesCheck.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

#include <array>

namespace clang::tidy::readability {

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
            const std::array<llvm::StringLiteral, 2> &Replacements) {
    // Lexer requires its input range to be null-terminated.
    SmallString<128> Condition =
        Lexer::getSourceText(CharSourceRange::getTokenRange(ConditionRange),
                             PP.getSourceManager(), PP.getLangOpts());
    Condition.push_back('\0');
    Lexer Lex(DirectiveLoc, PP.getLangOpts(), Condition.data(),
              Condition.data(), Condition.data() + Condition.size() - 1);
    Token Tok;
    bool Inverted = false; // The inverted form of #*def is #*ndef.
    std::size_t ParensNestingDepth = 0;
    for (;;) {
      if (Lex.LexFromRawLexer(Tok))
        return;

      if (Tok.is(tok::TokenKind::exclaim) ||
          (PP.getLangOpts().CPlusPlus &&
           Tok.is(tok::TokenKind::raw_identifier) &&
           Tok.getRawIdentifier() == "not"))
        Inverted = !Inverted;
      else if (Tok.is(tok::TokenKind::l_paren))
        ++ParensNestingDepth;
      else
        break;
    }

    if (Tok.isNot(tok::TokenKind::raw_identifier) ||
        Tok.getRawIdentifier() != "defined")
      return;

    bool NoMoreTokens = Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::TokenKind::l_paren)) {
      if (NoMoreTokens)
        return;
      ++ParensNestingDepth;
      NoMoreTokens = Lex.LexFromRawLexer(Tok);
    }

    if (Tok.isNot(tok::TokenKind::raw_identifier))
      return;
    const StringRef Macro = Tok.getRawIdentifier();

    while (!NoMoreTokens) {
      NoMoreTokens = Lex.LexFromRawLexer(Tok);
      if (Tok.isNot(tok::TokenKind::r_paren))
        return;
      --ParensNestingDepth;
    }

    if (ParensNestingDepth != 0)
      return;

    Check.diag(
        DirectiveLoc,
        "preprocessor condition can be written more concisely using '#%0'")
        << FixItHint::CreateReplacement(DirectiveLoc, Replacements[Inverted])
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
