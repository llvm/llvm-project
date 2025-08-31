//===--- UseConcisePreprocessorDirectivesCheck.cpp - clang-tidy -----------===//
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
#include "llvm/ADT/STLForwardCompat.h"

#include <array>

namespace clang::tidy::readability {

namespace {

enum class DirectiveKind : bool { If, Elif };

struct Rewrite {
  SourceLocation DirectiveLoc;
  SourceRange ConditionRange;
  StringRef Macro;
  bool Negated;
  DirectiveKind Directive;
};

struct StackEntry {
  SmallVector<Rewrite, 2> Rewrites;
  bool ApplyingRewritesWouldBreakConsistency = false;
};

class UseConciseDirectivesPPCallbacks final : public PPCallbacks {
public:
  UseConciseDirectivesPPCallbacks(UseConcisePreprocessorDirectivesCheck &Check,
                                  const Preprocessor &PP,
                                  bool PreserveConsistency)
      : Check(Check), PP(PP), PreserveConsistency(PreserveConsistency) {}

  void Ifdef(SourceLocation, const Token &, const MacroDefinition &) override {
    Stack.emplace_back();
  }

  void Ifndef(SourceLocation, const Token &, const MacroDefinition &) override {
    Stack.emplace_back();
  }

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind) override {
    Stack.emplace_back();
    handleCondition(Loc, ConditionRange, DirectiveKind::If);
  }

  void Elif(SourceLocation Loc, SourceRange ConditionRange, ConditionValueKind,
            SourceLocation) override {
    handleCondition(Loc, ConditionRange, DirectiveKind::Elif);
  }

  void Endif(SourceLocation, SourceLocation) override {
    static constexpr std::array<std::array<StringRef, 2>, 2> Replacements = {{
        {"ifdef", "ifndef"},
        {"elifdef", "elifndef"},
    }};

    const auto &[Rewrites, ApplyingRewritesWouldBreakConsistency] =
        Stack.back();
    if (!(PreserveConsistency && ApplyingRewritesWouldBreakConsistency)) {
      for (const auto &[DirectiveLoc, ConditionRange, Macro, Negated,
                        Directive] : Rewrites) {
        const StringRef Replacement =
            Replacements[llvm::to_underlying(Directive)][Negated];
        Check.diag(
            DirectiveLoc,
            "preprocessor condition can be written more concisely using '#%0'")
            << Replacement
            << FixItHint::CreateReplacement(DirectiveLoc, Replacement)
            << FixItHint::CreateReplacement(ConditionRange, Macro);
      }
    }

    Stack.pop_back();
  }

private:
  struct RewriteInfo {
    StringRef Macro;
    bool Negated;
  };

  void handleCondition(SourceLocation DirectiveLoc, SourceRange ConditionRange,
                       DirectiveKind Directive) {
    if (Directive != DirectiveKind::Elif || PP.getLangOpts().C23 ||
        PP.getLangOpts().CPlusPlus23)
      if (const std::optional<RewriteInfo> Rewrite =
              tryRewrite(ConditionRange)) {
        Stack.back().Rewrites.push_back({DirectiveLoc, ConditionRange,
                                         Rewrite->Macro, Rewrite->Negated,
                                         Directive});
        return;
      }

    if (!Stack.back().ApplyingRewritesWouldBreakConsistency)
      Stack.back().ApplyingRewritesWouldBreakConsistency =
          conditionContainsDefinedOperator(ConditionRange);
  }

  std::optional<RewriteInfo> tryRewrite(SourceRange ConditionRange) {
    // Lexer requires its input range to be null-terminated.
    const StringRef SourceText =
        Lexer::getSourceText(CharSourceRange::getTokenRange(ConditionRange),
                             PP.getSourceManager(), PP.getLangOpts());
    SmallString<128> Condition = SourceText;
    Condition.push_back('\0');
    Lexer Lex({}, PP.getLangOpts(), Condition.data(), Condition.data(),
              Condition.data() + Condition.size() - 1);
    Token Tok;
    bool Negated = false;
    std::size_t ParensNestingDepth = 0;
    for (;;) {
      if (Lex.LexFromRawLexer(Tok))
        return {};

      if (Tok.is(tok::TokenKind::exclaim) ||
          (PP.getLangOpts().CPlusPlus &&
           Tok.is(tok::TokenKind::raw_identifier) &&
           Tok.getRawIdentifier() == "not"))
        Negated = !Negated;
      else if (Tok.is(tok::TokenKind::l_paren))
        ++ParensNestingDepth;
      else
        break;
    }

    if (Tok.isNot(tok::TokenKind::raw_identifier) ||
        Tok.getRawIdentifier() != "defined")
      return {};

    bool NoMoreTokens = Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::TokenKind::l_paren)) {
      if (NoMoreTokens)
        return {};
      ++ParensNestingDepth;
      NoMoreTokens = Lex.LexFromRawLexer(Tok);
    }

    if (Tok.isNot(tok::TokenKind::raw_identifier))
      return {};

    // We need a stable StringRef into the ConditionRange, but because Lexer
    // forces us to work on a temporary null-terminated copy of the
    // ConditionRange, we need to do this ugly translation.
    const StringRef Macro = {
        SourceText.data() + (Tok.getRawIdentifier().data() - Condition.data()),
        Tok.getRawIdentifier().size()};

    while (!NoMoreTokens) {
      NoMoreTokens = Lex.LexFromRawLexer(Tok);
      if (Tok.isNot(tok::TokenKind::r_paren))
        return {};
      --ParensNestingDepth;
    }

    if (ParensNestingDepth != 0)
      return {};

    return {{Macro, Negated}};
  }

  bool conditionContainsDefinedOperator(SourceRange ConditionRange) {
    // Lexer requires its input range to be null-terminated.
    SmallString<128> Condition =
        Lexer::getSourceText(CharSourceRange::getTokenRange(ConditionRange),
                             PP.getSourceManager(), PP.getLangOpts());
    Condition.push_back('\0');
    Lexer Lex({}, PP.getLangOpts(), Condition.data(), Condition.data(),
              Condition.data() + Condition.size() - 1);

    for (Token Tok;;) {
      const bool NoMoreTokens = Lex.LexFromRawLexer(Tok);
      if (Tok.is(tok::TokenKind::raw_identifier) &&
          Tok.getRawIdentifier() == "defined")
        return true;
      if (NoMoreTokens)
        return false;
    }
  }

  UseConcisePreprocessorDirectivesCheck &Check;
  const Preprocessor &PP;
  const bool PreserveConsistency;
  SmallVector<StackEntry, 4> Stack;
};

} // namespace

UseConcisePreprocessorDirectivesCheck::UseConcisePreprocessorDirectivesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      PreserveConsistency(Options.get("PreserveConsistency", false)) {}

void UseConcisePreprocessorDirectivesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "PreserveConsistency", PreserveConsistency);
}

void UseConcisePreprocessorDirectivesCheck::registerPPCallbacks(
    const SourceManager &, Preprocessor *PP, Preprocessor *) {
  PP->addPPCallbacks(std::make_unique<UseConciseDirectivesPPCallbacks>(
      *this, *PP, PreserveConsistency));
}

} // namespace clang::tidy::readability
