//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawStringLiteralCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static bool containsEscapes(StringRef HayStack, StringRef Escapes) {
  size_t BackSlash = HayStack.find('\\');
  if (BackSlash == StringRef::npos)
    return false;

  while (BackSlash != StringRef::npos) {
    if (!Escapes.contains(HayStack[BackSlash + 1]))
      return false;
    BackSlash = HayStack.find('\\', BackSlash + 2);
  }

  return true;
}

static bool isRawStringLiteral(StringRef Text) {
  // Already a raw string literal if R comes before ".
  const size_t QuotePos = Text.find('"');
  assert(QuotePos != StringRef::npos);
  return (QuotePos > 0) && (Text[QuotePos - 1] == 'R');
}

static bool containsEscapedCharacters(const MatchFinder::MatchResult &Result,
                                      const StringLiteral *Literal,
                                      const CharsBitSet &DisallowedChars) {
  // FIXME: Handle L"", u8"", u"" and U"" literals.
  if (!Literal->isOrdinary())
    return false;

  for (const unsigned char C : Literal->getBytes())
    if (DisallowedChars.test(C))
      return false;

  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Literal->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  StringRef Text = Lexer::getSourceText(CharRange, *Result.SourceManager,
                                        Result.Context->getLangOpts());
  if (Text.empty() || isRawStringLiteral(Text))
    return false;

  return containsEscapes(Text, R"('\"?x01)");
}

static bool containsDelimiter(StringRef Bytes, const std::string &Delimiter) {
  return Bytes.find(Delimiter.empty()
                        ? std::string(R"lit()")lit")
                        : (")" + Delimiter + R"(")")) != StringRef::npos;
}

RawStringLiteralCheck::RawStringLiteralCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      DelimiterStem(Options.get("DelimiterStem", "lit")),
      ReplaceShorterLiterals(Options.get("ReplaceShorterLiterals", false)) {
  // Non-printing characters are disallowed:
  // \007 = \a bell
  // \010 = \b backspace
  // \011 = \t horizontal tab
  // \012 = \n new line
  // \013 = \v vertical tab
  // \014 = \f form feed
  // \015 = \r carriage return
  // \177 = delete
  for (const unsigned char C : StringRef("\000\001\002\003\004\005\006\a"
                                         "\b\t\n\v\f\r\016\017"
                                         "\020\021\022\023\024\025\026\027"
                                         "\030\031\032\033\034\035\036\037"
                                         "\177",
                                         33))
    DisallowedChars.set(C);

  // Non-ASCII are disallowed too.
  for (unsigned int C = 0x80U; C <= 0xFFU; ++C)
    DisallowedChars.set(static_cast<unsigned char>(C));
}

void RawStringLiteralCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DelimiterStem", DelimiterStem);
  Options.store(Opts, "ReplaceShorterLiterals", ReplaceShorterLiterals);
}

void RawStringLiteralCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      stringLiteral(unless(hasParent(predefinedExpr()))).bind("lit"), this);
}

static std::optional<StringRef>
createUserDefinedSuffix(const StringLiteral *Literal, const SourceManager &SM,
                        const LangOptions &LangOpts) {
  const CharSourceRange TokenRange =
      CharSourceRange::getTokenRange(Literal->getSourceRange());
  Token T;
  if (Lexer::getRawToken(Literal->getBeginLoc(), T, SM, LangOpts))
    return std::nullopt;
  const CharSourceRange CharRange =
      Lexer::makeFileCharRange(TokenRange, SM, LangOpts);
  if (T.hasUDSuffix()) {
    StringRef Text = Lexer::getSourceText(CharRange, SM, LangOpts);
    const size_t UDSuffixPos = Text.find_last_of('"');
    if (UDSuffixPos == StringRef::npos)
      return std::nullopt;
    return Text.slice(UDSuffixPos + 1, Text.size());
  }
  return std::nullopt;
}

static std::string createRawStringLiteral(const StringLiteral *Literal,
                                          const std::string &DelimiterStem,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  const StringRef Bytes = Literal->getBytes();
  std::string Delimiter;
  for (int I = 0; containsDelimiter(Bytes, Delimiter); ++I) {
    Delimiter = (I == 0) ? DelimiterStem : DelimiterStem + std::to_string(I);
  }

  std::optional<StringRef> UserDefinedSuffix =
      createUserDefinedSuffix(Literal, SM, LangOpts);

  if (Delimiter.empty())
    return (R"(R"()" + Bytes + R"lit()")lit" + UserDefinedSuffix.value_or(""))
        .str();

  return (R"(R")" + Delimiter + "(" + Bytes + ")" + Delimiter + R"(")" +
          UserDefinedSuffix.value_or(""))
      .str();
}

static bool compareStringLength(StringRef Replacement,
                                const StringLiteral *Literal,
                                const SourceManager &SM,
                                const LangOptions &LangOpts) {
  return Replacement.size() <=
         Lexer::MeasureTokenLength(Literal->getBeginLoc(), SM, LangOpts);
}

void RawStringLiteralCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("lit");
  if (Literal->getBeginLoc().isMacroID())
    return;
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();
  if (containsEscapedCharacters(Result, Literal, DisallowedChars)) {
    const std::string Replacement =
        createRawStringLiteral(Literal, DelimiterStem, SM, LangOpts);
    if (ReplaceShorterLiterals ||
        compareStringLength(Replacement, Literal, SM, LangOpts)) {
      diag(Literal->getBeginLoc(),
           "escaped string literal can be written as a raw string literal")
          << FixItHint::CreateReplacement(Literal->getSourceRange(),
                                          Replacement);
    }
  }
}

} // namespace clang::tidy::modernize
