//===--- RawStringLiteralCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RawStringLiteralCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {

bool containsEscapes(StringRef HayStack, StringRef Escapes) {
  size_t BackSlash = HayStack.find('\\');
  if (BackSlash == StringRef::npos)
    return false;

  while (BackSlash != StringRef::npos) {
    if (Escapes.find(HayStack[BackSlash + 1]) == StringRef::npos)
      return false;
    BackSlash = HayStack.find('\\', BackSlash + 2);
  }

  return true;
}

bool isRawStringLiteral(StringRef Text) {
  // Already a raw string literal if R comes before ".
  const size_t QuotePos = Text.find('"');
  assert(QuotePos != StringRef::npos);
  return (QuotePos > 0) && (Text[QuotePos - 1] == 'R');
}

bool containsEscapedCharacters(const MatchFinder::MatchResult &Result,
                               const StringLiteral *Literal,
                               const CharsBitSet &DisallowedChars) {
  // FIXME: Handle L"", u8"", u"" and U"" literals.
  if (!Literal->isAscii())
    return false;

  for (const unsigned char C : Literal->getBytes())
    if (DisallowedChars.test(C))
      return false;

  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Literal->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  StringRef Text = Lexer::getSourceText(CharRange, *Result.SourceManager,
                                        Result.Context->getLangOpts());
  if (isRawStringLiteral(Text))
    return false;

  return containsEscapes(Text, R"('\"?x01)");
}

bool containsDelimiter(StringRef Bytes, const std::string &Delimiter) {
  return Bytes.find(Delimiter.empty()
                        ? std::string(R"lit()")lit")
                        : (")" + Delimiter + R"(")")) != StringRef::npos;
}

std::string asRawStringLiteral(const StringLiteral *Literal,
                               const std::string &DelimiterStem) {
  const StringRef Bytes = Literal->getBytes();
  std::string Delimiter;
  for (int I = 0; containsDelimiter(Bytes, Delimiter); ++I) {
    Delimiter = (I == 0) ? DelimiterStem : DelimiterStem + std::to_string(I);
  }

  if (Delimiter.empty())
    return (R"(R"()" + Bytes + R"lit()")lit").str();

  return (R"(R")" + Delimiter + "(" + Bytes + ")" + Delimiter + R"(")").str();
}

} // namespace

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
  for (unsigned int C = 0x80u; C <= 0xFFu; ++C)
    DisallowedChars.set(static_cast<unsigned char>(C));
}

void RawStringLiteralCheck::storeOptions(ClangTidyOptions::OptionMap &Options) {
  ClangTidyCheck::storeOptions(Options);
  this->Options.store(Options, "ReplaceShorterLiterals",
                      ReplaceShorterLiterals);
}

void RawStringLiteralCheck::registerMatchers(MatchFinder *Finder) {
  // Raw string literals require C++11 or later.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(
      stringLiteral(unless(hasParent(predefinedExpr()))).bind("lit"), this);
}

void RawStringLiteralCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("lit");
  if (Literal->getBeginLoc().isMacroID())
    return;

  if (containsEscapedCharacters(Result, Literal, DisallowedChars)) {
    std::string Replacement = asRawStringLiteral(Literal, DelimiterStem);
    if (ReplaceShorterLiterals ||
        Replacement.length() <=
            Lexer::MeasureTokenLength(Literal->getBeginLoc(),
                                      *Result.SourceManager, getLangOpts()))
      replaceWithRawStringLiteral(Result, Literal, Replacement);
  }
}

void RawStringLiteralCheck::replaceWithRawStringLiteral(
    const MatchFinder::MatchResult &Result, const StringLiteral *Literal,
    StringRef Replacement) {
  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Literal->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  diag(Literal->getBeginLoc(),
       "escaped string literal can be written as a raw string literal")
      << FixItHint::CreateReplacement(CharRange, Replacement);
}

} // namespace modernize
} // namespace tidy
} // namespace clang
