//===--- NumericLiteralCaseFixer.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements NumericLiteralCaseFixer that standardizes character
/// case within numeric literal constants.
///
//===----------------------------------------------------------------------===//

#include "NumericLiteralCaseFixer.h"
#include "NumericLiteralInfo.h"

#include "llvm/ADT/StringExtras.h"

#include <algorithm>

namespace clang {
namespace format {

static std::string
transformComponent(StringRef Component,
                   FormatStyle::NumericLiteralComponentStyle ConfigValue) {
  switch (ConfigValue) {
  case FormatStyle::NLCS_Upper:
    return Component.upper();
  case FormatStyle::NLCS_Lower:
    return Component.lower();
  case FormatStyle::NLCS_Leave:
  default:
    return Component.str();
  }
}

/// @brief Test if Suffix matches a C++ literal reserved by the library.
/// Matches against all suffixes reserved in the C++23 standard
static bool matchesReservedSuffix(StringRef Suffix) {
  static constexpr std::array<StringRef, 11> SortedReservedSuffixes = {
      "d", "h", "i", "if", "il", "min", "ms", "ns", "s", "us", "y"};

  // This can be static_assert when we have access to constexpr is_sorted in
  // C++ 20.
  assert(llvm::is_sorted(SortedReservedSuffixes) &&
         "Must be sorted as precondition for lower_bound().");

  auto entry = llvm::lower_bound(SortedReservedSuffixes, Suffix);
  if (entry == SortedReservedSuffixes.cend())
    return false;
  return *entry == Suffix;
}

static std::optional<std::string> formatIfNeeded(StringRef IntegerLiteral,
                                                 const FormatStyle &Style) {
  char Separator;
  switch (Style.Language) {
  case FormatStyle::LK_CSharp:
  case FormatStyle::LK_Java:
  case FormatStyle::LK_JavaScript:
    Separator = '_';
    break;
  case FormatStyle::LK_C:
  case FormatStyle::LK_Cpp:
  case FormatStyle::LK_ObjC:
  default:
    Separator = '\'';
  }
  const NumericLiteralInfo N{IntegerLiteral, Separator};

  std::string Formatted{""};

  if (N.BaseLetterPos != llvm::StringRef::npos) {
    Formatted +=
        transformComponent(IntegerLiteral.take_front(1 + N.BaseLetterPos),
                           Style.NumericLiteralCase.Prefix);
  }
  // reformat this slice as HexDigit whether or not the digit has hexadecimal
  // characters because binary/decimal/octal digits are unchanged
  Formatted += transformComponent(
      IntegerLiteral.slice(
          N.BaseLetterPos == llvm::StringRef::npos ? 0 : 1 + N.BaseLetterPos,
          N.ExponentLetterPos == llvm::StringRef::npos
              ? N.SuffixPos == llvm::StringRef::npos ? IntegerLiteral.size()
                                                     : N.SuffixPos
              : N.ExponentLetterPos),
      Style.NumericLiteralCase.HexDigit);

  if (N.ExponentLetterPos != llvm::StringRef::npos) {
    Formatted += transformComponent(
        IntegerLiteral.slice(N.ExponentLetterPos,
                             N.SuffixPos == llvm::StringRef::npos
                                 ? IntegerLiteral.size()
                                 : N.SuffixPos),
        Style.NumericLiteralCase.ExponentLetter);
  }

  if (N.SuffixPos != llvm::StringRef::npos) {
    StringRef Suffix = IntegerLiteral.drop_front(N.SuffixPos);
    if (matchesReservedSuffix(Suffix) || Suffix.front() == '_') {
      // In C++, it is idiomatic, but NOT standardized to define user-defined
      // literals with a leading '_'. Omit user defined literals and standard
      // reserved suffixes from transformation.
      Formatted += Suffix.str();
    } else {
      Formatted += transformComponent(Suffix, Style.NumericLiteralCase.Suffix);
    }
  }

  if (Formatted == IntegerLiteral)
    return std::nullopt;
  else
    return Formatted;
}

std::pair<tooling::Replacements, unsigned>
NumericLiteralCaseFixer::process(const Environment &Env,
                                 const FormatStyle &Style) {

  const auto &SourceMgr = Env.getSourceManager();
  AffectedRangeManager AffectedRangeMgr(SourceMgr, Env.getCharRanges());

  const auto ID = Env.getFileID();
  const auto LangOpts = getFormattingLangOpts(Style);
  Lexer Lex(ID, SourceMgr.getBufferOrFake(ID), SourceMgr, LangOpts);
  Lex.SetCommentRetentionState(true);

  Token Tok;
  tooling::Replacements Result;
  bool Skip = false;

  while (!Lex.LexFromRawLexer(Tok)) {
    // Skip tokens that are too small to contain a formattable literal.
    // Size=2 is the smallest possible literal that could contain formattable
    // components, for example "1u".
    auto Length = Tok.getLength();
    if (Length < 2)
      continue;

    // Service clang-format off/on comments.
    auto Location = Tok.getLocation();
    auto Text = StringRef(SourceMgr.getCharacterData(Location), Length);
    if (Tok.is(tok::comment)) {
      if (isClangFormatOff(Text))
        Skip = true;
      else if (isClangFormatOn(Text))
        Skip = false;
      continue;
    }

    if (Skip || Tok.isNot(tok::numeric_constant) ||
        !AffectedRangeMgr.affectsCharSourceRange(
            CharSourceRange::getCharRange(Location, Tok.getEndLoc()))) {
      continue;
    }

    const auto Formatted = formatIfNeeded(Text, Style);
    if (Formatted) {
      assert(*Formatted != Text && "QuickNumericalConstantParser returned an "
                                   "unchanged value instead of nullopt");
      cantFail(Result.add(
          tooling::Replacement(SourceMgr, Location, Length, *Formatted)));
    }
  }

  return {Result, 0};
}

bool NumericLiteralCaseFixer::isActive(const FormatStyle &Style) {

  switch (Style.Language) {
  case FormatStyle::LK_C:
  case FormatStyle::LK_Cpp:
  case FormatStyle::LK_ObjC:
  case FormatStyle::LK_CSharp:
  case FormatStyle::LK_Java:
  case FormatStyle::LK_JavaScript:
    break;
  default:
    return false;
  }

  const FormatStyle::NumericLiteralCaseStyle LeaveAllCasesUntouched{};

  return Style.NumericLiteralCase != LeaveAllCasesUntouched;
}
} // namespace format
} // namespace clang
