//===--- NumericLiteralCaseFixer.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements NumericLiteralCaseFixer that standardizes character
/// case within numeric literals.
///
//===----------------------------------------------------------------------===//

#include "NumericLiteralCaseFixer.h"
#include "NumericLiteralInfo.h"

#include "llvm/ADT/StringExtras.h"

#include <algorithm>

namespace clang {
namespace format {

static bool isNumericLiteralCaseFixerNeeded(const FormatStyle &Style) {
  // Check if language is supported.
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

  // Check if style options are set.
  const auto &Option = Style.NumericLiteralCase;
  const auto Leave = FormatStyle::NLCS_Leave;
  return Option.Prefix != Leave || Option.HexDigit != Leave ||
         Option.ExponentLetter != Leave || Option.Suffix != Leave;
}

static std::string
transformComponent(StringRef Component,
                   FormatStyle::NumericLiteralComponentStyle ConfigValue) {
  switch (ConfigValue) {
  case FormatStyle::NLCS_Upper:
    return Component.upper();
  case FormatStyle::NLCS_Lower:
    return Component.lower();
  default:
    // Covers FormatStyle::NLCS_Leave.
    return Component.str();
  }
}

/// Test if Suffix matches a C++ literal reserved by the library.
/// Matches against all suffixes reserved in the C++23 standard.
static bool matchesReservedSuffix(StringRef Suffix) {
  static constexpr std::array<StringRef, 11> SortedReservedSuffixes = {
      "d", "h", "i", "if", "il", "min", "ms", "ns", "s", "us", "y",
  };

  // This can be static_assert when we have access to constexpr is_sorted in
  // C++ 20.
  assert(llvm::is_sorted(SortedReservedSuffixes) &&
         "Must be sorted as precondition for lower_bound().");

  auto entry = llvm::lower_bound(SortedReservedSuffixes, Suffix);
  if (entry == SortedReservedSuffixes.cend())
    return false;
  return *entry == Suffix;
}

static std::string format(StringRef NumericLiteral, const FormatStyle &Style) {
  const char Separator = Style.isCpp() ? '\'' : '_';
  const NumericLiteralInfo Info(NumericLiteral, Separator);
  const bool HasBaseLetter = Info.BaseLetterPos != StringRef::npos;
  const bool HasExponent = Info.ExponentLetterPos != StringRef::npos;
  const bool HasSuffix = Info.SuffixPos != StringRef::npos;

  std::string Formatted;

  if (HasBaseLetter) {
    Formatted +=
        transformComponent(NumericLiteral.take_front(1 + Info.BaseLetterPos),
                           Style.NumericLiteralCase.Prefix);
  }
  // Reformat this slice as HexDigit whether or not the digit has hexadecimal
  // characters because binary/decimal/octal digits are unchanged.
  Formatted += transformComponent(
      NumericLiteral.slice(HasBaseLetter ? 1 + Info.BaseLetterPos : 0,
                           HasExponent ? Info.ExponentLetterPos
                           : HasSuffix ? Info.SuffixPos
                                       : NumericLiteral.size()),
      Style.NumericLiteralCase.HexDigit);

  if (HasExponent) {
    Formatted += transformComponent(
        NumericLiteral.slice(Info.ExponentLetterPos,
                             HasSuffix ? Info.SuffixPos
                                       : NumericLiteral.size()),
        Style.NumericLiteralCase.ExponentLetter);
  }

  if (HasSuffix) {
    StringRef Suffix = NumericLiteral.drop_front(Info.SuffixPos);
    if (matchesReservedSuffix(Suffix) || Suffix.front() == '_') {
      // In C++, it is idiomatic, but NOT standardized to define user-defined
      // literals with a leading '_'. Omit user defined literals and standard
      // reserved suffixes from transformation.
      Formatted += Suffix.str();
    } else {
      Formatted += transformComponent(Suffix, Style.NumericLiteralCase.Suffix);
    }
  }

  return Formatted;
}

std::pair<tooling::Replacements, unsigned>
NumericLiteralCaseFixer::process(const Environment &Env,
                                 const FormatStyle &Style) {
  if (!isNumericLiteralCaseFixerNeeded(Style))
    return {};

  const auto &SourceMgr = Env.getSourceManager();
  AffectedRangeManager AffectedRangeMgr(SourceMgr, Env.getCharRanges());

  const auto ID = Env.getFileID();
  const auto LangOpts = getFormattingLangOpts(Style);
  Lexer Lex(ID, SourceMgr.getBufferOrFake(ID), SourceMgr, LangOpts);
  Lex.SetCommentRetentionState(true);

  Token Tok;
  tooling::Replacements Result;

  for (bool Skip = false; !Lex.LexFromRawLexer(Tok);) {
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

    const auto Formatted = format(Text, Style);
    if (Formatted != Text) {
      cantFail(Result.add(
          tooling::Replacement(SourceMgr, Location, Length, Formatted)));
    }
  }

  return {Result, 0};
}

} // namespace format
} // namespace clang
