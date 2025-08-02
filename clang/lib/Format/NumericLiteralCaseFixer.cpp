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

#include "llvm/ADT/StringExtras.h"

#include <algorithm>

namespace clang {
namespace format {

using CharTransformFn = char (*)(char C);
namespace {

/// @brief Collection of std::transform predicates for each part of a numeric
/// literal
struct FormatParameters {
  FormatParameters(FormatStyle::LanguageKind Language,
                   const FormatStyle::NumericLiteralCaseStyle &CaseStyle);

  CharTransformFn Prefix;
  CharTransformFn HexDigit;
  CharTransformFn FloatExponentSeparator;
  CharTransformFn Suffix;

  char Separator;
};

/// @brief Parse a single numeric constant from text into ranges that are
/// appropriate for applying NumericLiteralCaseStyle rules.
class QuickNumericalConstantParser {
public:
  QuickNumericalConstantParser(const StringRef &IntegerLiteral,
                               const FormatParameters &Transforms);

  /// @brief Reformats the numeric constant if needed.
  /// Calling this method invalidates the object's state.
  /// @return std::nullopt if no reformatting is required. std::optional<>
  /// containing the reformatted string otherwise.
  std::optional<std::string> formatIfNeeded() &&;

private:
  const StringRef &IntegerLiteral;
  const FormatParameters &Transforms;

  std::string Formatted;

  std::string::iterator PrefixBegin;
  std::string::iterator PrefixEnd;
  std::string::iterator HexDigitBegin;
  std::string::iterator HexDigitEnd;
  std::string::iterator FloatExponentSeparatorBegin;
  std::string::iterator FloatExponentSeparatorEnd;
  std::string::iterator SuffixBegin;
  std::string::iterator SuffixEnd;

  void parse();
  void applyFormatting();
};

} // namespace

static char noOpTransform(char C) { return C; }

static CharTransformFn getTransform(int8_t config_value) {
  switch (config_value) {
  case FormatStyle::NLCS_Always:
    return llvm::toUpper;
  case FormatStyle::NLCS_Never:
    return llvm::toLower;
  case FormatStyle::NLCS_Leave:
  default:
    return noOpTransform;
  }
}

/// @brief Test if Suffix matches a C++ literal reserved by the library.
/// Matches against all suffixes reserved in the C++23 standard
static bool matchesReservedSuffix(StringRef Suffix) {
  static const std::array<StringRef, 11> SortedReservedSuffixes = {
      "d", "h", "i", "if", "il", "min", "ms", "ns", "s", "us", "y"};

  auto entry = std::lower_bound(SortedReservedSuffixes.cbegin(),
                                SortedReservedSuffixes.cend(), Suffix);
  if (entry == SortedReservedSuffixes.cend())
    return false;
  return *entry == Suffix;
}

FormatParameters::FormatParameters(
    FormatStyle::LanguageKind Language,
    const FormatStyle::NumericLiteralCaseStyle &CaseStyle)
    : Prefix(getTransform(CaseStyle.UpperCasePrefix)),
      HexDigit(getTransform(CaseStyle.UpperCaseHexDigit)),
      FloatExponentSeparator(
          getTransform(CaseStyle.UpperCaseFloatExponentSeparator)),
      Suffix(getTransform(CaseStyle.UpperCaseSuffix)) {
  switch (Language) {
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
}

QuickNumericalConstantParser::QuickNumericalConstantParser(
    const StringRef &IntegerLiteral, const FormatParameters &Transforms)
    : IntegerLiteral(IntegerLiteral), Transforms(Transforms),
      Formatted(IntegerLiteral), PrefixBegin(Formatted.begin()),
      PrefixEnd(Formatted.begin()), HexDigitBegin(Formatted.begin()),
      HexDigitEnd(Formatted.begin()),
      FloatExponentSeparatorBegin(Formatted.begin()),
      FloatExponentSeparatorEnd(Formatted.begin()),
      SuffixBegin(Formatted.begin()), SuffixEnd(Formatted.begin()) {}

void QuickNumericalConstantParser::parse() {
  auto Cur = Formatted.begin();
  const auto End = Formatted.end();

  bool IsHex = false;
  bool IsFloat = false;

  // Find the range that contains the prefix.
  PrefixBegin = Cur;
  if (Cur != End && *Cur == '0') {
    ++Cur;
    if (Cur != End) {
      const char C = *Cur;
      switch (C) {
      case 'x':
      case 'X':
        IsHex = true;
        ++Cur;
        break;
      case 'b':
      case 'B':
        ++Cur;
        break;
      case 'o':
      case 'O':
        // Javascript uses 0o as octal prefix.
        ++Cur;
        break;
      default:
        break;
      }
    }
  }
  PrefixEnd = Cur;

  // Find the range that contains hex digits.
  HexDigitBegin = Cur;
  if (IsHex) {
    Cur = std::find_if_not(Cur, End, [this, &IsFloat](char C) {
      if (C == '.') {
        IsFloat = true;
        return true;
      }
      return C == Transforms.Separator || llvm::isHexDigit(C);
    });
  }
  HexDigitEnd = Cur;

  // Find the range that contains a floating point exponent separator.
  // Hex digits have already been scanned through the decimal point.
  // Decimal/octal/binary literals must fast forward through the decimal first.
  if (!IsHex) {
    Cur = std::find_if_not(Cur, End, [this, &IsFloat](char C) {
      if (C == '.') {
        IsFloat = true;
        return true;
      }
      return C == Transforms.Separator || llvm::isDigit(C);
    });
  }
  // The next character of a floating point literal will either be the
  // separator, or the start of a suffix.
  FloatExponentSeparatorBegin = Cur;
  if (IsFloat) {
    const char LSep = IsHex ? 'p' : 'e';
    const char USep = IsHex ? 'P' : 'E';
    Cur = std::find_if_not(
        Cur, End, [LSep, USep](char C) { return C == LSep || C == USep; });
  }
  FloatExponentSeparatorEnd = Cur;

  // Fast forward through the exponent part of a floating point literal.
  if (!IsFloat) {
  } else if (FloatExponentSeparatorBegin == FloatExponentSeparatorEnd) {
  } else {
    Cur = std::find_if_not(Cur, End, [](char C) {
      return llvm::isDigit(C) || C == '+' || C == '-';
    });
  }

  // Find the range containing a suffix if any.
  SuffixBegin = Cur;
  size_t const SuffixLen = End - Cur;
  StringRef suffix(&(*SuffixBegin), SuffixLen);
  if (!matchesReservedSuffix(suffix)) {
    Cur = std::find_if_not(Cur, End, [](char C) {
      // In C++, it is idiomatic, but NOT standard to define user-defined
      // literals with a leading '_'. Omit user defined literals from
      // transformation.
      return C != '_';
    });
  }
  SuffixEnd = Cur;
}

void QuickNumericalConstantParser::applyFormatting() {

  auto Start = Formatted.cbegin();
  auto End = Formatted.cend();

  assert(Start <= PrefixBegin && End >= PrefixBegin &&
         "PrefixBegin is out of bounds");
  assert(Start <= PrefixEnd && End >= PrefixEnd &&
         "PrefixEnd is out of bounds");
  assert(Start <= HexDigitBegin && End >= HexDigitBegin &&
         "HexDigitBegin is out of bounds");
  assert(Start <= HexDigitEnd && End >= HexDigitEnd &&
         "HexDigitEnd is out of bounds");
  assert(Start <= FloatExponentSeparatorBegin &&
         End >= FloatExponentSeparatorBegin &&
         "FloatExponentSeparatorBegin is out of bounds");
  assert(Start <= FloatExponentSeparatorEnd &&
         End >= FloatExponentSeparatorEnd &&
         "FloatExponentSeparatorEnd is out of bounds");
  assert(Start <= SuffixBegin && End >= SuffixBegin &&
         "SuffixBegin is out of bounds");
  assert(Start <= SuffixEnd && End >= SuffixEnd &&
         "SuffixEnd is out of bounds");

  std::transform(PrefixBegin, PrefixEnd, PrefixBegin, Transforms.Prefix);
  std::transform(HexDigitBegin, HexDigitEnd, HexDigitBegin,
                 Transforms.HexDigit);
  std::transform(FloatExponentSeparatorBegin, FloatExponentSeparatorEnd,
                 FloatExponentSeparatorBegin,
                 Transforms.FloatExponentSeparator);
  std::transform(SuffixBegin, SuffixEnd, SuffixBegin, Transforms.Suffix);
}

std::optional<std::string> QuickNumericalConstantParser::formatIfNeeded() && {
  parse();
  applyFormatting();

  return (Formatted == IntegerLiteral)
             ? std::nullopt
             : std::make_optional<std::string>(std::move(Formatted));
}

std::pair<tooling::Replacements, unsigned>
NumericLiteralCaseFixer::process(const Environment &Env,
                                 const FormatStyle &Style) {

  const auto &CaseStyle = Style.NumericLiteralCase;
  const FormatParameters Transforms{Style.Language, CaseStyle};

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

    const auto Formatted =
        QuickNumericalConstantParser(Text, Transforms).formatIfNeeded();
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
