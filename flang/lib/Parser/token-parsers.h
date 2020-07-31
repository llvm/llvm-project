//===-- lib/Parser/token-parsers.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_TOKEN_PARSERS_H_
#define FORTRAN_PARSER_TOKEN_PARSERS_H_

// These parsers are driven by the parsers of the Fortran grammar to consume
// the prescanned character stream and recognize context-sensitive tokens.

#include "basic-parsers.h"
#include "type-parsers.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/char-set.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/instrumented-parser.h"
#include "flang/Parser/provenance.h"
#include <cctype>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <optional>
#include <string>

namespace Fortran::parser {

// "xyz"_ch matches one instance of the characters x, y, or z without skipping
// any spaces before or after.  The parser returns the location of the character
// on success.
class AnyOfChars {
public:
  using resultType = const char *;
  constexpr AnyOfChars(const AnyOfChars &) = default;
  constexpr AnyOfChars(SetOfChars set) : set_{set} {}
  std::optional<const char *> Parse(ParseState &state) const {
    if (std::optional<const char *> at{state.PeekAtNextChar()}) {
      if (set_.Has(**at)) {
        state.UncheckedAdvance();
        state.set_anyTokenMatched();
        return at;
      }
    }
    state.Say(MessageExpectedText{set_});
    return std::nullopt;
  }

private:
  const SetOfChars set_;
};

constexpr AnyOfChars operator""_ch(const char str[], std::size_t n) {
  return AnyOfChars{SetOfChars(str, n)};
}

constexpr auto letter{"abcdefghijklmnopqrstuvwxyz"_ch};
constexpr auto digit{"0123456789"_ch};

// Skips over optional spaces.  Always succeeds.
constexpr struct Space {
  using resultType = Success;
  constexpr Space() {}
  static std::optional<Success> Parse(ParseState &state) {
    while (std::optional<const char *> p{state.PeekAtNextChar()}) {
      if (**p != ' ') {
        break;
      }
      state.UncheckedAdvance();
    }
    return {Success{}};
  }
} space;

// Skips a space that in free form requires a warning if it precedes a
// character that could begin an identifier or keyword.  Always succeeds.
inline void MissingSpace(ParseState &state) {
  if (!state.inFixedForm()) {
    state.Nonstandard(
        LanguageFeature::OptionalFreeFormSpace, "missing space"_en_US);
  }
}

constexpr struct SpaceCheck {
  using resultType = Success;
  constexpr SpaceCheck() {}
  static std::optional<Success> Parse(ParseState &state) {
    if (std::optional<const char *> p{state.PeekAtNextChar()}) {
      char ch{**p};
      if (ch == ' ') {
        state.UncheckedAdvance();
        return space.Parse(state);
      }
      if (IsLegalInIdentifier(ch)) {
        MissingSpace(state);
      }
    }
    return {Success{}};
  }
} spaceCheck;

// Matches a token string.  Spaces in the token string denote where
// spaces may appear in the source; they can be made mandatory for
// some free form keyword sequences.  Missing mandatory spaces in free
// form elicit a warning; they are not necessary for recognition.
// Spaces before and after the token are also skipped.
//
// Token strings appear in the grammar as C++ user-defined literals
// like "BIND ( C )"_tok and "SYNC ALL"_sptok.  The _tok suffix is implied
// when a string literal appears before the sequencing operator >> or
// after the sequencing operator /.  The literal "..."_id parses a
// token that cannot be a prefix of a longer identifier.
template <bool MandatoryFreeFormSpace = false, bool MustBeComplete = false>
class TokenStringMatch {
public:
  using resultType = Success;
  constexpr TokenStringMatch(const TokenStringMatch &) = default;
  constexpr TokenStringMatch(const char *str, std::size_t n)
      : str_{str}, bytes_{n} {}
  explicit constexpr TokenStringMatch(const char *str) : str_{str} {}
  std::optional<Success> Parse(ParseState &state) const {
    space.Parse(state);
    const char *start{state.GetLocation()};
    const char *p{str_};
    std::optional<const char *> at; // initially empty
    for (std::size_t j{0}; j < bytes_ && *p != '\0'; ++j, ++p) {
      bool spaceSkipping{*p == ' '};
      if (spaceSkipping) {
        if (j + 1 == bytes_ || p[1] == ' ' || p[1] == '\0') {
          continue; // redundant; ignore
        }
      }
      if (!at) {
        at = nextCh.Parse(state);
        if (!at) {
          return std::nullopt;
        }
      }
      if (spaceSkipping) {
        if (**at == ' ') {
          at = nextCh.Parse(state);
          if (!at) {
            return std::nullopt;
          }
        } else if constexpr (MandatoryFreeFormSpace) {
          MissingSpace(state);
        }
        // 'at' remains full for next iteration
      } else if (**at == ToLowerCaseLetter(*p)) {
        at.reset();
      } else {
        state.Say(start, MessageExpectedText{str_, bytes_});
        return std::nullopt;
      }
    }
    if constexpr (MustBeComplete) {
      if (auto after{state.PeekAtNextChar()}) {
        if (IsLegalInIdentifier(**after)) {
          state.Say(start, MessageExpectedText{str_, bytes_});
          return std::nullopt;
        }
      }
    }
    state.set_anyTokenMatched();
    if (IsLegalInIdentifier(p[-1])) {
      return spaceCheck.Parse(state);
    } else {
      return space.Parse(state);
    }
  }

private:
  const char *const str_;
  const std::size_t bytes_{std::string::npos};
};

constexpr TokenStringMatch<> operator""_tok(const char str[], std::size_t n) {
  return {str, n};
}

constexpr TokenStringMatch<true> operator""_sptok(
    const char str[], std::size_t n) {
  return {str, n};
}

constexpr TokenStringMatch<false, true> operator""_id(
    const char str[], std::size_t n) {
  return {str, n};
}

template <class PA>
inline constexpr std::enable_if_t<std::is_class_v<PA>,
    SequenceParser<TokenStringMatch<>, PA>>
operator>>(const char *str, const PA &p) {
  return SequenceParser<TokenStringMatch<>, PA>{TokenStringMatch<>{str}, p};
}

template <class PA>
inline constexpr std::enable_if_t<std::is_class_v<PA>,
    FollowParser<PA, TokenStringMatch<>>>
operator/(const PA &p, const char *str) {
  return FollowParser<PA, TokenStringMatch<>>{p, TokenStringMatch<>{str}};
}

template <class PA> inline constexpr auto parenthesized(const PA &p) {
  return "(" >> p / ")";
}

template <class PA> inline constexpr auto bracketed(const PA &p) {
  return "[" >> p / "]";
}

// Quoted character literal constants.
struct CharLiteralChar {
  using resultType = std::pair<char, bool /* was escaped */>;
  static std::optional<resultType> Parse(ParseState &state) {
    auto at{state.GetLocation()};
    if (std::optional<const char *> cp{nextCh.Parse(state)}) {
      char ch{**cp};
      if (ch == '\n') {
        state.Say(CharBlock{at, state.GetLocation()},
            "Unclosed character constant"_err_en_US);
        return std::nullopt;
      }
      if (ch == '\\') {
        // Most escape sequences in character literals are processed later,
        // but we have to look for quotes here so that doubled quotes work.
        if (std::optional<const char *> next{state.PeekAtNextChar()}) {
          char escaped{**next};
          if (escaped == '\'' || escaped == '"' || escaped == '\\') {
            state.UncheckedAdvance();
            return std::make_pair(escaped, true);
          }
        }
      }
      return std::make_pair(ch, false);
    }
    return std::nullopt;
  }
};

template <char quote> struct CharLiteral {
  using resultType = std::string;
  static std::optional<std::string> Parse(ParseState &state) {
    std::string str;
    static constexpr auto nextch{attempt(CharLiteralChar{})};
    while (auto ch{nextch.Parse(state)}) {
      if (ch->second) {
        str += '\\';
      } else if (ch->first == quote) {
        static constexpr auto doubled{attempt(AnyOfChars{SetOfChars{quote}})};
        if (!doubled.Parse(state)) {
          return str;
        }
      }
      str += ch->first;
    }
    return std::nullopt;
  }
};

// Parse "BOZ" binary literal quoted constants.
// As extensions, support X as an alternate hexadecimal marker, and allow
// BOZX markers to appear as suffixes.
struct BOZLiteral {
  using resultType = std::string;
  static std::optional<resultType> Parse(ParseState &state) {
    char base{'\0'};
    auto baseChar{[&base](char ch) -> bool {
      switch (ch) {
      case 'b':
      case 'o':
      case 'z':
        base = ch;
        return true;
      case 'x':
        base = 'z';
        return true;
      default:
        return false;
      }
    }};

    space.Parse(state);
    const char *start{state.GetLocation()};
    std::optional<const char *> at{nextCh.Parse(state)};
    if (!at) {
      return std::nullopt;
    }
    if (**at == 'x' &&
        !state.IsNonstandardOk(
            LanguageFeature::BOZExtensions, "nonstandard BOZ literal"_en_US)) {
      return std::nullopt;
    }
    if (baseChar(**at)) {
      at = nextCh.Parse(state);
      if (!at) {
        return std::nullopt;
      }
    }

    char quote = **at;
    if (quote != '\'' && quote != '"') {
      return std::nullopt;
    }

    std::string content;
    while (true) {
      at = nextCh.Parse(state);
      if (!at) {
        return std::nullopt;
      }
      if (**at == quote) {
        break;
      }
      if (**at == ' ') {
        continue;
      }
      if (!IsHexadecimalDigit(**at)) {
        return std::nullopt;
      }
      content += ToLowerCaseLetter(**at);
    }

    if (!base) {
      // extension: base allowed to appear as suffix, too
      if (!(at = nextCh.Parse(state)) || !baseChar(**at) ||
          !state.IsNonstandardOk(LanguageFeature::BOZExtensions,
              "nonstandard BOZ literal"_en_US)) {
        return std::nullopt;
      }
      spaceCheck.Parse(state);
    }

    if (content.empty()) {
      state.Say(start, "no digit in BOZ literal"_err_en_US);
      return std::nullopt;
    }
    return {std::string{base} + '"' + content + '"'};
  }
};

// R711 digit-string -> digit [digit]...
// N.B. not a token -- no space is skipped
constexpr struct DigitString {
  using resultType = CharBlock;
  static std::optional<resultType> Parse(ParseState &state) {
    if (std::optional<const char *> ch1{state.PeekAtNextChar()}) {
      if (IsDecimalDigit(**ch1)) {
        state.UncheckedAdvance();
        while (std::optional<const char *> p{state.PeekAtNextChar()}) {
          if (!IsDecimalDigit(**p)) {
            break;
          }
          state.UncheckedAdvance();
        }
        return CharBlock{*ch1, state.GetLocation()};
      }
    }
    return std::nullopt;
  }
} digitString;

struct SignedIntLiteralConstantWithoutKind {
  using resultType = CharBlock;
  static std::optional<resultType> Parse(ParseState &state) {
    resultType result{state.GetLocation()};
    static constexpr auto sign{maybe("+-"_ch / space)};
    if (sign.Parse(state)) {
      if (auto digits{digitString.Parse(state)}) {
        result.ExtendToCover(*digits);
        return result;
      }
    }
    return std::nullopt;
  }
};

constexpr struct DigitString64 {
  using resultType = std::uint64_t;
  static std::optional<std::uint64_t> Parse(ParseState &state) {
    std::optional<const char *> firstDigit{digit.Parse(state)};
    if (!firstDigit) {
      return std::nullopt;
    }
    std::uint64_t value = **firstDigit - '0';
    bool overflow{false};
    static constexpr auto getDigit{attempt(digit)};
    while (auto nextDigit{getDigit.Parse(state)}) {
      if (value > std::numeric_limits<std::uint64_t>::max() / 10) {
        overflow = true;
      }
      value *= 10;
      int digitValue = **nextDigit - '0';
      if (value > std::numeric_limits<std::uint64_t>::max() - digitValue) {
        overflow = true;
      }
      value += digitValue;
    }
    if (overflow) {
      state.Say(*firstDigit, "overflow in decimal literal"_err_en_US);
    }
    return {value};
  }
} digitString64;

// R707 signed-int-literal-constant -> [sign] int-literal-constant
// N.B. Spaces are consumed before and after the sign, since the sign
// and the int-literal-constant are distinct tokens.  Does not
// handle a trailing kind parameter.
static std::optional<std::int64_t> SignedInteger(
    const std::optional<std::uint64_t> &x, Location at, bool negate,
    ParseState &state) {
  if (!x) {
    return std::nullopt;
  }
  std::uint64_t limit{std::numeric_limits<std::int64_t>::max()};
  if (negate) {
    limit = -(limit + 1);
  }
  if (*x > limit) {
    state.Say(at, "overflow in signed decimal literal"_err_en_US);
  }
  std::int64_t value = *x;
  return std::make_optional<std::int64_t>(negate ? -value : value);
}

// R710 signed-digit-string -> [sign] digit-string
// N.B. Not a complete token -- no space is skipped.
// Used only in the exponent parts of real literal constants.
struct SignedDigitString {
  using resultType = std::int64_t;
  static std::optional<std::int64_t> Parse(ParseState &state) {
    std::optional<const char *> sign{state.PeekAtNextChar()};
    if (!sign) {
      return std::nullopt;
    }
    bool negate{**sign == '-'};
    if (negate || **sign == '+') {
      state.UncheckedAdvance();
    }
    return SignedInteger(digitString64.Parse(state), *sign, negate, state);
  }
};

// Variants of the above for use in FORMAT specifications, where spaces
// must be ignored.
struct DigitStringIgnoreSpaces {
  using resultType = std::uint64_t;
  static std::optional<std::uint64_t> Parse(ParseState &state) {
    static constexpr auto getFirstDigit{space >> digit};
    std::optional<const char *> firstDigit{getFirstDigit.Parse(state)};
    if (!firstDigit) {
      return std::nullopt;
    }
    std::uint64_t value = **firstDigit - '0';
    bool overflow{false};
    static constexpr auto getDigit{space >> attempt(digit)};
    while (auto nextDigit{getDigit.Parse(state)}) {
      if (value > std::numeric_limits<std::uint64_t>::max() / 10) {
        overflow = true;
      }
      value *= 10;
      int digitValue = **nextDigit - '0';
      if (value > std::numeric_limits<std::uint64_t>::max() - digitValue) {
        overflow = true;
      }
      value += digitValue;
    }
    if (overflow) {
      state.Say(*firstDigit, "overflow in decimal literal"_err_en_US);
    }
    return value;
  }
};

struct PositiveDigitStringIgnoreSpaces {
  using resultType = std::int64_t;
  static std::optional<std::int64_t> Parse(ParseState &state) {
    Location at{state.GetLocation()};
    return SignedInteger(
        DigitStringIgnoreSpaces{}.Parse(state), at, false /*positive*/, state);
  }
};

struct SignedDigitStringIgnoreSpaces {
  using resultType = std::int64_t;
  static std::optional<std::int64_t> Parse(ParseState &state) {
    static constexpr auto getSign{space >> attempt("+-"_ch)};
    bool negate{false};
    if (std::optional<const char *> sign{getSign.Parse(state)}) {
      negate = **sign == '-';
    }
    Location at{state.GetLocation()};
    return SignedInteger(
        DigitStringIgnoreSpaces{}.Parse(state), at, negate, state);
  }
};

// Legacy feature: Hollerith literal constants
struct HollerithLiteral {
  using resultType = std::string;
  static std::optional<std::string> Parse(ParseState &state) {
    space.Parse(state);
    const char *start{state.GetLocation()};
    std::optional<std::uint64_t> charCount{
        DigitStringIgnoreSpaces{}.Parse(state)};
    if (!charCount || *charCount < 1) {
      return std::nullopt;
    }
    static constexpr auto letterH{"h"_ch};
    std::optional<const char *> h{letterH.Parse(state)};
    if (!h) {
      return std::nullopt;
    }
    std::string content;
    for (auto j{*charCount}; j-- > 0;) {
      int chBytes{UTF_8CharacterBytes(state.GetLocation())};
      for (int bytes{chBytes}; bytes > 0; --bytes) {
        if (std::optional<const char *> at{nextCh.Parse(state)}) {
          if (chBytes == 1 && !std::isprint(**at)) {
            state.Say(start, "Bad character in Hollerith"_err_en_US);
            return std::nullopt;
          }
          content += **at;
        } else {
          state.Say(start, "Insufficient characters in Hollerith"_err_en_US);
          return std::nullopt;
        }
      }
    }
    return content;
  }
};

constexpr struct ConsumedAllInputParser {
  using resultType = Success;
  constexpr ConsumedAllInputParser() {}
  static inline std::optional<Success> Parse(ParseState &state) {
    if (state.IsAtEnd()) {
      return {Success{}};
    }
    return std::nullopt;
  }
} consumedAllInput;

template <char goal> struct SkipPast {
  using resultType = Success;
  constexpr SkipPast() {}
  constexpr SkipPast(const SkipPast &) {}
  static std::optional<Success> Parse(ParseState &state) {
    while (std::optional<const char *> p{state.GetNextChar()}) {
      if (**p == goal) {
        return {Success{}};
      }
    }
    return std::nullopt;
  }
};

template <char goal> struct SkipTo {
  using resultType = Success;
  constexpr SkipTo() {}
  constexpr SkipTo(const SkipTo &) {}
  static std::optional<Success> Parse(ParseState &state) {
    while (std::optional<const char *> p{state.PeekAtNextChar()}) {
      if (**p == goal) {
        return {Success{}};
      }
      state.UncheckedAdvance();
    }
    return std::nullopt;
  }
};

// A common idiom in the Fortran grammar is an optional item (usually
// a nonempty comma-separated list) that, if present, must follow a comma
// and precede a doubled colon.  When the item is absent, the comma must
// not appear, and the doubled colons are optional.
//   [[, xyz] ::]     is  optionalBeforeColons(xyz)
//   [[, xyz]... ::]  is  optionalBeforeColons(nonemptyList(xyz))
template <typename PA> inline constexpr auto optionalBeforeColons(const PA &p) {
  using resultType = std::optional<typename PA::resultType>;
  return "," >> construct<resultType>(p) / "::" ||
      ("::"_tok || !","_tok) >> pure<resultType>();
}
template <typename PA>
inline constexpr auto optionalListBeforeColons(const PA &p) {
  using resultType = std::list<typename PA::resultType>;
  return "," >> nonemptyList(p) / "::" ||
      ("::"_tok || !","_tok) >> pure<resultType>();
}

// Skip over empty lines, leading spaces, and some compiler directives (viz.,
// the ones that specify the source form) that might appear before the
// next statement.  Skip over empty statements (bare semicolons) when
// not in strict standard conformance mode.  Always succeeds.
constexpr struct SkipStuffBeforeStatement {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &state) {
    if (UserState * ustate{state.userState()}) {
      if (ParsingLog * log{ustate->log()}) {
        // Save memory: vacate the parsing log before each statement unless
        // we're logging the whole parse for debugging.
        if (!ustate->instrumentedParse()) {
          log->clear();
        }
      }
    }
    while (std::optional<const char *> at{state.PeekAtNextChar()}) {
      if (**at == '\n' || **at == ' ') {
        state.UncheckedAdvance();
      } else if (**at == '!') {
        static const char fixed[] = "!dir$ fixed\n", free[] = "!dir$ free\n";
        static constexpr std::size_t fixedBytes{sizeof fixed - 1};
        static constexpr std::size_t freeBytes{sizeof free - 1};
        std::size_t remain{state.BytesRemaining()};
        if (remain >= fixedBytes && std::memcmp(*at, fixed, fixedBytes) == 0) {
          state.set_inFixedForm(true).UncheckedAdvance(fixedBytes);
        } else if (remain >= freeBytes &&
            std::memcmp(*at, free, freeBytes) == 0) {
          state.set_inFixedForm(false).UncheckedAdvance(freeBytes);
        } else {
          break;
        }
      } else if (**at == ';' &&
          state.IsNonstandardOk(
              LanguageFeature::EmptyStatement, "empty statement"_en_US)) {
        state.UncheckedAdvance();
      } else {
        break;
      }
    }
    return {Success{}};
  }
} skipStuffBeforeStatement;

// R602 underscore -> _
constexpr auto underscore{"_"_ch};

// Characters besides letters and digits that may appear in names.
// N.B. Don't accept an underscore if it is immediately followed by a
// quotation mark, so that kindParam_"character literal" is parsed properly.
// PGI and ifort accept '$' in identifiers, even as the initial character.
// Cray and gfortran accept '$', but not as the first character.
// Cray accepts '@' as well.
constexpr auto otherIdChar{underscore / !"'\""_ch ||
    extension<LanguageFeature::PunctuationInNames>("$@"_ch)};

constexpr auto logicalTRUE{
    (".TRUE."_tok ||
        extension<LanguageFeature::LogicalAbbreviations>(".T."_tok)) >>
    pure(true)};
constexpr auto logicalFALSE{
    (".FALSE."_tok ||
        extension<LanguageFeature::LogicalAbbreviations>(".F."_tok)) >>
    pure(false)};

// deprecated: Hollerith literals
constexpr auto rawHollerithLiteral{
    deprecated<LanguageFeature::Hollerith>(HollerithLiteral{})};

template <typename A> constexpr decltype(auto) verbatim(A x) {
  return sourced(construct<Verbatim>(x));
}

} // namespace Fortran::parser
#endif // FORTRAN_PARSER_TOKEN_PARSERS_H_
