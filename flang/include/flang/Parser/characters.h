//===-- include/flang/Parser/characters.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_CHARACTERS_H_
#define FORTRAN_PARSER_CHARACTERS_H_

// Define some character classification predicates and
// conversions here to avoid dependences upon <cctype> and
// also to accomodate Fortran tokenization.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace Fortran::parser {

extern bool useHexadecimalEscapeSequences;

// We can easily support Fortran program source in any character
// set whose first 128 code points correspond to ASCII codes 0-127 (ISO/IEC646).
// The specific encodings that we can handle include:
//   LATIN_1: ISO 8859-1 Latin-1
//   UTF_8: Multi-byte encoding of Unicode (ISO/IEC 10646)
enum class Encoding { LATIN_1, UTF_8 };

inline constexpr bool IsUpperCaseLetter(char ch) {
  return ch >= 'A' && ch <= 'Z';
}

inline constexpr bool IsLowerCaseLetter(char ch) {
  return ch >= 'a' && ch <= 'z';
}

inline constexpr bool IsLetter(char ch) {
  return IsUpperCaseLetter(ch) || IsLowerCaseLetter(ch);
}

inline constexpr bool IsDecimalDigit(char ch) { return ch >= '0' && ch <= '9'; }

inline constexpr bool IsHexadecimalDigit(char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') ||
      (ch >= 'a' && ch <= 'f');
}

inline constexpr bool IsOctalDigit(char ch) { return ch >= '0' && ch <= '7'; }

inline constexpr bool IsLegalIdentifierStart(char ch) {
  return IsLetter(ch) || ch == '_' || ch == '@' || ch == '$';
}

inline constexpr bool IsLegalInIdentifier(char ch) {
  return IsLegalIdentifierStart(ch) || IsDecimalDigit(ch);
}

inline constexpr bool IsPrintable(char ch) { return ch >= ' ' && ch <= '~'; }

inline constexpr bool IsWhiteSpace(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' ||
      ch == '\r';
}

inline constexpr char ToLowerCaseLetter(char ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

inline constexpr char ToLowerCaseLetter(char &&ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

inline std::string ToLowerCaseLetters(std::string_view str) {
  std::string lowered{str};
  for (char &ch : lowered) {
    ch = ToLowerCaseLetter(ch);
  }
  return lowered;
}

inline constexpr char ToUpperCaseLetter(char ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

inline constexpr char ToUpperCaseLetter(char &&ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

inline std::string ToUpperCaseLetters(std::string_view str) {
  std::string raised{str};
  for (char &ch : raised) {
    ch = ToUpperCaseLetter(ch);
  }
  return raised;
}

inline constexpr bool IsSameApartFromCase(char x, char y) {
  return ToLowerCaseLetter(x) == ToLowerCaseLetter(y);
}

inline constexpr char DecimalDigitValue(char ch) { return ch - '0'; }

inline constexpr char HexadecimalDigitValue(char ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 10
      : IsLowerCaseLetter(ch)  ? ch - 'a' + 10
                               : DecimalDigitValue(ch);
}

inline constexpr std::optional<char> BackslashEscapeValue(char ch) {
  switch (ch) {
  case 'a':
    return std::nullopt; // '\a';  PGF90 doesn't know \a
  case 'b':
    return '\b';
  case 'f':
    return '\f';
  case 'n':
    return '\n';
  case 'r':
    return '\r';
  case 't':
    return '\t';
  case 'v':
    return '\v';
  case '"':
  case '\'':
  case '\\':
    return ch;
  default:
    return std::nullopt;
  }
}

inline constexpr std::optional<char> BackslashEscapeChar(char ch) {
  switch (ch) {
  case '\a':
    return std::nullopt; // 'a';  PGF90 doesn't know \a
  case '\b':
    return 'b';
  case '\f':
    return 'f';
  case '\n':
    return 'n';
  case '\r':
    return 'r';
  case '\t':
    return 't';
  case '\v':
    return 'v';
  case '"':
  case '\'':
  case '\\':
    return ch;
  default:
    return std::nullopt;
  }
}

// Does not include spaces or line ending characters.
inline constexpr bool IsValidFortranTokenCharacter(char ch) {
  switch (ch) {
  case '"':
  case '%':
  case '\'':
  case '(':
  case ')':
  case '*':
  case '+':
  case ',':
  case '-':
  case '.':
  case '/':
  case ':':
  case ';':
  case '<':
  case '=':
  case '>':
  case '[':
  case ']':
    return true;
  default:
    return IsLegalIdentifierStart(ch) || IsDecimalDigit(ch);
  }
}

struct EncodedCharacter {
  static constexpr int maxEncodingBytes{6};
  char buffer[maxEncodingBytes];
  int bytes{0};
};

template <Encoding ENCODING> EncodedCharacter EncodeCharacter(char32_t ucs);
template <> EncodedCharacter EncodeCharacter<Encoding::LATIN_1>(char32_t);
template <> EncodedCharacter EncodeCharacter<Encoding::UTF_8>(char32_t);

EncodedCharacter EncodeCharacter(Encoding, char32_t ucs);

template <Encoding ENCODING, typename STRING>
std::string EncodeString(const STRING &);
extern template std::string EncodeString<Encoding::LATIN_1, std::string>(
    const std::string &);
extern template std::string EncodeString<Encoding::UTF_8, std::u32string>(
    const std::u32string &);

// EmitQuotedChar drives callbacks "emit" and "insert" to output the
// bytes of an encoding for a codepoint.
template <typename NORMAL, typename INSERTED>
void EmitQuotedChar(char32_t ch, const NORMAL &emit, const INSERTED &insert,
    bool backslashEscapes = true, Encoding encoding = Encoding::UTF_8) {
  auto emitOneByte{[&](std::uint8_t ch) {
    if (backslashEscapes && (ch < ' ' || ch >= 0x7f || ch == '\\')) {
      if (std::optional<char> escape{BackslashEscapeChar(ch)}) {
        insert('\\');
        emit(*escape);
      } else if (useHexadecimalEscapeSequences) {
        insert('\\');
        insert('x');
        int top{ch >> 4}, bottom{ch & 0xf};
        insert(top > 9 ? 'a' + top - 10 : '0' + top);
        insert(bottom > 9 ? 'a' + bottom - 10 : '0' + bottom);
      } else {
        // octal escape sequence; always emit 3 digits to avoid ambiguity
        insert('\\');
        insert('0' + (ch >> 6));
        insert('0' + ((ch >> 3) & 7));
        insert('0' + (ch & 7));
      }
    } else if (ch == '\n') { // always escape newlines
      insert('\\');
      insert('n');
    } else {
      emit(ch);
    }
  }};
  if (ch <= 0x7f) {
    emitOneByte(ch);
  } else if (backslashEscapes && useHexadecimalEscapeSequences) {
    insert('\\');
    insert('u');
    if (ch > 0xffff) {
      unsigned c1{(ch >> 28) & 0xf}, c2{(ch >> 24) & 0xf}, c3{(ch >> 20) & 0xf},
          c4{(ch >> 16) & 0xf};
      insert(c1 > 9 ? 'a' + c1 - 10 : '0' + c1);
      insert(c2 > 9 ? 'a' + c2 - 10 : '0' + c2);
      insert(c3 > 9 ? 'a' + c3 - 10 : '0' + c3);
      insert(c4 > 9 ? 'a' + c4 - 10 : '0' + c4);
    }
    unsigned c1{(ch >> 12) & 0xf}, c2{(ch >> 8) & 0xf}, c3{(ch >> 4) & 0xf},
        c4{ch & 0xf};
    insert(c1 > 9 ? 'a' + c1 - 10 : '0' + c1);
    insert(c2 > 9 ? 'a' + c2 - 10 : '0' + c2);
    insert(c3 > 9 ? 'a' + c3 - 10 : '0' + c3);
    insert(c4 > 9 ? 'a' + c4 - 10 : '0' + c4);
  } else {
    EncodedCharacter encoded{EncodeCharacter(encoding, ch)};
    for (int j{0}; j < encoded.bytes; ++j) {
      emitOneByte(encoded.buffer[j]);
    }
  }
}

std::string QuoteCharacterLiteral(const std::string &,
    bool backslashEscapes = true, Encoding = Encoding::LATIN_1);
std::string QuoteCharacterLiteral(const std::u16string &,
    bool backslashEscapes = true, Encoding = Encoding::UTF_8);
std::string QuoteCharacterLiteral(const std::u32string &,
    bool backslashEscapes = true, Encoding = Encoding::UTF_8);

int UTF_8CharacterBytes(const char *);

struct DecodedCharacter {
  char32_t codepoint{0};
  int bytes{0}; // signifying failure
};

template <Encoding ENCODING>
DecodedCharacter DecodeRawCharacter(const char *, std::size_t);
template <>
DecodedCharacter DecodeRawCharacter<Encoding::LATIN_1>(
    const char *, std::size_t);

template <>
DecodedCharacter DecodeRawCharacter<Encoding::UTF_8>(const char *, std::size_t);

// DecodeCharacter optionally handles backslash escape sequences, too.
template <Encoding ENCODING>
DecodedCharacter DecodeCharacter(
    const char *, std::size_t, bool backslashEscapes);
extern template DecodedCharacter DecodeCharacter<Encoding::LATIN_1>(
    const char *, std::size_t, bool);
extern template DecodedCharacter DecodeCharacter<Encoding::UTF_8>(
    const char *, std::size_t, bool);

DecodedCharacter DecodeCharacter(
    Encoding, const char *, std::size_t, bool backslashEscapes);

template <typename RESULT, Encoding ENCODING>
RESULT DecodeString(const std::string &, bool backslashEscapes);
extern template std::string DecodeString<std::string, Encoding::LATIN_1>(
    const std::string &, bool);
extern template std::u16string DecodeString<std::u16string, Encoding::UTF_8>(
    const std::string &, bool);
extern template std::u32string DecodeString<std::u32string, Encoding::UTF_8>(
    const std::string &, bool);
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_CHARACTERS_H_
