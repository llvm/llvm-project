//===-- lib/Parser/characters.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/characters.h"
#include "flang/Common/idioms.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace Fortran::parser {

bool useHexadecimalEscapeSequences{false};

int UTF_8CharacterBytes(const char *p) {
  if ((*p & 0x80) == 0) {
    return 1;
  } else if ((*p & 0xe0) == 0xc0) {
    return 2;
  } else if ((*p & 0xf0) == 0xe0) {
    return 3;
  } else if ((*p & 0xf8) == 0xf0) {
    return 4;
  } else if ((*p & 0xfc) == 0xf8) {
    return 5;
  } else {
    return 6;
  }
}

template <typename STRING>
std::string QuoteCharacterLiteralHelper(
    const STRING &str, bool backslashEscapes, Encoding encoding) {
  std::string result{'"'};
  const auto emit{[&](char ch) { result += ch; }};
  for (auto ch : str) {
    using CharT = std::decay_t<decltype(ch)>;
    char32_t ch32{static_cast<std::make_unsigned_t<CharT>>(ch)};
    if (ch32 == static_cast<unsigned char>('"')) {
      emit('"'); // double the " when it appears in the text
    }
    EmitQuotedChar(ch32, emit, emit, backslashEscapes, encoding);
  }
  result += '"';
  return result;
}

std::string QuoteCharacterLiteral(
    const std::string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

std::string QuoteCharacterLiteral(
    const std::u16string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

std::string QuoteCharacterLiteral(
    const std::u32string &str, bool backslashEscapes, Encoding encoding) {
  return QuoteCharacterLiteralHelper(str, backslashEscapes, encoding);
}

template <> EncodedCharacter EncodeCharacter<Encoding::LATIN_1>(char32_t ucs) {
  CHECK(ucs <= 0xff);
  EncodedCharacter result;
  result.buffer[0] = ucs;
  result.bytes = 1;
  return result;
}

template <> EncodedCharacter EncodeCharacter<Encoding::UTF_8>(char32_t ucs) {
  // N.B. char32_t is unsigned
  EncodedCharacter result;
  if (ucs <= 0x7f) {
    result.buffer[0] = ucs;
    result.bytes = 1;
  } else if (ucs <= 0x7ff) {
    result.buffer[0] = 0xc0 | (ucs >> 6);
    result.buffer[1] = 0x80 | (ucs & 0x3f);
    result.bytes = 2;
  } else if (ucs <= 0xffff) {
    result.buffer[0] = 0xe0 | (ucs >> 12);
    result.buffer[1] = 0x80 | ((ucs >> 6) & 0x3f);
    result.buffer[2] = 0x80 | (ucs & 0x3f);
    result.bytes = 3;
  } else if (ucs <= 0x1fffff) {
    // UCS actually only goes up to 0x10ffff, but the
    // UTF-8 encoding can handle 32 bits.
    result.buffer[0] = 0xf0 | (ucs >> 18);
    result.buffer[1] = 0x80 | ((ucs >> 12) & 0x3f);
    result.buffer[2] = 0x80 | ((ucs >> 6) & 0x3f);
    result.buffer[3] = 0x80 | (ucs & 0x3f);
    result.bytes = 4;
  } else if (ucs <= 0x3ffffff) {
    result.buffer[0] = 0xf8 | (ucs >> 24);
    result.buffer[1] = 0x80 | ((ucs >> 18) & 0x3f);
    result.buffer[2] = 0x80 | ((ucs >> 12) & 0x3f);
    result.buffer[3] = 0x80 | ((ucs >> 6) & 0x3f);
    result.buffer[4] = 0x80 | (ucs & 0x3f);
    result.bytes = 5;
  } else {
    result.buffer[0] = 0xfc | (ucs >> 30);
    result.buffer[1] = 0x80 | ((ucs >> 24) & 0x3f);
    result.buffer[2] = 0x80 | ((ucs >> 18) & 0x3f);
    result.buffer[3] = 0x80 | ((ucs >> 12) & 0x3f);
    result.buffer[4] = 0x80 | ((ucs >> 6) & 0x3f);
    result.buffer[5] = 0x80 | (ucs & 0x3f);
    result.bytes = 6;
  }
  return result;
}

EncodedCharacter EncodeCharacter(Encoding encoding, char32_t ucs) {
  switch (encoding) {
    SWITCH_COVERS_ALL_CASES
  case Encoding::LATIN_1:
    return EncodeCharacter<Encoding::LATIN_1>(ucs);
  case Encoding::UTF_8:
    return EncodeCharacter<Encoding::UTF_8>(ucs);
  }
}

template <Encoding ENCODING, typename STRING>
std::string EncodeString(const STRING &str) {
  std::string result;
  for (auto ch : str) {
    char32_t uch{static_cast<std::make_unsigned_t<decltype(ch)>>(ch)};
    EncodedCharacter encoded{EncodeCharacter<ENCODING>(uch)};
    result.append(encoded.buffer, static_cast<std::size_t>(encoded.bytes));
  }
  return result;
}

template std::string EncodeString<Encoding::LATIN_1, std::string>(
    const std::string &);
template std::string EncodeString<Encoding::UTF_8, std::u16string>(
    const std::u16string &);
template std::string EncodeString<Encoding::UTF_8, std::u32string>(
    const std::u32string &);

template <>
DecodedCharacter DecodeRawCharacter<Encoding::LATIN_1>(
    const char *cp, std::size_t bytes) {
  if (bytes >= 1) {
    return {*reinterpret_cast<const std::uint8_t *>(cp), 1};
  } else {
    return {};
  }
}

template <>
DecodedCharacter DecodeRawCharacter<Encoding::UTF_8>(
    const char *cp, std::size_t bytes) {
  auto p{reinterpret_cast<const std::uint8_t *>(cp)};
  char32_t ch{*p};
  if (ch <= 0x7f) {
    return {ch, 1};
  } else if ((ch & 0xf8) == 0xf0 && bytes >= 4 && ch > 0xf0 &&
      ((p[1] | p[2] | p[3]) & 0xc0) == 0x80) {
    ch = ((ch & 7) << 6) | (p[1] & 0x3f);
    ch = (ch << 6) | (p[2] & 0x3f);
    ch = (ch << 6) | (p[3] & 0x3f);
    return {ch, 4};
  } else if ((ch & 0xf0) == 0xe0 && bytes >= 3 && ch > 0xe0 &&
      ((p[1] | p[2]) & 0xc0) == 0x80) {
    ch = ((ch & 0xf) << 6) | (p[1] & 0x3f);
    ch = (ch << 6) | (p[2] & 0x3f);
    return {ch, 3};
  } else if ((ch & 0xe0) == 0xc0 && bytes >= 2 && ch > 0xc0 &&
      (p[1] & 0xc0) == 0x80) {
    ch = ((ch & 0x1f) << 6) | (p[1] & 0x3f);
    return {ch, 2};
  } else {
    return {}; // not valid UTF-8
  }
}

static DecodedCharacter DecodeEscapedCharacter(
    const char *cp, std::size_t bytes) {
  if (cp[0] == '\\' && bytes >= 2) {
    if (std::optional<char> escChar{BackslashEscapeValue(cp[1])}) {
      return {static_cast<unsigned char>(*escChar), 2};
    } else if (IsOctalDigit(cp[1])) {
      std::size_t maxLen{std::min(std::size_t{4}, bytes)};
      char32_t code{static_cast<char32_t>(DecimalDigitValue(cp[1]))};
      std::size_t len{2}; // so far
      for (; code <= 037 && len < maxLen && IsOctalDigit(cp[len]); ++len) {
        code = 8 * code + DecimalDigitValue(cp[len]);
      }
      return {code, static_cast<int>(len)};
    } else if (bytes >= 4 && ToLowerCaseLetter(cp[1]) == 'x' &&
        IsHexadecimalDigit(cp[2]) && IsHexadecimalDigit(cp[3])) {
      return {static_cast<char32_t>(16 * HexadecimalDigitValue(cp[2]) +
                  HexadecimalDigitValue(cp[3])),
          4};
    } else if (IsLetter(cp[1])) {
      // Unknown escape - ignore the '\' (PGI compatibility)
      return {static_cast<unsigned char>(cp[1]), 2};
    } else {
      // Not an escape character.
      return {'\\', 1};
    }
  }
  return {static_cast<unsigned char>(cp[0]), 1};
}

template <Encoding ENCODING>
static DecodedCharacter DecodeEscapedCharacters(
    const char *cp, std::size_t bytes) {
  char buffer[EncodedCharacter::maxEncodingBytes];
  int count[EncodedCharacter::maxEncodingBytes];
  std::size_t at{0}, len{0};
  for (; len < EncodedCharacter::maxEncodingBytes && at < bytes; ++len) {
    DecodedCharacter code{DecodeEscapedCharacter(cp + at, bytes - at)};
    buffer[len] = code.codepoint;
    at += code.bytes;
    count[len] = at;
  }
  DecodedCharacter code{DecodeCharacter<ENCODING>(buffer, len, false)};
  if (code.bytes > 0) {
    code.bytes = count[code.bytes - 1];
  } else {
    code.codepoint = buffer[0] & 0xff;
    code.bytes = count[0];
  }
  return code;
}

template <Encoding ENCODING>
DecodedCharacter DecodeCharacter(
    const char *cp, std::size_t bytes, bool backslashEscapes) {
  if (backslashEscapes && bytes >= 2 && *cp == '\\') {
    return DecodeEscapedCharacters<ENCODING>(cp, bytes);
  } else {
    return DecodeRawCharacter<ENCODING>(cp, bytes);
  }
}

template DecodedCharacter DecodeCharacter<Encoding::LATIN_1>(
    const char *, std::size_t, bool);
template DecodedCharacter DecodeCharacter<Encoding::UTF_8>(
    const char *, std::size_t, bool);

DecodedCharacter DecodeCharacter(Encoding encoding, const char *cp,
    std::size_t bytes, bool backslashEscapes) {
  switch (encoding) {
    SWITCH_COVERS_ALL_CASES
  case Encoding::LATIN_1:
    return DecodeCharacter<Encoding::LATIN_1>(cp, bytes, backslashEscapes);
  case Encoding::UTF_8:
    return DecodeCharacter<Encoding::UTF_8>(cp, bytes, backslashEscapes);
  }
}

template <typename RESULT, Encoding ENCODING>
RESULT DecodeString(const std::string &s, bool backslashEscapes) {
  RESULT result;
  const char *p{s.c_str()};
  for (auto bytes{s.size()}; bytes != 0;) {
    DecodedCharacter decoded{
        DecodeCharacter<ENCODING>(p, bytes, backslashEscapes)};
    if (decoded.bytes > 0) {
      if (static_cast<std::size_t>(decoded.bytes) <= bytes) {
        result.append(1, decoded.codepoint);
        bytes -= decoded.bytes;
        p += decoded.bytes;
        continue;
      }
    }
    result.append(1, static_cast<uint8_t>(*p));
    ++p;
    --bytes;
  }
  return result;
}

template std::string DecodeString<std::string, Encoding::LATIN_1>(
    const std::string &, bool);
template std::u16string DecodeString<std::u16string, Encoding::UTF_8>(
    const std::string &, bool);
template std::u32string DecodeString<std::u32string, Encoding::UTF_8>(
    const std::string &, bool);
} // namespace Fortran::parser
