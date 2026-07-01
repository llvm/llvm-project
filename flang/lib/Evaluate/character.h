//===-- lib/Evaluate/character.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHARACTER_H_
#define FORTRAN_EVALUATE_CHARACTER_H_

#include "flang/Evaluate/character-value.h"
#include "flang/Evaluate/type.h"
#include <cstdint>
#include <string>

// Provides implementations of intrinsic functions operating on character
// scalars.

namespace Fortran::evaluate {

class CharacterUtils {
  using CharacterValue = Scalar<Type<TypeCategory::Character>>;
  using CharT = char32_t;

public:
  // CHAR also implements ACHAR under assumption that character encodings
  // contain ASCII
  static CharacterValue CHAR(std::uint64_t code, int kind) {
    return CharacterOfKind(1, static_cast<CharT>(code), kind);
  }

  // ICHAR also implements IACHAR under assumption that character encodings
  // contain ASCII
  static std::int64_t ICHAR(const CharacterValue &c) {
    CHECK(c.length() == 1);
    // Mask to the character kind width to avoid sign extension for KIND=1.
    auto ch{static_cast<std::uint64_t>(c[0])};
    switch (c.kind()) {
    case 1:
      return static_cast<std::int64_t>(ch & 0xffu);
    case 2:
      return static_cast<std::int64_t>(ch & 0xffffu);
    case 4:
      return static_cast<std::int64_t>(ch & 0xffffffffu);
    }
    llvm_unreachable("unsupported character kind");
  }

  static CharacterValue NEW_LINE(int kind) {
    return CharacterOfKind(1, NewLine(), kind);
  }

  static CharacterValue ADJUSTL(const CharacterValue &str) {
    auto pos{str.find_first_not_of(Space())};
    if (pos != CharacterValue::npos && pos != 0) {
      return CharacterValue{
          str.substr(pos) + CharacterOfKind(pos, Space(), str.kind())};
    }
    // else empty or only spaces, or no leading spaces
    return str;
  }

  static CharacterValue ADJUSTR(const CharacterValue &str) {
    auto pos{str.find_last_not_of(Space())};
    if (pos != CharacterValue::npos && pos != str.length() - 1) {
      auto delta{str.length() - 1 - pos};
      return CharacterValue{
          CharacterOfKind(delta, Space(), str.kind()) + str.substr(0, pos + 1)};
    }
    // else empty or only spaces, or no trailing spaces
    return str;
  }

  static ConstantSubscript INDEX(const CharacterValue &str,
      const CharacterValue &substr, bool back = false) {
    auto pos{back ? str.rfind(substr) : str.find(substr)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript SCAN(
      const CharacterValue &str, const CharacterValue &set, bool back = false) {
    auto pos{back ? str.find_last_of(set) : str.find_first_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript VERIFY(
      const CharacterValue &str, const CharacterValue &set, bool back = false) {
    auto pos{back ? str.find_last_not_of(set) : str.find_first_not_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  // Resize adds spaces on the right if the new size is bigger than the
  // original, or by trimming the rightmost characters otherwise.
  static CharacterValue Resize(
      const CharacterValue &str, std::size_t newLength) {
    auto oldLength{str.length()};
    if (newLength > oldLength) {
      return str + CharacterOfKind(newLength - oldLength, Space(), str.kind());
    } else {
      return str.substr(0, newLength);
    }
  }

  static ConstantSubscript LEN_TRIM(const CharacterValue &str) {
    auto j{str.length()};
    for (; j >= 1; --j) {
      if (str[j - 1] != CharT{0x20}) {
        break;
      }
    }
    return static_cast<ConstantSubscript>(j);
  }

  static CharacterValue REPEAT(
      const CharacterValue &str, ConstantSubscript ncopies) {
    CharacterValue result{CharacterOfKind(0, CharT{}, str.kind())};
    if (!str.empty() && ncopies > 0) {
      result.reserve(ncopies * str.size());
      while (ncopies-- > 0) {
        result += str;
      }
    }
    return result;
  }

  static CharacterValue TRIM(const CharacterValue &str) {
    return str.substr(0, LEN_TRIM(str));
  }

private:
  // Following helpers assume that character encodings contain ASCII
  static constexpr CharT Space() { return 0x20; }
  static constexpr CharT NewLine() { return 0x0a; }

  static CharacterValue CharacterOfKind(std::size_t n, CharT ch, int kind) {
    switch (kind) {
    case 1:
      return CharacterValue(n, static_cast<char>(ch));
    case 2:
      return CharacterValue(n, static_cast<char16_t>(ch));
    case 4:
      return CharacterValue(n, ch);
    }
    llvm_unreachable("unsupported character kind");
  }
};

} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_CHARACTER_H_
