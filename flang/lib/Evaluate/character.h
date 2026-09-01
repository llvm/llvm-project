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
  using Character = Scalar<Type<TypeCategory::Character>>;
  using CharT = char32_t;

public:
  // CHAR also implements ACHAR under assumption that character encodings
  // contain ASCII
  static Character CHAR(int kind, std::uint64_t code) {
    return Character{kind, 1, static_cast<CharT>(code)};
  }

  // ICHAR also implements IACHAR under assumption that character encodings
  // contain ASCII
  static std::int64_t ICHAR(const Character &c) {
    CHECK(c.length() == 1);
    // Mask to the character kind width to avoid sign extension
    auto ch{static_cast<std::uint64_t>(c[0])};
    switch (c.kind()) {
    case 1:
      return static_cast<std::int64_t>(ch & 0xffu);
    case 2:
      return static_cast<std::int64_t>(ch & 0xffffu);
    case 4:
      return static_cast<std::int64_t>(ch & 0xffffffffu);
    }
    DIE("unsupported character kind");
  }

  static Character NEW_LINE(int kind) { return Character{kind, 1, NewLine()}; }

  static Character ADJUSTL(const Character &str) {
    const int kind{str.kind()};
    auto pos{str.find_first_not_of(Space())};
    if (pos != Character::npos && pos != 0) {
      return Character{str.substr(pos) + Character{kind, pos, Space()}};
    }
    // else empty or only spaces, or no leading spaces
    return str;
  }

  static Character ADJUSTR(const Character &str) {
    const int kind{str.kind()};
    auto pos{str.find_last_not_of(Space())};
    if (pos != Character::npos && pos != str.length() - 1) {
      auto delta{str.length() - 1 - pos};
      return Character{
          Character{kind, delta, Space()} + str.substr(0, pos + 1)};
    }
    // else empty or only spaces, or no trailing spaces
    return str;
  }

  static ConstantSubscript INDEX(
      const Character &str, const Character &substr, bool back = false) {
    auto pos{back ? str.rfind(substr) : str.find(substr)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript SCAN(
      const Character &str, const Character &set, bool back = false) {
    auto pos{back ? str.find_last_of(set) : str.find_first_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  static ConstantSubscript VERIFY(
      const Character &str, const Character &set, bool back = false) {
    auto pos{back ? str.find_last_not_of(set) : str.find_first_not_of(set)};
    return static_cast<ConstantSubscript>(pos == str.npos ? 0 : pos + 1);
  }

  // Resize adds spaces on the right if the new size is bigger than the
  // original, or by trimming the rightmost characters otherwise.
  static Character Resize(const Character &str, std::size_t newLength) {
    const int kind{str.kind()};
    auto oldLength{str.length()};
    if (newLength > oldLength) {
      return str + Character{kind, newLength - oldLength, Space()};
    } else {
      return str.substr(0, newLength);
    }
  }

  static ConstantSubscript LEN_TRIM(const Character &str) {
    auto j{str.length()};
    for (; j >= 1; --j) {
      if (str[j - 1] != ' ') {
        break;
      }
    }
    return static_cast<ConstantSubscript>(j);
  }

  static Character REPEAT(const Character &str, ConstantSubscript ncopies) {
    const int kind{str.kind()};
    Character result{Character::Zero(kind)};
    if (!str.empty() && ncopies > 0) {
      result.reserve(ncopies * str.size());
      while (ncopies-- > 0) {
        result += str;
      }
    }
    return result;
  }

  static Character TRIM(const Character &str) {
    return str.substr(0, LEN_TRIM(str));
  }

private:
  // Following helpers assume that character encodings contain ASCII
  static constexpr CharT Space() { return 0x20; }
  static constexpr CharT NewLine() { return 0x0a; }
};

} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_CHARACTER_H_
