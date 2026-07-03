//===-- include/flang/Evaluate/character-value.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHAR_VALUE_H_
#define FORTRAN_EVALUATE_CHAR_VALUE_H_

#include "flang/Evaluate/object-sizes.h"
#include <cstddef>
#include <iosfwd>
#include <optional>

namespace Fortran::evaluate::value {

class CharacterValueImpl;

// ----------------------------------------------------------------------------
// CharacterValue: runtime-kind character string (kind 1/2/4).
// Interface compatible to std::string
// ----------------------------------------------------------------------------
class CharacterValue {
public:
  CharacterValue();
  CharacterValue(const CharacterValue &);
  CharacterValue(CharacterValue &&);
  ~CharacterValue();
  CharacterValue &operator=(const CharacterValue &);
  CharacterValue &operator=(CharacterValue &&);

  explicit CharacterValue(std::string s);
  explicit CharacterValue(std::u16string s);
  explicit CharacterValue(std::u32string s);

  // Fill constructors: create a string of n copies of the given character.
  CharacterValue(std::size_t n, char c);
  CharacterValue(std::size_t n, char16_t c);
  CharacterValue(std::size_t n, char32_t c);

  // Comparison operators
  bool operator==(const CharacterValue &y) const;
  bool operator!=(const CharacterValue &y) const { return !(*this == y); }
  bool operator<(const CharacterValue &y) const;
  bool operator>(const CharacterValue &y) const { return y < *this; }
  bool operator<=(const CharacterValue &y) const { return !(y < *this); }
  bool operator>=(const CharacterValue &y) const { return !(*this < y); }

  int kind() const;
  int bits() const;
  bool IsMonostate() const;
  bool IsZero() const;
  static CharacterValue Zero(int kind);

  std::size_t size() const;

  // String length (synonym for size()).
  std::size_t length() const { return size(); }

  // True when the string is empty.
  bool empty() const { return size() == 0; }

  // Assign n copies of the given character, fixing the kind from the char type.
  void assign(std::size_t n, char c);
  void assign(std::size_t n, char16_t c);
  void assign(std::size_t n, char32_t c);

  // Assign from a raw character pointer and length.
  void assign(const char *p, std::size_t n);
  void assign(const char16_t *p, std::size_t n);
  void assign(const char32_t *p, std::size_t n);

  // Erase from position pos to end.
  void erase(std::size_t pos);

  // Append n copies of the given character (widened to the stored type).
  void append(std::size_t n, char c);
  void append(std::size_t n, char16_t c);
  void append(std::size_t n, char32_t c);

  // Replace the substring [pos, pos+len) with characters from other.
  CharacterValue &replace(
      std::size_t pos, std::size_t len, const CharacterValue &other);

  // Return a suffix starting at pos.
  CharacterValue substr(std::size_t pos) const;

  // Return a substring of len characters starting at pos.
  CharacterValue substr(std::size_t pos, std::size_t len) const;

  // Return the string as std::string if kind==1, or nullopt otherwise.
  std::optional<std::string> ToStdString() const;

  // Reserve storage for at least n characters.
  void reserve(std::size_t n);

  // Return the character at position i as char32_t (safe for all kinds).
  char32_t operator[](std::size_t i) const;

  // Concatenate two same-kind strings.
  CharacterValue operator+(const CharacterValue &y) const;

  // Append another same-kind string.
  CharacterValue &operator+=(const CharacterValue &y);

  // Append a character, converting it to the string's element type.
  CharacterValue &operator+=(char c);

  // Sentinel value for "not found" positions (same as std::string::npos).
  static constexpr std::size_t npos{~std::size_t{0}};

  // Find-family methods; return npos when not found.
  // Argument is a scalar character (widened to char32_t for dispatch).
  std::size_t find_first_not_of(char c) const {
    return find_first_not_of_char(static_cast<char32_t>(c));
  }
  std::size_t find_first_not_of(char16_t c) const {
    return find_first_not_of_char(static_cast<char32_t>(c));
  }
  std::size_t find_first_not_of(char32_t c) const {
    return find_first_not_of_char(c);
  }
  std::size_t find_last_not_of(char c) const {
    return find_last_not_of_char(static_cast<char32_t>(c));
  }
  std::size_t find_last_not_of(char16_t c) const {
    return find_last_not_of_char(static_cast<char32_t>(c));
  }
  std::size_t find_last_not_of(char32_t c) const {
    return find_last_not_of_char(c);
  }

  // Find-family methods using a CharacterValue as the character set.
  std::size_t find_first_not_of(const CharacterValue &set) const;
  std::size_t find_last_not_of(const CharacterValue &set) const;

  // Find-family methods taking a CharacterValue set/pattern.
  std::size_t find(const CharacterValue &pattern) const;
  std::size_t rfind(const CharacterValue &pattern) const;
  std::size_t find_first_of(const CharacterValue &set) const;
  std::size_t find_last_of(const CharacterValue &set) const;

  // Byte size of one character unit (1, 2, or 4).
  std::size_t charSize() const;

  // Raw byte pointer to the underlying character data.
  const void *data() const;
  void *charData();
  const void *charData() const;

  CharacterValueImpl &value() {
    return *reinterpret_cast<CharacterValueImpl *>(this);
  }
  const CharacterValueImpl &value() const {
    return *reinterpret_cast<const CharacterValueImpl *>(this);
  }

private:
  std::size_t find_first_not_of_char(char32_t c) const;
  std::size_t find_last_not_of_char(char32_t c) const;

  alignas(
      detail::kCharacterObjectAlign) char opaque_[detail::kCharacterObjectSize];
};

static_assert(sizeof(CharacterValue) == detail::kCharacterObjectSize);
static_assert(alignof(CharacterValue) == detail::kCharacterObjectAlign);

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_CHAR_VALUE_H_
