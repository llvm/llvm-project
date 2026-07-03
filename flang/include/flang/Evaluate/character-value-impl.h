//===-- include/flang/Evaluate/character-value-impl.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_
#define FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_

#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace Fortran::evaluate::value {

// CharacterValueImpl: runtime-kind character string (kind 1/2/4).
// Interface compatible to std::string
class CharacterValueImpl {
public:
  CharacterValueImpl() = default;
  explicit CharacterValueImpl(std::string s) : storage_{std::move(s)} {}
  explicit CharacterValueImpl(std::u16string s) : storage_{std::move(s)} {}
  explicit CharacterValueImpl(std::u32string s) : storage_{std::move(s)} {}

  // Fill constructors: create a string of n copies of the given character.
  CharacterValueImpl(std::size_t n, char c) : storage_{std::string(n, c)} {}
  CharacterValueImpl(std::size_t n, char16_t c)
      : storage_{std::u16string(n, c)} {}
  CharacterValueImpl(std::size_t n, char32_t c)
      : storage_{std::u32string(n, c)} {}

  // Comparison operators
  bool operator==(const CharacterValueImpl &y) const {
    return storage_ == y.storage_;
  }
  bool operator!=(const CharacterValueImpl &y) const { return !(*this == y); }
  bool operator<(const CharacterValueImpl &y) const;
  bool operator>(const CharacterValueImpl &y) const { return y < *this; }
  bool operator<=(const CharacterValueImpl &y) const { return !(y < *this); }
  bool operator>=(const CharacterValueImpl &y) const { return !(*this < y); }

  int kind() const;
  int bits() const;
  bool IsMonostate() const { return storage_.index() == 0; }
  bool IsZero() const;
  bool StoreRawBytes(void *to, std::size_t bytes) const;
  static CharacterValueImpl Zero(int kind);

  // Dispatch a callable over the active string alternative, mirroring
  // RealValueImpl::WithReal.  Precondition: not monostate.
  template <typename F>
  auto WithChar(F &&f) const
      -> decltype(std::declval<F>()(std::declval<const std::string &>())) {
    switch (storage_.index()) {
    case 1:
      return f(std::get<std::string>(storage_));
    case 2:
      return f(std::get<std::u16string>(storage_));
    case 3:
      return f(std::get<std::u32string>(storage_));
    default:
      llvm_unreachable("operation on uninitialized CharacterValue");
    }
  }

  std::size_t size() const;

  // String length (synonym for size()).
  std::size_t length() const { return size(); }

  // True when the string is empty.
  bool empty() const { return size() == 0; }

  // Assign n copies of the given character, fixing the kind from the char type.
  void assign(std::size_t n, char c) { storage_ = std::string(n, c); }
  void assign(std::size_t n, char16_t c) { storage_ = std::u16string(n, c); }
  void assign(std::size_t n, char32_t c) { storage_ = std::u32string(n, c); }

  // Assign from a raw character pointer and length.
  void assign(const char *p, std::size_t n) { storage_ = std::string(p, n); }
  void assign(const char16_t *p, std::size_t n) {
    storage_ = std::u16string(p, n);
  }
  void assign(const char32_t *p, std::size_t n) {
    storage_ = std::u32string(p, n);
  }

  // Erase from position pos to end.
  void erase(std::size_t pos);

  // Append n copies of the given character (widened to the stored type).
  void append(std::size_t n, char c);
  void append(std::size_t n, char16_t c);
  void append(std::size_t n, char32_t c);

  // Replace the substring [pos, pos+len) with characters from other.
  CharacterValueImpl &replace(
      std::size_t pos, std::size_t len, const CharacterValueImpl &other);

  // Return a suffix starting at pos.
  CharacterValueImpl substr(std::size_t pos) const;

  // Return a substring of len characters starting at pos.
  CharacterValueImpl substr(std::size_t pos, std::size_t len) const;

  // Return the string as std::string if kind==1, or nullopt otherwise.
  std::optional<std::string> ToStdString() const;

  // Reserve storage for at least n characters.
  void reserve(std::size_t n);

  // Return the character at position i as char32_t (safe for all kinds).
  char32_t operator[](std::size_t i) const;

  // Concatenate two same-kind strings.
  CharacterValueImpl operator+(const CharacterValueImpl &y) const;

  // Append another same-kind string.
  CharacterValueImpl &operator+=(const CharacterValueImpl &y);

  // Append a character, converting it to the string's element type.
  CharacterValueImpl &operator+=(char c);

  // Sentinel value for "not found" positions (same as std::string::npos).
  static constexpr std::size_t npos{std::string::npos};

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

  // Find-family methods using a CharacterValueImpl as the character set.
  std::size_t find_first_not_of(const CharacterValueImpl &set) const;
  std::size_t find_last_not_of(const CharacterValueImpl &set) const;

  // Find-family methods taking a CharacterValueImpl set/pattern.
  std::size_t find(const CharacterValueImpl &pattern) const;
  std::size_t rfind(const CharacterValueImpl &pattern) const;
  std::size_t find_first_of(const CharacterValueImpl &set) const;
  std::size_t find_last_of(const CharacterValueImpl &set) const;

  // Byte size of one character unit (1, 2, or 4).
  std::size_t charSize() const;

  // Raw byte pointer to the underlying character data.
  const void *data() const { return charData(); }
  void *charData();
  const void *charData() const;

private:
  std::size_t find_first_not_of_char(char32_t c) const;
  std::size_t find_last_not_of_char(char32_t c) const;

  using Storage =
      std::variant<std::monostate, std::string, std::u16string, std::u32string>;
  Storage storage_;
};

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_
