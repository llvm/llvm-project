//===-- include/flang/Evaluate/character-value.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHARACTER_VALUE_H_
#define FORTRAN_EVALUATE_CHARACTER_VALUE_H_

#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/object-sizes.h"
#include "flang/Evaluate/type.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>

namespace Fortran::evaluate::value {
class CharacterValueImpl;
using common::KindsEnum;

/// A character string with dynamic character representation with
/// std::basic_string-like API.
///
/// The character type is dynamic between char, char16_t, and char32_t. As being
/// able to represent all values, char32_t is used when passing single
/// characters. It is also kind-aware, i.e. knows which CHARACTER kind it
/// currently represents.
///
/// The implementation is hidden from this header using a pImpl-like idiom.
class CharacterValue {
public:
  // rule-of-five
  ~CharacterValue();
  CharacterValue(const CharacterValue &);
  CharacterValue(CharacterValue &&);
  CharacterValue &operator=(const CharacterValue &);
  CharacterValue &operator=(CharacterValue &&);

  // ctors

  /// A default-initialized CharacterValue is in a so-called "monostate"; it
  /// represents an empty string, but its kind is not yet known. Not all
  /// operations are supported in this state.
  CharacterValue();

  explicit CharacterValue(KindsEnum kind, std::string s);
  explicit CharacterValue(KindsEnum kind, std::u16string s);
  explicit CharacterValue(KindsEnum kind, std::u32string s);

  /// Fill constructor: create a string of n copies of the given character.
  CharacterValue(KindsEnum kind, std::size_t n, char32_t c);

  // Named ctors
  static CharacterValue Zero(KindsEnum kind);

  static CharacterValue FromRawBytes(
      KindsEnum kind, const void *raw, size_t byteSize);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Whether this object represents a default-initialized value (zero) of
  /// not-yet-known kind.
  bool IsMonostate() const;

  /// The kind of the value currently stored.
  KindsEnum kind() const;

  bool empty() const;
  std::size_t size() const;
  std::size_t length() const { return size(); }

  /// Byte size of one character unit (1, 2, or 4).
  std::size_t charSize() const { return static_cast<std::size_t>(kind()); }

  /// Number of bytes accessed by FromRawBytes/StoreRawBytes
  size_t bytesStored() const { return length() * charSize(); }

  // Casting to other representations
  std::optional<llvm::StringRef> AsStringRef() const;
  std::optional<std::string> AsStdString() const {
    if (auto str{AsStringRef()}) {
      return str->str();
    }
    return std::nullopt;
  }
  std::optional<std::u16string> AsU16String() const;
  std::optional<std::u32string> AsU32String() const;

  /// Force conversion to Ascii even if this means loss of information
  std::string ToStdString() const;

  template <typename CharT, typename = std::void_t<std::basic_string<CharT>>>
  std::optional<std::basic_string<CharT>> AsBasicString() const {
    if constexpr (std::is_same_v<char, CharT>) {
      return AsStdString();
    } else if constexpr (std::is_same_v<char16_t, CharT>) {
      return AsU16String();
    } else if constexpr (std::is_same_v<char32_t, CharT>) {
      return AsU32String();
    } else {
      static_assert(false, "Must be one of the supported character types");
    }
  }

  // Comparisons
  Ordering Compare(const CharacterValue &y) const;
  bool operator<(const CharacterValue &y) const;
  bool operator<=(const CharacterValue &y) const { return !(y < *this); }
  bool operator==(const CharacterValue &y) const;
  bool operator!=(const CharacterValue &y) const { return !(*this == y); }
  bool operator>=(const CharacterValue &y) const { return !(*this < y); }
  bool operator>(const CharacterValue &y) const { return y < *this; }

  CharacterValue ToAscii(KindsEnum kind) const;

  /// Assign n copies of the given character, fixing the kind from the char
  /// type.
  void assign(KindsEnum kind, std::size_t n, char32_t c);

  /// Assign from a raw character pointer and length.
  void assign(const char *p, std::size_t n);
  void assign(const char16_t *p, std::size_t n);
  void assign(const char32_t *p, std::size_t n);

  /// Erase from position pos to end.
  void erase(std::size_t pos);

  /// Append n copies of the given character (widened to the stored type).
  void append(std::size_t n, char32_t c);

  /// Replace the substring [pos, pos+len) with characters from other.
  CharacterValue &replace(
      std::size_t pos, std::size_t len, const CharacterValue &other);

  /// Return a suffix starting at pos.
  CharacterValue substr(std::size_t pos) const;

  /// Return a substring of len characters starting at pos.
  CharacterValue substr(std::size_t pos, std::size_t len) const;

  /// Reserve storage for at least n characters.
  void reserve(std::size_t n);

  /// Return the character at position i
  char32_t operator[](std::size_t i) const;

  /// Concatenate two same-kind strings.
  CharacterValue operator+(const CharacterValue &y) const;

  /// Append another same-kind string.
  CharacterValue &operator+=(const CharacterValue &y);

  /// Append a character, converting it to the string's element type.
  CharacterValue &operator+=(char c);

  /// Sentinel value for "not found" positions (same as std::string::npos).
  static constexpr std::size_t npos{std::string::npos};

  // Find-family methods; return npos when not found.
  std::size_t find(const CharacterValue &pattern) const;
  std::size_t rfind(const CharacterValue &pattern) const;
  std::size_t find_first_of(const CharacterValue &set) const;
  std::size_t find_last_of(const CharacterValue &set) const;
  std::size_t find_first_not_of(char32_t c) const;
  std::size_t find_last_not_of(char32_t c) const;
  std::size_t find_first_not_of(const CharacterValue &set) const;
  std::size_t find_last_not_of(const CharacterValue &set) const;

  /// Raw byte pointer to the underlying character data
  void *data();
  const void *data() const;

  /// Like data(), but pre-casted to char
  char *charData() { return static_cast<char *>(data()); }
  const char *charData() const { return static_cast<const char *>(data()); }

  void *at(size_t pos) { return &charData()[pos * charSize()]; }
  const void *at(size_t pos) const { return &charData()[pos * charSize()]; }

  /// Writes a string of characters to \p dst. \o is the the number of bytes to
  /// be written; must be a multiple of the size of a single character.  If \p s
  /// is smaller that \p size, the rest of the memory is set to spaces. If \p s
  /// is shorter than size, only the first characters are written.
  /// If \p changes points to bool, it will be set to true if any bytes at \p
  /// dst have changed.
  void StoreRawBytes(void *dst, size_t size, bool *changed = nullptr) const;

  template <typename F>
  static auto withCharProto(KindsEnum kind, F &&f)
      -> decltype(std::declval<F>()(std::declval<char>())) {
    switch (kind) {
    case KindsEnum::Kind1:
      return f(char{});
    case KindsEnum::Kind2:
      return f(char16_t{});
    case KindsEnum::Kind4:
      return f(char32_t{});
    default:
      llvm_unreachable("unsupported character kind/monostate");
    }
  }

  template <typename F> decltype(auto) withStdString(F &&f) const {
    switch (kind()) {
    case KindsEnum::Kind1:
      return f(*AsStdString());
    case KindsEnum::Kind2:
      return f(*AsU16String());
    case KindsEnum::Kind4:
      return f(*AsU32String());
    default:
      llvm_unreachable("unsupported kind/monostate");
    }
  }

private:
  static CharacterValue FromImpl(const CharacterValueImpl &y);
  static CharacterValue FromImpl(CharacterValueImpl &&y);

  CharacterValueImpl &impl() {
    return *reinterpret_cast<CharacterValueImpl *>(this);
  }
  const CharacterValueImpl &impl() const {
    return *reinterpret_cast<const CharacterValueImpl *>(this);
  }

  [[maybe_unused]] alignas(
      detail::kCharacterObjectAlign) char opaque_[detail::kCharacterObjectSize];
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::CharacterValue &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_CHARACTER_VALUE_H_
