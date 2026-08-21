//===-- include/flang/Evaluate/character-value-impl.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_
#define FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_

#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/common.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace Fortran::evaluate::value {
using common::KindsEnum;

class CharacterValueImpl {
  using Storage =
      std::variant<std::monostate, std::string, std::u16string, std::u32string>;

public:
  // rule-of-five
  ~CharacterValueImpl() = default;
  CharacterValueImpl(const CharacterValueImpl &) = default;
  CharacterValueImpl(CharacterValueImpl &&) = default;
  CharacterValueImpl &operator=(const CharacterValueImpl &) = default;
  CharacterValueImpl &operator=(CharacterValueImpl &&) = default;

  CharacterValueImpl() = default;
  explicit CharacterValueImpl(KindsEnum kind, std::string s) {
    withCharProto(kind, [&](auto c) {
      using CharT = std::decay_t<decltype(c)>;
      using StringT = std::basic_string<CharT>;
      if (std::is_same_v<StringT, std::string>) {
        storage_ = std::move(s);
      } else {
        StringT buf;
        buf.resize(s.length());
        for (auto [i, c] : llvm::enumerate(s)) {
          buf[i] = c;
        }
        storage_ = std::move(buf);
      }
    });

    CHECK(this->kind() == kind);
  }

  explicit CharacterValueImpl(KindsEnum kind, std::u16string s)
      : storage_{std::move(s)} {
    CHECK(kind == KindsEnum::Kind2);
    CHECK(this->kind() == kind);
  }

  explicit CharacterValueImpl(KindsEnum kind, std::u32string s)
      : storage_{std::move(s)} {
    CHECK(kind == KindsEnum::Kind4);
    CHECK(this->kind() == kind);
  }

  /// Fill constructors: create a string of n copies of the given character.
  CharacterValueImpl(KindsEnum kind, std::size_t n, char32_t c);

  static CharacterValueImpl Zero(KindsEnum kind);

  static CharacterValueImpl FromRawBytes(
      KindsEnum kind, const void *raw, size_t byteSize);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  std::optional<llvm::StringRef> AsStringRef() const;

  /// Return the string as std::string if kind==1, or nullopt otherwise.
  std::optional<std::string> AsStdString() const;
  std::optional<std::u16string> AsU16String() const;
  std::optional<std::u32string> AsU32String() const;

  std::string ToStdString() const;

  bool IsMonostate() const { return storage_.index() == 0; }
  KindsEnum kind() const {
    return withCharProto(
        [](auto ct) { return static_cast<KindsEnum>(sizeof(ct)); });
  }

  /// Byte size of one character unit (1, 2, or 4).
  std::size_t charSize() const;

  /// Number of characters in this string.
  std::size_t size() const;

  /// String length (synonym for size()).
  std::size_t length() const { return size(); }

  /// True when the string is empty.
  bool empty() const { return size() == 0; }

  /// Raw byte pointer to the underlying character data.
  void *data() { return charData(); }
  const void *data() const { return charData(); }
  void *charData();
  const void *charData() const;

  // Comparison operators
  Ordering Compare(const CharacterValueImpl &y) const;
  bool operator<(const CharacterValueImpl &y) const;
  bool operator<=(const CharacterValueImpl &y) const { return !(y < *this); }
  bool operator==(const CharacterValueImpl &y) const;
  bool operator!=(const CharacterValueImpl &y) const { return !(*this == y); }
  bool operator>=(const CharacterValueImpl &y) const { return !(*this < y); }
  bool operator>(const CharacterValueImpl &y) const { return y < *this; }

  /// Assign n copies of the given character.
  void assign(KindsEnum kind, std::size_t n, char32_t c);

  /// Assign from a raw character pointer and length.
  void assign(const char *p, std::size_t n) { storage_ = std::string(p, n); }
  void assign(const char16_t *p, std::size_t n) {
    storage_ = std::u16string(p, n);
  }
  void assign(const char32_t *p, std::size_t n) {
    storage_ = std::u32string(p, n);
  }

  /// Erase from position pos to end.
  void erase(std::size_t pos);

  /// Append n copies of the given character.
  void append(std::size_t n, char32_t c);

  /// Replace the substring [pos, pos+len) with characters from other.
  CharacterValueImpl &replace(
      std::size_t pos, std::size_t len, const CharacterValueImpl &other);

  /// Return a suffix starting at pos.
  CharacterValueImpl substr(std::size_t pos) const;

  /// Return a substring of len characters starting at pos.
  CharacterValueImpl substr(std::size_t pos, std::size_t len) const;

  CharacterValueImpl ToAscii(KindsEnum kind) const;

  /// Reserve storage for at least n characters.
  void reserve(std::size_t n);

  /// Return the character at position i as char32_t (safe for all kinds).
  char32_t operator[](std::size_t i) const;

  /// Concatenate two same-kind strings.
  CharacterValueImpl operator+(const CharacterValueImpl &y) const;

  /// Append another same-kind string.
  CharacterValueImpl &operator+=(const CharacterValueImpl &y);

  /// Append a character, converting it to the string's element type.
  CharacterValueImpl &operator+=(char c);

  /// Sentinel value for "not found" positions (same as std::string::npos).
  static constexpr std::size_t npos{std::string::npos};

  // Find-family methods; return npos when not found.
  std::size_t find_first_not_of(char c) const {
    return find_first_not_of(static_cast<char32_t>(c));
  }
  std::size_t find_first_not_of(char16_t c) const {
    return find_first_not_of(static_cast<char32_t>(c));
  }
  std::size_t find_first_not_of(char32_t c) const;
  std::size_t find_last_not_of(char c) const {
    return find_last_not_of(static_cast<char32_t>(c));
  }
  std::size_t find_last_not_of(char16_t c) const {
    return find_last_not_of(static_cast<char32_t>(c));
  }
  std::size_t find_last_not_of(char32_t c) const;
  std::size_t find_first_not_of(const CharacterValueImpl &set) const;
  std::size_t find_last_not_of(const CharacterValueImpl &set) const;
  std::size_t find(const CharacterValueImpl &pattern) const;
  std::size_t rfind(const CharacterValueImpl &pattern) const;
  std::size_t find_first_of(const CharacterValueImpl &set) const;
  std::size_t find_last_of(const CharacterValueImpl &set) const;

  void StoreRawBytes(
      void *dst, std::size_t size, bool *changed = nullptr) const;

  // Compile-time dispatchers to current/specified kind

  template <typename F>
  auto withCharProto(F &&f) const
      -> decltype(std::declval<F>()(std::declval<char>())) {
    switch (storage_.index()) {
    case 1:
      return f(char{});
    case 2:
      return f(char16_t{});
    case 3:
      return f(char32_t{});
    default:
      llvm_unreachable("unsupported character kind/monostate");
    }
  }

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

  template <typename F>
  auto withStdString(F &&f) const
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

private:
  Storage storage_;
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::CharacterValueImpl &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_CHARACTER_VALUE_IMPL_H_
