//===-- User literal for unsigned integers ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This set of user defined literals allows uniform constructions of constants
// up to 256 bits and also help with unit tests (EXPECT_EQ requires the same
// type for LHS and RHS).
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_INTEGER_LITERALS_H
#define LLVM_LIBC_SRC___SUPPORT_INTEGER_LITERALS_H

#include "src/__support/CPP/limits.h"        // CHAR_BIT
#include "src/__support/UInt128.h"           // UInt128
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include <stddef.h>                          // size_t
#include <stdint.h>                          // uintxx_t

namespace LIBC_NAMESPACE {

LIBC_INLINE constexpr uint8_t operator""_u8(unsigned long long value) {
  return static_cast<uint8_t>(value);
}

LIBC_INLINE constexpr uint16_t operator""_u16(unsigned long long value) {
  return static_cast<uint16_t>(value);
}

LIBC_INLINE constexpr uint32_t operator""_u32(unsigned long long value) {
  return static_cast<uint32_t>(value);
}

LIBC_INLINE constexpr uint64_t operator""_u64(unsigned long long value) {
  return static_cast<uint64_t>(value);
}

namespace internal {

// Creates a T by reading digits from an array.
template <typename T>
LIBC_INLINE constexpr T accumulate(int base, const uint8_t *digits,
                                   size_t size) {
  T value{};
  for (; size; ++digits, --size) {
    value *= base;
    value += *digits;
  }
  return value;
}

// A static buffer to hold the digits for a T.
template <typename T, int base> struct DigitBuffer {
  static_assert(base == 2 || base == 10 || base == 16);
  // One character provides log2(base) bits.
  // Base 2 and 16 provide exactly one and four bits per character respectively.
  // For base 10, a character provides log2(10) ≈ 3.32... which we round to 3
  // for the purpose of buffer allocation.
  LIBC_INLINE_VAR static constexpr size_t BITS_PER_DIGIT = base == 2    ? 1
                                                           : base == 10 ? 3
                                                           : base == 16 ? 4
                                                                        : 0;
  LIBC_INLINE_VAR static constexpr size_t MAX_DIGITS =
      sizeof(T) * CHAR_BIT / BITS_PER_DIGIT;
  LIBC_INLINE_VAR static constexpr uint8_t INVALID_DIGIT = 255;

  uint8_t digits[MAX_DIGITS] = {};
  size_t size = 0;

  constexpr DigitBuffer(const char *str) {
    for (; *str != '\0'; ++str)
      push(*str);
  }

  // Returns the digit for a particular character.
  // Returns INVALID_DIGIT if the character is invalid.
  LIBC_INLINE static constexpr uint8_t get_digit_value(const char c) {
    const auto to_lower = [](char c) { return c | 32; };
    const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    const auto is_alpha = [](char c) {
      return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
    };
    if (is_digit(c))
      return static_cast<uint8_t>(c - '0');
    if (base > 10 && is_alpha(c))
      return static_cast<uint8_t>(to_lower(c) - 'a' + 10);
    return INVALID_DIGIT;
  }

  // Adds a single character to this buffer.
  LIBC_INLINE constexpr void push(char c) {
    if (c == '\'')
      return; // ' is valid but not taken into account.
    const uint8_t value = get_digit_value(c);
    if (value == INVALID_DIGIT || size >= MAX_DIGITS) {
      // During constant evaluation `__builtin_unreachable` will halt the
      // compiler as it is not executable. This is preferable over `assert` that
      // will only trigger in debug mode. Also we can't use `static_assert`
      // because `value` and `size` are not constant.
      __builtin_unreachable(); // invalid or too many characters.
    }
    digits[size] = value;
    ++size;
  }
};

// Generic implementation for native types (including __uint128_t or ExtInt
// where available).
template <typename T> struct Parser {
  template <int base> LIBC_INLINE static constexpr T parse(const char *str) {
    const DigitBuffer<T, base> buffer(str);
    return accumulate<T>(base, buffer.digits, buffer.size);
  }
};

// Specialization for UInt<N>.
// Because this code runs at compile time we try to make it efficient. For
// binary and hexadecimal formats we read digits by chunks of 64 bits and
// produce the BigInt internal representation direcly. For decimal numbers we
// go the slow path and use slower BigInt arithmetic.
template <size_t N> struct Parser<LIBC_NAMESPACE::UInt<N>> {
  using UIntT = UInt<N>;
  template <int base> static constexpr UIntT parse(const char *str) {
    const DigitBuffer<UIntT, base> buffer(str);
    if constexpr (base == 10) {
      // Slow path, we sum and multiply BigInt for each digit.
      return accumulate<UIntT>(base, buffer.digits, buffer.size);
    } else {
      // Fast path, we consume blocks of WordType and creates the BigInt's
      // internal representation directly.
      using WordArrayT = decltype(UIntT::val);
      using WordType = typename WordArrayT::value_type;
      WordArrayT array = {};
      size_t size = buffer.size;
      const uint8_t *digit_ptr = buffer.digits + size;
      for (size_t i = 0; i < array.size(); ++i) {
        constexpr size_t DIGITS = DigitBuffer<WordType, base>::MAX_DIGITS;
        const size_t chunk = size > DIGITS ? DIGITS : size;
        digit_ptr -= chunk;
        size -= chunk;
        array[i] = accumulate<WordType>(base, digit_ptr, chunk);
      }
      return UIntT(array);
    }
  }
};

// Detects the base of the number and dispatches to the right implementation.
template <typename T>
LIBC_INLINE constexpr T parse_with_prefix(const char *ptr) {
  using P = Parser<T>;
  if (ptr[0] == '0' && ptr[1] == 'x')
    return P::template parse<16>(ptr + 2);
  else if (ptr[0] == '0' && ptr[1] == 'b')
    return P::template parse<2>(ptr + 2);
  else
    return P::template parse<10>(ptr);
}

} // namespace internal

LIBC_INLINE constexpr UInt128 operator""_u128(const char *x) {
  return internal::parse_with_prefix<UInt128>(x);
}

LIBC_INLINE constexpr auto operator""_u256(const char *x) {
  return internal::parse_with_prefix<UInt<256>>(x);
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_INTEGER_LITERALS_H
