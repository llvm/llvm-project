//===-- Internal utils for wctype conversion code - common ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H

#include "hdr/stdint_proxy.h"
#include "slice.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {

namespace internal_wctype_conversion_utils {

// Multiplies two 64-bit unsigned integers and returns the high 64 bits
LIBC_INLINE constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
  return (static_cast<UInt128>(a) * static_cast<UInt128>(b)) >> 64;
}

// Computes the ceiling of the division of a by b
template <typename T> LIBC_INLINE static constexpr T div_ceil(T a, T b) {
  // works for positive or negative, matches "round toward +infinity"
  LIBC_ASSERT(b != 0);

  T quotient = a / b;
  T remainder = a % b;

  // If there is a remainder AND the division is not already upward
  if (remainder != 0 && ((a > 0) == (b > 0))) {
    quotient += 1;
  }

  return quotient;
}

// Checks if a number is a power of two
template <typename T>
LIBC_INLINE static constexpr bool is_power_of_two(T number) {
  static_assert(cpp::is_unsigned_v<T>,
                "is_power_of_two requires unsigned type");
  return number != 0 && (number & (number - 1)) == 0;
}

// Checks if a signed number is a power of two
template <typename T>
LIBC_INLINE static constexpr bool is_power_of_two_signed(T number) {
  static_assert(cpp::is_signed_v<T>,
                "is_power_of_two_signed requires signed type");
  return number > 0 && (number & (number - 1)) == 0;
}

// Maps a function over an array and returns the resulting array
template <typename T, typename Fn, size_t N>
LIBC_INLINE static constexpr auto map(const cpp::array<T, N> &container,
                                      Fn func) {
  using R = cpp::invoke_result_t<Fn, T>;
  cpp::array<R, N> out{};

  for (size_t i = 0; i < N; ++i) {
    out[i] = func(container[i]);
  }

  return out;
}

// Applies func to each element of an iterable, returning false on early exit
template <typename Iterator, typename Fn>
LIBC_INLINE static constexpr auto try_for_each(Iterator &&iter, Fn &&func) {
  for (auto &&x : iter) {
    if (!func(x)) {
      return false; // early exit
    }
  }

  return true;
}

// Sums up all elements in a container. Alternative to accumulate.
template <typename T> LIBC_INLINE static constexpr T sum(Slice<T> container) {
  size_t acc = 0;

  for (T const item : container) {
    acc += item;
  }

  return acc;
}

// Wrapping addition for integral types
template <typename T> LIBC_INLINE static constexpr T wrapping_add(T a, T b) {
  static_assert(cpp::is_integral_v<T>, "wrapping_add requires integral type");

  while (b != 0) {
    T carry = a & b;
    a = a ^ b;
    b = carry << 1;
  }
  return a;
}

// Wrapping multiplication for integral types
template <typename T> LIBC_INLINE static constexpr T wrapping_mul(T a, T b) {
  static_assert(cpp::is_integral_v<T>, "wrapping_mul requires integral type");

  T result = 0;

  while (b != 0) {
    if (b & 1) {
      result = result + a;
    }
    a = a << 1;
    b = static_cast<cpp::make_unsigned_t<T>>(b) >> 1;
  }

  return result;
}

// Counts the number of zero elements in an array
template <typename T, size_t N>
LIBC_INLINE static constexpr auto count_zeros(cpp::array<T, N> &container) {
  size_t counter = 0;

  for (auto element : container) {
    if (!element) {
      counter++;
    }
  }

  return counter;
}

// Rotates bits to the right for unsigned integral types
template <typename T>
LIBC_INLINE static constexpr T rotate_right(T number, size_t rotation) {
  static_assert(cpp::is_unsigned_v<T>, "rotate_right requires unsigned type");

  constexpr size_t BITS = cpp::numeric_limits<T>::digits;
  rotation %= BITS;
  return (number >> rotation) | (number << (BITS - rotation));
}

// Converts a 32-bit unsigned integer to an array of 4 little-endian bytes
LIBC_INLINE static constexpr cpp::array<uint8_t, 4>
to_le_bytes(uint32_t number) {
  return {
      static_cast<uint8_t>(number),
      static_cast<uint8_t>(number >> 8),
      static_cast<uint8_t>(number >> 16),
      static_cast<uint8_t>(number >> 24),
  };
}

// Bit-casts a pointer of one type to another type. This is different from the
// cpp::bit_cast in that it copies the bytes manually to local variable.
template <typename To, typename From>
LIBC_INLINE static constexpr To ptr_bit_cast(From *from) {
  To to{};
  char *dst = reinterpret_cast<char *>(&to);
  const char *src = reinterpret_cast<const char *>(from);
  for (unsigned i = 0; i < sizeof(To); ++i)
    dst[i] = src[i];
  return to;
}

// Sorts an array using merge sort algorithm
template <typename T, size_t N>
LIBC_INLINE static constexpr auto array_sort(cpp::array<T, N> &arr) {
  if constexpr (N <= 1)
    return arr; // base case

  constexpr size_t MID = N / 2;

  // Split array into left and right halves
  cpp::array<T, MID> left{};
  cpp::array<T, N - MID> right{};

  for (size_t i = 0; i < MID; ++i)
    left[i] = arr[i];
  for (size_t i = MID; i < N; ++i)
    right[i - MID] = arr[i];

  // Recursively sort each half
  left = array_sort(left);
  right = array_sort(right);

  // Merge halves
  cpp::array<T, N> result{};
  size_t li = 0, ri = 0, ki = 0;

  while (li < MID && ri < N - MID) {
    result[ki++] = (left[li] <= right[ri]) ? left[li++] : right[ri++];
  }
  while (li < MID)
    result[ki++] = left[li++];
  while (ri < N - MID)
    result[ki++] = right[ri++];

  return result;
}

} // namespace internal_wctype_conversion_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_UTILS_H
