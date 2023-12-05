//===-- Mimics llvm/ADT/Bit.h -----------------------------------*- C++ -*-===//
// Provides useful bit functions.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BIT_H
#define LLVM_LIBC_SRC___SUPPORT_BIT_H

#include "src/__support/CPP/type_traits.h"   // make_unsigned
#include "src/__support/macros/attributes.h" // LIBC_INLINE

namespace LIBC_NAMESPACE {

// The following overloads are matched based on what is accepted by
// __builtin_clz/ctz* rather than using the exactly-sized aliases from stdint.h.
// This way, we can avoid making any assumptions about integer sizes and let the
// compiler match for us.
namespace __internal {

template <typename T> LIBC_INLINE int constexpr correct_zero(T val, int bits) {
  if (val == T(0))
    return sizeof(T(0)) * 8;
  else
    return bits;
}

template <typename T> LIBC_INLINE constexpr int clz(T val);
template <> LIBC_INLINE int clz<unsigned char>(unsigned char val) {
  return __builtin_clz(static_cast<unsigned int>(val)) -
         8 * static_cast<int>(sizeof(unsigned int) - sizeof(unsigned char));
}
template <> LIBC_INLINE int clz<unsigned short>(unsigned short val) {
  return __builtin_clz(static_cast<unsigned int>(val)) -
         8 * static_cast<int>(sizeof(unsigned int) - sizeof(unsigned short));
}
template <> LIBC_INLINE int clz<unsigned int>(unsigned int val) {
  return __builtin_clz(val);
}
template <>
LIBC_INLINE constexpr int clz<unsigned long int>(unsigned long int val) {
  return __builtin_clzl(val);
}
template <>
LIBC_INLINE constexpr int
clz<unsigned long long int>(unsigned long long int val) {
  return __builtin_clzll(val);
}

template <typename T> LIBC_INLINE constexpr int ctz(T val);
template <> LIBC_INLINE int ctz<unsigned char>(unsigned char val) {
  return __builtin_ctz(static_cast<unsigned int>(val));
}
template <> LIBC_INLINE int ctz<unsigned short>(unsigned short val) {
  return __builtin_ctz(static_cast<unsigned int>(val));
}
template <> LIBC_INLINE int ctz<unsigned int>(unsigned int val) {
  return __builtin_ctz(val);
}
template <>
LIBC_INLINE constexpr int ctz<unsigned long int>(unsigned long int val) {
  return __builtin_ctzl(val);
}
template <>
LIBC_INLINE constexpr int
ctz<unsigned long long int>(unsigned long long int val) {
  return __builtin_ctzll(val);
}
} // namespace __internal

template <typename T> LIBC_INLINE constexpr int safe_ctz(T val) {
  return __internal::correct_zero(val, __internal::ctz(val));
}

template <typename T> LIBC_INLINE constexpr int unsafe_ctz(T val) {
  return __internal::ctz(val);
}

template <typename T> LIBC_INLINE constexpr int safe_clz(T val) {
  return __internal::correct_zero(val, __internal::clz(val));
}

template <typename T> LIBC_INLINE constexpr int unsafe_clz(T val) {
  return __internal::clz(val);
}

template <typename T> LIBC_INLINE constexpr T next_power_of_two(T val) {
  if (val == 0)
    return 1;
  T idx = safe_clz(val - 1);
  return static_cast<T>(1) << ((8ull * sizeof(T)) - idx);
}

template <typename T> LIBC_INLINE constexpr bool is_power_of_two(T val) {
  return val != 0 && (val & (val - 1)) == 0;
}

template <typename T> LIBC_INLINE constexpr T offset_to(T val, T align) {
  return (-val) & (align - 1);
}

template <typename T> LIBC_INLINE constexpr T rotate_left(T val, T amount) {
  // Implementation taken from "Safe, Efficient, and Portable Rotate in C/C++"
  // https://blog.regehr.org/archives/1063
  // Using the safe version as the rotation pattern is now recognized by both
  // GCC and Clang.
  using U = cpp::make_unsigned_t<T>;
  U v = static_cast<U>(val);
  U a = static_cast<U>(amount);
  return (v << a) | (v >> ((-a) & (sizeof(U) * 8 - 1)));
}
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_BIT_H
