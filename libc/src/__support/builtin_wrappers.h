//===--Convenient template for builtins -------------------------*- C++ -*-===//
//             (Count Lead Zeroes) and (Count Trailing Zeros)
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_BUILTIN_WRAPPERS_H
#define LLVM_LIBC_SRC_SUPPORT_BUILTIN_WRAPPERS_H

#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {

// The following overloads are matched based on what is accepted by
// __builtin_clz/ctz* rather than using the exactly-sized aliases from stdint.h.
// This way, we can avoid making any assumptions about integer sizes and let the
// compiler match for us.
namespace __internal {

template <typename T> static inline int correct_zero(T val, int bits) {
  if (val == T(0))
    return sizeof(T(0)) * 8;
  else
    return bits;
}

template <typename T> static inline int clz(T val);
template <> inline int clz<unsigned int>(unsigned int val) {
  return __builtin_clz(val);
}
template <> inline int clz<unsigned long int>(unsigned long int val) {
  return __builtin_clzl(val);
}
template <> inline int clz<unsigned long long int>(unsigned long long int val) {
  return __builtin_clzll(val);
}

template <typename T> static inline int ctz(T val);
template <> inline int ctz<unsigned int>(unsigned int val) {
  return __builtin_ctz(val);
}
template <> inline int ctz<unsigned long int>(unsigned long int val) {
  return __builtin_ctzl(val);
}
template <> inline int ctz<unsigned long long int>(unsigned long long int val) {
  return __builtin_ctzll(val);
}
} // namespace __internal

template <typename T> static inline int safe_ctz(T val) {
  return __internal::correct_zero(val, __internal::ctz(val));
}

template <typename T> static inline int unsafe_ctz(T val) {
  return __internal::ctz(val);
}

template <typename T> static inline int safe_clz(T val) {
  return __internal::correct_zero(val, __internal::clz(val));
}

template <typename T> static inline int unsafe_clz(T val) {
  return __internal::clz(val);
}

// Add with carry
template <typename T>
inline constexpr cpp::enable_if_t<
    cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, T>
add_with_carry(T a, T b, T carry_in, T &carry_out) {
  T tmp = a + carry_in;
  T sum = b + tmp;
  carry_out = (sum < b) || (tmp < a);
  return sum;
}

#if __has_builtin(__builtin_addc)
// https://clang.llvm.org/docs/LanguageExtensions.html#multiprecision-arithmetic-builtins

template <>
inline unsigned char add_with_carry<unsigned char>(unsigned char a,
                                                   unsigned char b,
                                                   unsigned char carry_in,
                                                   unsigned char &carry_out) {
  return __builtin_addcb(a, b, carry_in, &carry_out);
}

template <>
inline unsigned short
add_with_carry<unsigned short>(unsigned short a, unsigned short b,
                               unsigned short carry_in,
                               unsigned short &carry_out) {
  return __builtin_addcs(a, b, carry_in, &carry_out);
}

template <>
inline unsigned int add_with_carry<unsigned int>(unsigned int a, unsigned int b,
                                                 unsigned int carry_in,
                                                 unsigned int &carry_out) {
  return __builtin_addc(a, b, carry_in, &carry_out);
}

template <>
inline unsigned long add_with_carry<unsigned long>(unsigned long a,
                                                   unsigned long b,
                                                   unsigned long carry_in,
                                                   unsigned long &carry_out) {
  return __builtin_addcl(a, b, carry_in, &carry_out);
}

template <>
inline unsigned long long
add_with_carry<unsigned long long>(unsigned long long a, unsigned long long b,
                                   unsigned long long carry_in,
                                   unsigned long long &carry_out) {
  return __builtin_addcll(a, b, carry_in, &carry_out);
}
#endif // __has_builtin(__builtin_addc)

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_BUILTIN_WRAPPERS_H
