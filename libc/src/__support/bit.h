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

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_BIT_H
