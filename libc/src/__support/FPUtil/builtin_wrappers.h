//===--Convenient template for builtins -------------------------*- C++ -*-===//
//             (Count Lead Zeroes) and (Count Trailing Zeros)
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_BUILTIN_WRAPPERS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_BUILTIN_WRAPPERS_H

namespace __llvm_libc {
namespace fputil {

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

template <typename T> static inline bool isnan(T val) {
  return __builtin_isnan(val);
}

template <typename T> static inline bool isinf(T val) {
  return __builtin_isinf(val);
}

template <typename T> static inline bool isfinite(T val) {
  return __builtin_isfinite(val);
}

inline float quiet_NaN(float) { return __builtin_nanf(""); }
inline double quiet_NaN(double) { return __builtin_nan(""); }
inline long double quiet_NaN(long double) { return __builtin_nanl(""); }

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_BUILTIN_WRAPPERS_H
