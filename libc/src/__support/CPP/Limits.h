//===-- A self contained equivalent of std::limits --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_LIMITS_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_LIMITS_H

#include "UInt.h"

#include <limits.h>

namespace __llvm_libc {
namespace cpp {

template <class T> class NumericLimits {
public:
  static constexpr T max();
  static constexpr T min();
};

// TODO: Add NumericLimits specializations as needed for new types.

template <> class NumericLimits<int> {
public:
  static constexpr int max() { return INT_MAX; }
  static constexpr int min() { return INT_MIN; }
};
template <> class NumericLimits<unsigned int> {
public:
  static constexpr unsigned int max() { return UINT_MAX; }
  static constexpr unsigned int min() { return 0; }
};
template <> class NumericLimits<long> {
public:
  static constexpr long max() { return LONG_MAX; }
  static constexpr long min() { return LONG_MIN; }
};
template <> class NumericLimits<unsigned long> {
public:
  static constexpr unsigned long max() { return ULONG_MAX; }
  static constexpr unsigned long min() { return 0; }
};
template <> class NumericLimits<long long> {
public:
  static constexpr long long max() { return LLONG_MAX; }
  static constexpr long long min() { return LLONG_MIN; }
};
template <> class NumericLimits<unsigned long long> {
public:
  static constexpr unsigned long long max() { return ULLONG_MAX; }
  static constexpr unsigned long long min() { return 0; }
};
template <> class NumericLimits<short> {
public:
  static constexpr short max() { return SHRT_MAX; }
  static constexpr short min() { return SHRT_MIN; }
};
template <> class NumericLimits<unsigned short> {
public:
  static constexpr unsigned short max() { return USHRT_MAX; }
  static constexpr unsigned short min() { return 0; }
};
template <> class NumericLimits<char> {
public:
  static constexpr char max() { return CHAR_MAX; }
  static constexpr char min() { return CHAR_MIN; }
};
template <> class NumericLimits<unsigned char> {
public:
  static constexpr unsigned char max() { return UCHAR_MAX; }
  static constexpr unsigned char min() { return 0; }
};
// This specialization enables two things:
// 1. On platforms where UInt128 resolves to UInt<128>, this specialization
//    provides limits of UInt128.
// 2. On platforms where UInt128 resolves to __uint128_t, this specialization
//    allows us to unittest UInt<128>.
template <> class NumericLimits<UInt<128>> {
public:
  static constexpr UInt<128> max() { return ~UInt<128>(0); }
  static constexpr UInt<128> min() { return 0; }
};
#ifdef __SIZEOF_INT128__
// On platform where UInt128 resolves to __uint128_t, this specialization
// provides the limits of UInt128.
template <> class NumericLimits<__uint128_t> {
public:
  static constexpr __uint128_t max() { return ~__uint128_t(0); }
  static constexpr __uint128_t min() { return 0; }
};
#endif

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_LIMITS_H
