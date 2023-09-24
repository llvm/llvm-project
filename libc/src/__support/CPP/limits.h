//===-- A self contained equivalent of std::limits --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H

#include <limits.h>

namespace __llvm_libc {
namespace cpp {

// Some older gcc distributions don't define these for 32 bit targets.
#ifndef LLONG_MAX
constexpr size_t LLONG_BIT_WIDTH = sizeof(long long) * 8;
constexpr long long LLONG_MAX = ~0LL ^ (1LL << (LLONG_BIT_WIDTH - 1));
constexpr long long LLONG_MIN = 1LL << (LLONG_BIT_WIDTH - 1);
constexpr unsigned long long ULLONG_MAX = ~0ULL;
#endif

template <class T> class numeric_limits {
public:
  static constexpr T max();
  static constexpr T min();
};

// TODO: Add numeric_limits specializations as needed for new types.

template <> class numeric_limits<int> {
public:
  static constexpr int max() { return INT_MAX; }
  static constexpr int min() { return INT_MIN; }
};
template <> class numeric_limits<unsigned int> {
public:
  static constexpr unsigned int max() { return UINT_MAX; }
  static constexpr unsigned int min() { return 0; }
};
template <> class numeric_limits<long> {
public:
  static constexpr long max() { return LONG_MAX; }
  static constexpr long min() { return LONG_MIN; }
};
template <> class numeric_limits<unsigned long> {
public:
  static constexpr unsigned long max() { return ULONG_MAX; }
  static constexpr unsigned long min() { return 0; }
};
template <> class numeric_limits<long long> {
public:
  static constexpr long long max() { return LLONG_MAX; }
  static constexpr long long min() { return LLONG_MIN; }
};
template <> class numeric_limits<unsigned long long> {
public:
  static constexpr unsigned long long max() { return ULLONG_MAX; }
  static constexpr unsigned long long min() { return 0; }
};
template <> class numeric_limits<short> {
public:
  static constexpr short max() { return SHRT_MAX; }
  static constexpr short min() { return SHRT_MIN; }
};
template <> class numeric_limits<unsigned short> {
public:
  static constexpr unsigned short max() { return USHRT_MAX; }
  static constexpr unsigned short min() { return 0; }
};
template <> class numeric_limits<char> {
public:
  static constexpr char max() { return CHAR_MAX; }
  static constexpr char min() { return CHAR_MIN; }
};
template <> class numeric_limits<signed char> {
public:
  static constexpr signed char max() { return SCHAR_MAX; }
  static constexpr signed char min() { return SCHAR_MIN; }
};
template <> class numeric_limits<unsigned char> {
public:
  static constexpr unsigned char max() { return UCHAR_MAX; }
  static constexpr unsigned char min() { return 0; }
};
#ifdef __SIZEOF_INT128__
// On platform where UInt128 resolves to __uint128_t, this specialization
// provides the limits of UInt128.
template <> class numeric_limits<__uint128_t> {
public:
  static constexpr __uint128_t max() { return ~__uint128_t(0); }
  static constexpr __uint128_t min() { return 0; }
};
#endif

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H
