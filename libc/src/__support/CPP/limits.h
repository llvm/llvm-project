//===-- A self contained equivalent of std::limits --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE

#include <limits.h> // CHAR_BIT

namespace LIBC_NAMESPACE {
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
  LIBC_INLINE static constexpr T max();
  LIBC_INLINE static constexpr T min();
};

// TODO: Add numeric_limits specializations as needed for new types.

template <> class numeric_limits<int> {
public:
  LIBC_INLINE static constexpr int max() { return INT_MAX; }
  LIBC_INLINE static constexpr int min() { return INT_MIN; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT * sizeof(int) - 1;
};
template <> class numeric_limits<unsigned int> {
public:
  LIBC_INLINE static constexpr unsigned int max() { return UINT_MAX; }
  LIBC_INLINE static constexpr unsigned int min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT * sizeof(unsigned int);
};
template <> class numeric_limits<long> {
public:
  LIBC_INLINE static constexpr long max() { return LONG_MAX; }
  LIBC_INLINE static constexpr long min() { return LONG_MIN; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT * sizeof(long) - 1;
};
template <> class numeric_limits<unsigned long> {
public:
  LIBC_INLINE static constexpr unsigned long max() { return ULONG_MAX; }
  LIBC_INLINE static constexpr unsigned long min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits =
      CHAR_BIT * sizeof(unsigned long);
};
template <> class numeric_limits<long long> {
public:
  LIBC_INLINE static constexpr long long max() { return LLONG_MAX; }
  LIBC_INLINE static constexpr long long min() { return LLONG_MIN; }
  LIBC_INLINE_VAR static constexpr int digits =
      CHAR_BIT * sizeof(long long) - 1;
};
template <> class numeric_limits<unsigned long long> {
public:
  LIBC_INLINE static constexpr unsigned long long max() { return ULLONG_MAX; }
  LIBC_INLINE static constexpr unsigned long long min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits =
      CHAR_BIT * sizeof(unsigned long long);
};
template <> class numeric_limits<short> {
public:
  LIBC_INLINE static constexpr short max() { return SHRT_MAX; }
  LIBC_INLINE static constexpr short min() { return SHRT_MIN; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT * sizeof(short) - 1;
};
template <> class numeric_limits<unsigned short> {
public:
  LIBC_INLINE static constexpr unsigned short max() { return USHRT_MAX; }
  LIBC_INLINE static constexpr unsigned short min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits =
      CHAR_BIT * sizeof(unsigned short);
};
template <> class numeric_limits<char> {
public:
  LIBC_INLINE static constexpr char max() { return CHAR_MAX; }
  LIBC_INLINE static constexpr char min() { return CHAR_MIN; }
};
template <> class numeric_limits<signed char> {
public:
  LIBC_INLINE static constexpr signed char max() { return SCHAR_MAX; }
  LIBC_INLINE static constexpr signed char min() { return SCHAR_MIN; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT - 1;
};
template <> class numeric_limits<unsigned char> {
public:
  LIBC_INLINE static constexpr unsigned char max() { return UCHAR_MAX; }
  LIBC_INLINE static constexpr unsigned char min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits = CHAR_BIT;
};
#ifdef __SIZEOF_INT128__
// On platform where UInt128 resolves to __uint128_t, this specialization
// provides the limits of UInt128.
template <> class numeric_limits<__uint128_t> {
public:
  LIBC_INLINE static constexpr __uint128_t max() { return ~__uint128_t(0); }
  LIBC_INLINE static constexpr __uint128_t min() { return 0; }
  LIBC_INLINE_VAR static constexpr int digits =
      CHAR_BIT * sizeof(__uint128_t) - 1;
};
#endif

} // namespace cpp
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_LIMITS_H
