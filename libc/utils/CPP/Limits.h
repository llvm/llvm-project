//===-- A self contained equivalent of std::limits --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_CPP_LIMITS_H
#define LLVM_LIBC_UTILS_CPP_LIMITS_H

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
  static constexpr long max() { return ULONG_MAX; }
  static constexpr long min() { return 0; }
};
template <> class NumericLimits<long long> {
public:
  static constexpr long long max() { return LLONG_MAX; }
  static constexpr long long min() { return LLONG_MIN; }
};
template <> class NumericLimits<unsigned long long> {
public:
  static constexpr long max() { return ULLONG_MAX; }
  static constexpr long min() { return 0; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_LIMITS_H
