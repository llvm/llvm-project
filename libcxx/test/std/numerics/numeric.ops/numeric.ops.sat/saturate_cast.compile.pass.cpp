//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class R, class T>
//   constexpr R saturate_cast(T x) noexcept;                    // freestanding

#include <concepts>
#include <numeric>

#include "test_macros.h"

template <typename R, typename T>
concept CanDo = requires(T x) {
  { std::saturate_cast<R>(x) } -> std::same_as<R>;
};

template <typename R, typename T>
constexpr void test_constraint_success() {
  static_assert(CanDo<R, T>);
  static_assert(CanDo<T, T>);
  static_assert(CanDo<T, R>);
}

template <typename T>
constexpr void test_constraint_fail() {
  using I = int;
  using R = T;
  static_assert(!CanDo<R, T>);
  static_assert(!CanDo<T, R>);
  static_assert(!CanDo<I, T>);
  static_assert(!CanDo<T, I>);
}

constexpr void test() {
  // Contraint success - Signed
  using SI = long long int;
  test_constraint_success<SI, signed char>();
  test_constraint_success<SI, short int>();
  test_constraint_success<SI, signed char>();
  test_constraint_success<SI, short int>();
  test_constraint_success<SI, int>();
  test_constraint_success<SI, long int>();
  test_constraint_success<int, long long int>();
#ifndef TEST_HAS_NO_INT128
  test_constraint_success<__int128_t, SI>();
#endif
  // Contraint success - Unsigned
  using UI = unsigned long long int;
  test_constraint_success<UI, unsigned char>();
  test_constraint_success<UI, unsigned short int>();
  test_constraint_success<UI, unsigned int>();
  test_constraint_success<UI, unsigned long int>();
  test_constraint_success<unsigned int, unsigned long long int>();
#ifndef TEST_HAS_NO_INT128
  test_constraint_success<UI, __uint128_t>();
#endif

  // Contraint failure
  test_constraint_fail<bool>();
  test_constraint_fail<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_constraint_fail<wchar_t>();
#endif
  test_constraint_fail<char8_t>();
  test_constraint_fail<char16_t>();
  test_constraint_fail<char32_t>();
  test_constraint_fail<float>();
  test_constraint_fail<double>();
  test_constraint_fail<long double>();
}
