//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class T>
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <concepts>
#include <numeric>

#include "test_macros.h"

template <typename T, typename U>
concept CanDo = requires(T x, U y) {
  { std::sub_sat(x, y) } -> std::same_as<T>;
};

template <typename T, typename U>
constexpr void test_constraint_success() {
  static_assert(CanDo<T, T>);
  static_assert(!CanDo<U, T>);
  static_assert(!CanDo<T, U>);
}

template <typename T>
constexpr void test_constraint_fail() {
  using I = int;
  static_assert(!CanDo<T, T>);
  static_assert(!CanDo<I, T>);
  static_assert(!CanDo<T, I>);
}

constexpr void test() {
  // Contraint success - Signed
  using SI = long long int;
  test_constraint_success<signed char, SI>();
  test_constraint_success<short int, SI>();
  test_constraint_success<signed char, SI>();
  test_constraint_success<short int, SI>();
  test_constraint_success<int, SI>();
  test_constraint_success<long int, SI>();
  test_constraint_success<long long int, int>();
#ifndef TEST_HAS_NO_INT128
  test_constraint_success<__int128_t, SI>();
#endif
  // Contraint success - Unsigned
  using UI = unsigned long long int;
  test_constraint_success<unsigned char, UI>();
  test_constraint_success<unsigned short int, UI>();
  test_constraint_success<unsigned int, UI>();
  test_constraint_success<unsigned long int, UI>();
  test_constraint_success<unsigned long long int, unsigned int>();
#ifndef TEST_HAS_NO_INT128
  test_constraint_success<__uint128_t, UI>();
#endif

  // Contraint failure
  test_constraint_fail<bool>();
  test_constraint_fail<char>();
#ifndef TEST_HAS_NO_INT128
  test_constraint_fail<wchar_t>();
#endif
  test_constraint_fail<char8_t>();
  test_constraint_fail<char16_t>();
  test_constraint_fail<char32_t>();
  test_constraint_fail<float>();
  test_constraint_fail<double>();
  test_constraint_fail<long double>();
}
