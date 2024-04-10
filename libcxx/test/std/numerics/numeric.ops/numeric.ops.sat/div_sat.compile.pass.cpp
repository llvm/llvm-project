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
// constexpr T div_sat(T x, T y) noexcept;                     // freestanding

#include <concepts>
#include <numeric>
#include <type_traits>

#include "test_macros.h"

// Constraints

template <typename T, typename U>
concept CanDo = requires(T x, U y) {
  { std::div_sat(x, y) } -> std::same_as<T>;
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

//  A function call expression that violates the precondition in the Preconditions: element is not a core constant expression (7.7 [expr.const]).

template <auto N>
using QuotT = std::integral_constant<decltype(N), std::div_sat(N, N)>;

template <auto N>
QuotT<N> div_by_zero();

template <auto N>
concept CanDivByZero = requires { div_by_zero<N>(); };

static_assert(!CanDivByZero<static_cast<signed char>(0)>);
static_assert(!CanDivByZero<static_cast<short int>(0)>);
static_assert(!CanDivByZero<0>);
static_assert(!CanDivByZero<0L>);
static_assert(!CanDivByZero<0LL>);
#ifndef TEST_HAS_NO_INT128
static_assert(!CanDivByZero<static_cast<__int128_t>(0)>);
#endif
static_assert(!CanDivByZero<static_cast<unsigned char>(0)>);
static_assert(!CanDivByZero<static_cast<unsigned short int>(0)>);
static_assert(!CanDivByZero<0U>);
static_assert(!CanDivByZero<0UL>);
static_assert(!CanDivByZero<0ULL>);
#ifndef TEST_HAS_NO_INT128
static_assert(!CanDivByZero<static_cast<__uint128_t>(0)>);
#endif
