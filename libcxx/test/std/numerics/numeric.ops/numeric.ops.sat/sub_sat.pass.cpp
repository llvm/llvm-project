//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// The test uses "Placeholder variables with no name"

// <numeric>

// template<class T>
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

#include "test_macros.h"

template <typename IntegerT>
constexpr bool test_signed() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  // TODO(LLVM-20) remove [[maybe_unused]] since all supported compilers support "Placeholder variables with no name"
  [[maybe_unused]] std::same_as<IntegerT> decltype(auto) _ = std::sub_sat(minVal, maxVal);

  static_assert(noexcept(std::sub_sat(minVal, maxVal)));

  // clang-format off

  // Limit values (-1, 0, 1, min, max)

  assert(std::sub_sat(IntegerT{-1}, IntegerT{-1}) == IntegerT{ 0});
  assert(std::sub_sat(IntegerT{-1}, IntegerT{ 0}) == IntegerT{-1});
  assert(std::sub_sat(IntegerT{-1}, IntegerT{ 1}) == IntegerT{-2});
  assert(std::sub_sat(IntegerT{-1},       minVal) == IntegerT{-1} - minVal);
  assert(std::sub_sat(IntegerT{-1},       maxVal) == IntegerT{-1} - maxVal);

  assert(std::sub_sat(IntegerT{ 0}, IntegerT{-1}) == IntegerT{ 1});
  assert(std::sub_sat(IntegerT{ 0}, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::sub_sat(IntegerT{ 0}, IntegerT{ 1}) == IntegerT{-1});
  assert(std::sub_sat(IntegerT{ 0},       minVal) == maxVal); // saturated
  assert(std::sub_sat(IntegerT{ 0},       maxVal) == -maxVal);

  assert(std::sub_sat(      minVal, IntegerT{-1}) == minVal - IntegerT{-1});
  assert(std::sub_sat(      minVal, IntegerT{ 0}) == minVal);
  assert(std::sub_sat(      minVal, IntegerT{ 1}) == minVal); // saturated
  assert(std::sub_sat(      minVal,       minVal) == IntegerT{0});
  assert(std::sub_sat(      minVal,       maxVal) == minVal); // saturated

  assert(std::sub_sat(      maxVal, IntegerT{-1}) == maxVal); // saturated
  assert(std::sub_sat(      maxVal, IntegerT{ 0}) == maxVal);
  assert(std::sub_sat(      maxVal, IntegerT{ 1}) == maxVal - IntegerT{ 1});
  assert(std::sub_sat(      maxVal,       minVal) == maxVal); // saturated
  assert(std::sub_sat(      maxVal,       maxVal) == IntegerT{0});

  // No saturation (no limit values)

  assert(std::sub_sat(IntegerT{ 27}, IntegerT{-28}) ==  55);
  assert(std::sub_sat(IntegerT{ 27}, IntegerT{ 28}) ==  -1);
  assert(std::sub_sat(IntegerT{-27}, IntegerT{ 28}) == -55);
  assert(std::sub_sat(IntegerT{-27}, IntegerT{-28}) ==   1);

  // Saturation (no limit values)

  {
    constexpr IntegerT lesserVal = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT biggerVal = maxVal / IntegerT{2} + IntegerT{28};
    assert(std::sub_sat(lesserVal, biggerVal) == minVal); // saturated
  }
  {
    constexpr IntegerT biggerVal = maxVal / IntegerT{2} + IntegerT{28};
    constexpr IntegerT lesserVal = minVal / IntegerT{2} + IntegerT{27};
    assert(std::sub_sat(biggerVal, lesserVal) == maxVal); // saturated
  }

  // clang-format on

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  // TODO(LLVM-20) remove [[maybe_unused]] since all supported compilers support "Placeholder variables with no name"
  [[maybe_unused]] std::same_as<IntegerT> decltype(auto) _ = std::sub_sat(minVal, maxVal);

  static_assert(noexcept(std::sub_sat(minVal, maxVal)));

  // clang-format off

  // Limit values (0, 1, min, max)

  assert(std::sub_sat(IntegerT{0}, IntegerT{0}) == IntegerT{0});
  assert(std::sub_sat(IntegerT{0}, IntegerT{1}) == minVal); // saturated
  assert(std::sub_sat(IntegerT{0},      minVal) == minVal);
  assert(std::sub_sat(IntegerT{0},      maxVal) == minVal); // saturated

  assert(std::sub_sat(IntegerT{1}, IntegerT{0}) == IntegerT{1});
  assert(std::sub_sat(IntegerT{1}, IntegerT{1}) == IntegerT{0});
  assert(std::sub_sat(IntegerT{1},      minVal) == IntegerT{1});
  assert(std::sub_sat(IntegerT{1},      maxVal) == minVal); // saturated

  assert(std::sub_sat(     minVal, IntegerT{0}) == IntegerT{0});
  assert(std::sub_sat(     minVal, IntegerT{1}) == minVal);
  assert(std::sub_sat(     minVal,      maxVal) == minVal);
  assert(std::sub_sat(     minVal,      maxVal) == minVal);

  assert(std::sub_sat(     maxVal, IntegerT{0}) == maxVal);
  assert(std::sub_sat(     maxVal, IntegerT{1}) == maxVal - IntegerT{1});
  assert(std::sub_sat(     maxVal,      minVal) == maxVal);
  assert(std::sub_sat(     maxVal,      maxVal) == IntegerT{0});

  // Saturation (no limit values)

  {
    constexpr IntegerT lesserVal = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT biggerVal = maxVal / IntegerT{2} + IntegerT{28};
    assert(std::sub_sat(lesserVal, biggerVal) == minVal); // saturated
  }

  // clang-format on

  return true;
}

constexpr bool test() {
  // Signed
  test_signed<signed char>();
  test_signed<short int>();
  test_signed<int>();
  test_signed<long int>();
  test_signed<long long int>();
#ifndef TEST_HAS_NO_INT128
  test_signed<__int128_t>();
#endif
  // Unsigned
  test_unsigned<unsigned char>();
  test_unsigned<unsigned short int>();
  test_unsigned<unsigned int>();
  test_unsigned<unsigned long int>();
  test_unsigned<unsigned long long int>();
#ifndef TEST_HAS_NO_INT128
  test_unsigned<__uint128_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
