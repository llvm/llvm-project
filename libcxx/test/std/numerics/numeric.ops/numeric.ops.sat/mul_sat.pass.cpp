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
// constexpr T mul_sat(T x, T y) noexcept;                     // freestanding

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
  [[maybe_unused]] std::same_as<IntegerT> decltype(auto) _ = std::mul_sat(minVal, maxVal);

  static_assert(noexcept(std::mul_sat(minVal, maxVal)));

  // clang-format off

  // Limit values (-1, 0, 1, min, max)

  assert(std::mul_sat(IntegerT{-1}, IntegerT{-1}) == IntegerT{ 1});
  assert(std::mul_sat(IntegerT{-1}, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{-1}, IntegerT{ 1}) == IntegerT{-1});
  assert(std::mul_sat(IntegerT{-1},       minVal) == maxVal); // saturated
  assert(std::mul_sat(IntegerT{-1},       maxVal) == -maxVal);

  assert(std::mul_sat(IntegerT{ 0}, IntegerT{-1}) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{ 0}, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{ 0}, IntegerT{ 1}) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{ 0},       minVal) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{ 0},       maxVal) == IntegerT{ 0});

  assert(std::mul_sat(IntegerT{ 1}, IntegerT{-1}) == IntegerT{-1});
  assert(std::mul_sat(IntegerT{ 1}, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::mul_sat(IntegerT{ 1}, IntegerT{ 1}) == IntegerT{ 1});
  assert(std::mul_sat(IntegerT{ 1},       minVal) == minVal);
  assert(std::mul_sat(IntegerT{ 1},       maxVal) == maxVal);

  assert(std::mul_sat(      minVal, IntegerT{-1}) == maxVal); // saturated
  assert(std::mul_sat(      minVal, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::mul_sat(      minVal, IntegerT{ 1}) == minVal);
  assert(std::mul_sat(      minVal,       minVal) == maxVal); // saturated
  assert(std::mul_sat(      minVal,       maxVal) == minVal); // saturated

  assert(std::mul_sat(      maxVal, IntegerT{-1}) == -maxVal);
  assert(std::mul_sat(      maxVal, IntegerT{ 0}) == IntegerT{ 0});
  assert(std::mul_sat(      maxVal, IntegerT{ 1}) == maxVal); // saturated
  assert(std::mul_sat(      maxVal,       minVal) == minVal); // saturated
  assert(std::mul_sat(      maxVal,       maxVal) == maxVal); // saturated

  // No saturation (no limit values)

  assert(std::mul_sat(IntegerT{27}, IntegerT{ 2}) == IntegerT{54});
  assert(std::mul_sat(IntegerT{ 2}, IntegerT{28}) == IntegerT{56});

  // Saturation (no limit values)

  {
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = minVal / IntegerT{2} + IntegerT{28};
    assert(std::mul_sat(x, y) == maxVal); // saturated
  }
  {
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};
    assert(std::mul_sat(x, y) == minVal); // saturated
  }
  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = minVal / IntegerT{2} + IntegerT{28};
    assert(std::mul_sat(x, y) == minVal); // saturated
  }
  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};
    assert(std::mul_sat(x, y) == maxVal); // saturated
  }

  // clang-format on

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  // TODO(LLVM-20) remove [[maybe_unused]] since all supported compilers support "Placeholder variables with no name"
  [[maybe_unused]] std::same_as<IntegerT> decltype(auto) _ = std::mul_sat(minVal, maxVal);

  static_assert(noexcept(std::mul_sat(minVal, maxVal)));

  // clang-format off

  // No saturation (0, 1)

  assert(std::mul_sat(IntegerT{0}, IntegerT{0}) == IntegerT{0});
  assert(std::mul_sat(IntegerT{0}, IntegerT{1}) == IntegerT{0});
  assert(std::mul_sat(IntegerT{0},      minVal) == IntegerT{0});
  assert(std::mul_sat(IntegerT{0},      maxVal) == IntegerT{0});

  assert(std::mul_sat(IntegerT{1}, IntegerT{0}) == IntegerT{0});
  assert(std::mul_sat(IntegerT{1}, IntegerT{1}) == IntegerT{1});
  assert(std::mul_sat(IntegerT{1},      minVal) == minVal);
  assert(std::mul_sat(IntegerT{1},      maxVal) == maxVal);

  assert(std::mul_sat(     minVal, IntegerT{0}) == IntegerT{0});
  assert(std::mul_sat(     minVal, IntegerT{1}) == minVal);
  assert(std::mul_sat(     minVal,      maxVal) == minVal);
  assert(std::mul_sat(     minVal,      maxVal) == minVal);

  assert(std::mul_sat(     maxVal, IntegerT{0}) == IntegerT{0});
  assert(std::mul_sat(     maxVal, IntegerT{1}) == maxVal);
  assert(std::mul_sat(     maxVal,      minVal) == IntegerT{0});
  assert(std::mul_sat(     maxVal,      maxVal) == maxVal); // saturated

  // No saturation (no limit values)

  assert(std::mul_sat(IntegerT{28}, IntegerT{2}) == IntegerT{56});

  // Saturation (no limit values

  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};
    assert(std::mul_sat(x, y) == maxVal); // saturated
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
