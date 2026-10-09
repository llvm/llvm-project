//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

// REQUIRES: std-at-least-c++26

#include <array>
#include <cassert>
#include <concepts>
#include <numeric>
#include <simd>

#include "type_algorithms.h"
#include "../../utils.h"

namespace dp = std::datapar;

template <class T, std::size_t N>
constexpr void test(dp::simd<T, N> lhs, dp::simd<T, N> rhs, std::array<bool, N> expected) {
  { // Test operator<
    std::same_as<typename dp::simd<T, N>::mask_type> auto result = lhs < rhs;
    for (size_t i = 0; i != N; ++i) {
      assert(result[i] == expected[i]);
    }
  }
  { // Test operator>
    std::same_as<typename dp::simd<T, N>::mask_type> auto result = rhs > lhs;
    for (size_t i = 0; i != N; ++i) {
      assert(result[i] == expected[i]);
    }
  }
  { // Test operator>=
    std::same_as<typename dp::simd<T, N>::mask_type> auto result = lhs >= rhs;
    for (size_t i = 0; i != N; ++i) {
      assert(result[i] == !expected[i]);
    }
  }
  { // Test operator<=
    std::same_as<typename dp::simd<T, N>::mask_type> auto result = rhs <= lhs;
    for (size_t i = 0; i != N; ++i) {
      assert(result[i] == !expected[i]);
    }
  }
}

constexpr bool test() {
  types::for_each(types::vectorizable_types{}, []<class T> {
    test<T, 4>(std::array<T, 4>{1, 2, 3, 4}, std::array<T, 4>{1, 2, 3, 4}, {false, false, false, false});
    test<T, 4>(std::array<T, 4>{4, 3, 2, 1}, std::array<T, 4>{1, 2, 3, 4}, {false, false, true, true});
    test<T, 4>(std::array<T, 4>{1, 2, 3, 4}, std::array<T, 4>{1, 2, 4, 3}, {false, false, true, false});
    test<T, 4>(std::array<T, 4>{1, 1, 3, 4}, std::array<T, 4>{1, 2, 4, 1}, {false, true, true, false});
  });
  types::for_each(types::vectorizable_float_types{}, []<class T> {
    constexpr auto nan = std::numeric_limits<T>::quiet_NaN();
    dp::simd<T, 4> a = std::array<T, 4>{nan, nan, nan, nan};
    dp::simd<T, 4> b = a;
    assert(dp::none_of(a < b));
    assert(dp::none_of(a > b));
    assert(dp::none_of(a <= b));
    assert(dp::none_of(a >= b));
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
