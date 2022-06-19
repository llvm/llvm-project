//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <utility>

// template<class R, class T>
//   constexpr bool in_range(T t) noexcept;               // C++20

#include <utility>
#include <limits>
#include <numeric>
#include <tuple>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

template <typename T>
struct Tuple {
  T min;
  T max;
  T mid;
  constexpr Tuple() {
    min = std::numeric_limits<T>::min();
    max = std::numeric_limits<T>::max();
    mid = std::midpoint(min, max);
  }
};

template <typename T>
constexpr void test_in_range1() {
  constexpr Tuple<T> tup;
  assert(std::in_range<T>(tup.min));
  assert(std::in_range<T>(tup.min + 1));
  assert(std::in_range<T>(tup.max));
  assert(std::in_range<T>(tup.max - 1));
  assert(std::in_range<T>(tup.mid));
  assert(std::in_range<T>(tup.mid - 1));
  assert(std::in_range<T>(tup.mid + 1));
}

constexpr void test_in_range() {
  constexpr Tuple<uint8_t> utup8;
  constexpr Tuple<int8_t> stup8;
  assert(!std::in_range<int8_t>(utup8.max));
  assert(std::in_range<short>(utup8.max));
  assert(!std::in_range<uint8_t>(stup8.min));
  assert(std::in_range<int8_t>(utup8.mid));
  assert(!std::in_range<uint8_t>(stup8.mid));
  assert(!std::in_range<uint8_t>(-1));
}

template <class... Ts>
constexpr void test1(const std::tuple<Ts...>&) {
  (test_in_range1<Ts>() , ...);
}

constexpr bool test() {
  std::tuple<
#ifndef TEST_HAS_NO_INT128
      __int128_t, __uint128_t,
#endif
      unsigned long long, long long, unsigned long, long, unsigned int, int,
      unsigned short, short, unsigned char, signed char> types;
  test1(types);
  test_in_range();
  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::in_range<int>(-1));
  test();
  static_assert(test());
  return 0;
}
