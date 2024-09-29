//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// template <class T, size_t N, class U>
//   typename inplace_vector<T, N>::size_type
//   erase(inplace_vector<T, N>& c, const U& value);

#include <inplace_vector>
#include <optional>
#include <cassert>

#include "test_macros.h"

template <class S, class U>
constexpr void test0(S s, U val, S expected, std::size_t expected_erased_count) {
  ASSERT_SAME_TYPE(typename S::size_type, decltype(std::erase(s, val)));
  assert(expected_erased_count == std::erase(s, val));
  assert(s == expected);
}

template <class T, std::size_t N>
constexpr void test1() {
  using S = std::inplace_vector<T, N>;

  test0(S(), 1, S(), 0);

  if constexpr (S::capacity() >= 1) {
    test0(S({1}), 1, S(), 1);
    test0(S({1}), 2, S({1}), 0);
  }

  if constexpr (S::capacity() >= 2) {
    test0(S({1, 2}), 1, S({2}), 1);
    test0(S({1, 2}), 2, S({1}), 1);
    test0(S({1, 2}), 3, S({1, 2}), 0);
    test0(S({1, 1}), 1, S(), 2);
    test0(S({1, 1}), 3, S({1, 1}), 0);
  }

  if constexpr (S::capacity() >= 3) {
    test0(S({1, 2, 3}), 1, S({2, 3}), 1);
    test0(S({1, 2, 3}), 2, S({1, 3}), 1);
    test0(S({1, 2, 3}), 3, S({1, 2}), 1);
    test0(S({1, 2, 3}), 4, S({1, 2, 3}), 0);

    test0(S({1, 1, 1}), 1, S(), 3);
    test0(S({1, 1, 1}), 2, S({1, 1, 1}), 0);
    test0(S({1, 1, 2}), 1, S({2}), 2);
    test0(S({1, 1, 2}), 2, S({1, 1}), 1);
    test0(S({1, 1, 2}), 3, S({1, 1, 2}), 0);
    test0(S({1, 2, 2}), 1, S({2, 2}), 1);
    test0(S({1, 2, 2}), 2, S({1}), 2);
    test0(S({1, 2, 2}), 3, S({1, 2, 2}), 0);
  }

  //  Test cross-type erasure
  using opt = std::optional<typename S::value_type>;
  test0(S(), opt(), S(), 0);
  if constexpr (S::capacity() >= 3) {
    test0(S({1, 2, 1}), opt(), S({1, 2, 1}), 0);
    test0(S({1, 2, 1}), opt(1), S({2}), 2);
    test0(S({1, 2, 1}), opt(2), S({1, 1}), 1);
    test0(S({1, 2, 1}), opt(3), S({1, 2, 1}), 0);
  }
}

template <class T>
constexpr void test2() {
  test1<T, 0>();
  test1<T, 3>();
  test1<T, 10>();
  test1<T, 100>();
}

constexpr bool tests() {
  test2<int>();
  test2<long>();
  test2<double>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
