//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Some basic examples of how adjacent_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <cstddef>
#include <ranges>

#include <cassert>
#include <string_view>
#include <vector>

constexpr void test_adjacent_pairs() {
  std::vector v = {1, 2, 3, 4};

  std::pair<size_t, size_t> expected_index{0, 1};
  for (auto [x, y] : v | std::views::adjacent<2>) {
    assert(x == v[expected_index.first]);
    assert(y == v[expected_index.second]);
    assert(&x == &v[expected_index.first]);
    assert(&y == &v[expected_index.second]);
    ++expected_index.first;
    ++expected_index.second;
  }
}

constexpr void test_string_view() {
  std::string_view sv = "123456789";
  auto v              = sv | std::views::adjacent<3>;
  auto [a, b, c]      = *v.begin();
  assert(a == '1');
  assert(b == '2');
  assert(c == '3');
}

constexpr bool test() {
  test_adjacent_pairs();
  test_string_view();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
