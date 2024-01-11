//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il); // Since C++26

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <vector>

#include "test_macros.h"

#include <print>

#if TEST_STD_VER >= 26

constexpr int kEmptySpanValue = -1;

template <std::size_t N>
int take_last_element(std::span<const int, N> sp, std::size_t expectedSize) {
  static_assert(std::dynamic_extent != sp.extent);
  assert(sp.size() == expectedSize);
  if (sp.size() == 0)
    return kEmptySpanValue;
  return sp.at(sp.size() - 1);
}

int take_last_element(std::span<const int> sp, std::size_t expectedSize) {
  static_assert(std::dynamic_extent == sp.extent);
  std::println(stderr, "size ----- {}", sp.size());
  assert(sp.size() == expectedSize);
  if (sp.size() == 0)
    return kEmptySpanValue;
  return sp.at(sp.size() - 1);
}

bool test_span() {
  // Static
  // {
  //   int lastElem = take_last_element<0>({{}}, 0);
  //   assert(lastElem == kEmptySpanValue);
  // }
  // {
  //   int lastElem = take_last_element<1>({{1}}, 1);
  //   assert(lastElem == kEmptySpanValue);
  // }
  // {
  //   int lastElem = take_last_element<4>(std::array{1, 2, 3, 9084}, 4);
  //   assert(lastElem == 9084);
  // }
  // {
  //   int lastElem = take_last_element<4>(std::initializer_list<int>{1, 2, 3, 9084}, 4);
  //   assert(lastElem == 9084);
  // }
  // std::span<const int, 4>({1, 2, 3, 9084, 5});
  // {
  //   int lastElem = take_last_element(std::span<const int, 4>({1, 2, 3, 9084}), 4);
  //   assert(lastElem == 9084);
  // }
  // Dynamic
  // {
  //   int lastElem = take_last_element({{}}, 1);
  //   assert(lastElem == kEmptySpanValue);
  // }
  {
    int lastElem = take_last_element({{1, 2, 3, 9084}}, 4);
    assert(lastElem == 9084);
  }
  // {
  //   int lastElem = take_last_element(std::vector{1, 2, 3, 9084}, 4);
  //   assert(lastElem == 9084);
  // }
  {
    int lastElem = take_last_element(std::initializer_list<int>{1, 2, 3, 9084}, 4);
    assert(lastElem == 9084);
  }
  {
    int lastElem = take_last_element(std::span<const int>({1, 2, 3, 9084}), 4);
    assert(lastElem == 9084);
  }

  return true;
}

#endif

struct Sink {
  constexpr Sink() = default;
  constexpr Sink(Sink*) {}
};

constexpr std::size_t count(std::span<const Sink> sp) { return sp.size(); }

template <std::size_t N>
constexpr std::size_t count_n(std::span<const Sink, N> sp) {
  return sp.size();
}

constexpr bool test() {
#if TEST_STD_VER >= 26
  // Dynamic extent
  {
    Sink a[10];

    assert(count({a}) == 1);
    assert(count({a, a + 10}) == 2);
    assert(count({a, a + 1, a + 2}) == 3);
    // assert(count_n<3>(std::array{a, a + 1, a + 2}) == 3);
    // assert(count_n<3>(std::array{a, a + 1, a + 2, a + 3, a + 4}) == 3);
  }
#else
  {
    Sink a[10];

    assert(count({a}) == 10);
    assert(count({a, a + 10}) == 10);
    assert(count_n<10>({a}) == 10);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#if TEST_STD_VER >= 26
  test_span();
#endif

  return 0;
}
