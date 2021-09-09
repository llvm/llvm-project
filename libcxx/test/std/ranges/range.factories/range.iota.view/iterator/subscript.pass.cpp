//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr W operator[](difference_type n) const
//   requires advanceable<W>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

template<class T>
constexpr void testType() {
  {
    std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
      assert(iter[i] == T(i));
  }
  {
    std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
      assert(iter[i] == T(i + 10));
  }
  {
    const std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
      assert(iter[i] == T(i));
  }
  {
    const std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
      assert(iter[i] == T(i + 10));
  }
}

constexpr bool test() {
  testType<SomeInt>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
