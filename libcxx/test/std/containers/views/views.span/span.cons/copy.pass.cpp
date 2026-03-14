//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

//  constexpr span(const span& other) noexcept = default;

#include <span>
#include <cassert>
#include <string>
#include <utility>

#include "test_macros.h"

template <class T>
constexpr void test() {
  ASSERT_NOEXCEPT(std::span<T>(std::declval<std::span<T> const&>()));
  ASSERT_NOEXCEPT(std::span<T>{std::declval<std::span<T> const&>()});

  // dynamic_extent
  {
    std::span<T> x;
    std::span<T> copy(x);
    assert(copy.data() == x.data());
    assert(copy.size() == x.size());
  }
  {
    T array[3] = {};
    std::span<T> x(array, 3);
    std::span<T> copy(x);
    assert(copy.data() == array);
    assert(copy.size() == 3);
  }
  {
    T array[3] = {};
    std::span<T> x(array, 2);
    std::span<T> copy(x);
    assert(copy.data() == array);
    assert(copy.size() == 2);
  }

  // static extent
  {
    std::span<T, 0> x;
    std::span<T, 0> copy(x);
    assert(copy.data() == x.data());
    assert(copy.size() == x.size());
  }
  {
    T array[3] = {};
    std::span<T, 3> x(array);
    std::span<T, 3> copy(x);
    assert(copy.data() == array);
    assert(copy.size() == 3);
  }
  {
    T array[2] = {};
    std::span<T, 2> x(array);
    std::span<T, 2> copy(x);
    assert(copy.data() == array);
    assert(copy.size() == 2);
  }
}

struct Foo {};

constexpr bool test_all() {
  test<int>();
  test<const int>();
  test<volatile int>();
  test<const volatile int>();

  test<long>();
  test<const long>();
  test<volatile long>();
  test<const volatile long>();

  test<double>();
  test<const double>();
  test<volatile double>();
  test<const volatile double>();

  // Note: Can't test non-fundamental types with volatile because we require `T*` to be indirectly_readable,
  //       which isn't the case when T is volatile.
  test<Foo>();
  test<const Foo>();

  test<std::string>();
  test<const std::string>();

  // Regression test for https://llvm.org/PR104496
  {
    struct Incomplete;
    std::span<Incomplete> x;
    std::span<Incomplete> copy(x);
    assert(copy.data() == x.data());
    assert(copy.size() == x.size());
  }

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
