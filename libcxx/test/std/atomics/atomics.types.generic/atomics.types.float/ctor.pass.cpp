//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// constexpr atomic() noexcept;
// constexpr atomic(floating-point-type) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

template <class T>
constinit std::atomic<T> a1 = T();

template <class T>
constinit std::atomic<T> a2 = T(5.2);

template <class T>
constexpr void testOne() {
  static_assert(std::is_nothrow_constructible_v<std::atomic<T>>);
  static_assert(std::is_nothrow_constructible_v<std::atomic<T>, T>);

  // constexpr atomic() noexcept;
  {
    std::atomic<T> a = {};
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(a.load() == T(0));
    }
  }

  // constexpr atomic(floating-point-type) noexcept;
  {
    std::atomic<T> a = T(5.2);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(a.load() == T(5.2));
    }
  }

  // test constinit
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(a1<T> == T(0.0));
    assert(a2<T> == T(5.2));
  }
}

constexpr bool test() {
  testOne<float>();
  testOne<double>();
  // TODO https://llvm.org/PR48634
  // testOne<long double>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
