//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// void swap(const tuple& rhs);

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <tuple>
#include <utility>

#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
class SwapThrower {
  void swap(SwapThrower&) = delete;
  void swap(const SwapThrower&) const = delete;
};

void swap(const SwapThrower&, const SwapThrower&) { throw 0.f; }

static_assert(std::is_swappable_v<const SwapThrower>);
static_assert(std::is_swappable_with_v<const SwapThrower&, const SwapThrower&>);

void test_noexcept() {
  const std::tuple<SwapThrower> t1;
  const std::tuple<SwapThrower> t2;

  try {
    t1.swap(t2);
    std::swap(t1, t2);
    assert(false);
  } catch (float) {
  }

  try {
    std::swap(std::as_const(t1), std::as_const(t2));
    assert(false);
  } catch (float) {
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

struct ConstSwappable {
  mutable int i;
};

constexpr void swap(const ConstSwappable& lhs, const ConstSwappable& rhs) { std::swap(lhs.i, rhs.i); }

constexpr bool test() {
  {
    typedef std::tuple<const ConstSwappable> T;
    const T t0(ConstSwappable{0});
    T t1(ConstSwappable{1});
    t0.swap(t1);
    assert(std::get<0>(t0).i == 1);
    assert(std::get<0>(t1).i == 0);
  }
  {
    typedef std::tuple<ConstSwappable, ConstSwappable> T;
    const T t0({0}, {1});
    const T t1({2}, {3});
    t0.swap(t1);
    assert(std::get<0>(t0).i == 2);
    assert(std::get<1>(t0).i == 3);
    assert(std::get<0>(t1).i == 0);
    assert(std::get<1>(t1).i == 1);
  }
  {
    typedef std::tuple<ConstSwappable, const ConstSwappable, const ConstSwappable> T;
    const T t0({0}, {1}, {2});
    const T t1({3}, {4}, {5});
    t0.swap(t1);
    assert(std::get<0>(t0).i == 3);
    assert(std::get<1>(t0).i == 4);
    assert(std::get<2>(t0).i == 5);
    assert(std::get<0>(t1).i == 0);
    assert(std::get<1>(t1).i == 1);
    assert(std::get<2>(t1).i == 2);
  }
  return true;
}

int main(int, char**) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  test_noexcept();
#endif
  test();

// gcc cannot have mutable member in constant expression
#if !defined(TEST_COMPILER_GCC)
  static_assert(test());
#endif

  return 0;
}
