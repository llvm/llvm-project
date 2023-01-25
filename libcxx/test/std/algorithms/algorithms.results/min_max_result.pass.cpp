//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class I1, class I2>
// struct min_max_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::min_max_result<A>, std::ranges::min_max_result<int>>);

struct B {
  B(int);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, std::ranges::min_max_result<int>>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, std::ranges::min_max_result<int>&>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, const std::ranges::min_max_result<int>>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, const std::ranges::min_max_result<int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::min_max_result<C>, std::ranges::min_max_result<int>&>);

// has to be convertible via const&
static_assert(std::is_convertible_v<std::ranges::min_max_result<int>&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::min_max_result<int>&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<std::ranges::min_max_result<int>&&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::min_max_result<int>&&, std::ranges::min_max_result<long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::min_max_result<MoveOnly>>);

// should not be copy constructible
static_assert(!std::is_copy_constructible_v<std::ranges::min_max_result<MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::min_max_result<NotConvertible>, std::ranges::min_max_result<int>>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  {
    std::ranges::min_max_result<double> res{10, 1};
    assert(res.min == 10);
    assert(res.max == 1);
    std::ranges::min_max_result<ConvertibleFrom<int>> res2 = res;
    assert(res2.min.content == 10);
    assert(res2.max.content == 1);
  }
  {
    std::ranges::min_max_result<MoveOnly> res{MoveOnly{}, MoveOnly{}};
    assert(res.min.get() == 1);
    assert(res.max.get() == 1);
    auto res2 = std::move(res);
    assert(res.min.get() == 0);
    assert(res.max.get() == 0);
    assert(res2.min.get() == 1);
    assert(res2.max.get() == 1);
  }
  {
    auto [min, max] = std::ranges::min_max_result<int>{1, 2};
    assert(min == 1);
    assert(max == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
