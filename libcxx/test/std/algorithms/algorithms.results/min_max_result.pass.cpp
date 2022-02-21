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

// template <class I1, class I2>
// struct min_max_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

struct A {
  explicit A(int);
};
static_assert(!std::is_constructible_v<std::ranges::min_max_result<A>, std::ranges::min_max_result<int>>);

struct B {
  B(const int&);
  B(int&&);
};
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, std::ranges::min_max_result<int>>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, std::ranges::min_max_result<int>&>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, const std::ranges::min_max_result<int>>);
static_assert(std::is_constructible_v<std::ranges::min_max_result<B>, const std::ranges::min_max_result<int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::min_max_result<C>, std::ranges::min_max_result<int>&>);

static_assert(std::is_convertible_v<std::ranges::min_max_result<int>&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::min_max_result<int>&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<std::ranges::min_max_result<int>&&, std::ranges::min_max_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::min_max_result<int>&&, std::ranges::min_max_result<long>>);

struct NotConvertible {};
static_assert(!std::is_convertible_v<std::ranges::min_max_result<NotConvertible>, std::ranges::min_max_result<int>>);

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
    [[maybe_unused]] auto res2 = static_cast<std::ranges::min_max_result<MoveOnly>>(std::move(res));
    assert(res.min.get() == 0);
    assert(res.max.get() == 0);
  }
  auto [min, max] = std::ranges::min_max_result<int>{1, 2};
  assert(min == 1);
  assert(max == 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
