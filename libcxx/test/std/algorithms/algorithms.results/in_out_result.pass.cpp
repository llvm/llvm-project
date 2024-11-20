//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class I1, class O1>
// struct in_out_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::in_out_result<A, A>, std::ranges::in_out_result<int, int>>);

struct B {
  B(int);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::in_out_result<B, B>, std::ranges::in_out_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_out_result<B, B>, std::ranges::in_out_result<int, int>&>);
static_assert(std::is_constructible_v<std::ranges::in_out_result<B, B>, const std::ranges::in_out_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_out_result<B, B>, const std::ranges::in_out_result<int, int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::in_out_result<C, C>, std::ranges::in_out_result<int, int>&>);

// has to be convertible via const&
static_assert(std::is_convertible_v<std::ranges::in_out_result<int, int>&, std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_result<int, int>&, std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<std::ranges::in_out_result<int, int>&&, std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_result<int, int>&&, std::ranges::in_out_result<long, long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::in_out_result<MoveOnly, int>>);
static_assert(std::is_move_constructible_v<std::ranges::in_out_result<int, MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::in_out_result<NotConvertible, int>, std::ranges::in_out_result<int, NotConvertible>>);
static_assert(!std::is_convertible_v<std::ranges::in_out_result<int, NotConvertible>, std::ranges::in_out_result<NotConvertible, int>>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  // Checks that conversion operations are correct.
  {
    std::ranges::in_out_result<double, int> res{10, 1};
    assert(res.in == 10);
    assert(res.out == 1);
    std::ranges::in_out_result<ConvertibleFrom<double>, ConvertibleFrom<int>> res2 = res;
    assert(res2.in.content == 10);
    assert(res2.out.content == 1);
  }

  // Checks that conversions are possible when one of the types is move-only.
  {
    std::ranges::in_out_result<MoveOnly, int> res{MoveOnly{}, 10};
    assert(res.in.get() == 1);
    assert(res.out == 10);
    auto res2 = std::move(res);
    assert(res.in.get() == 0);
    assert(res.out == 10);
    assert(res2.in.get() == 1);
    assert(res2.out == 10);
  }

  // Checks that structured bindings get the correct values.
  {
    auto [min, max] = std::ranges::in_out_result<int, int>{1, 2};
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
