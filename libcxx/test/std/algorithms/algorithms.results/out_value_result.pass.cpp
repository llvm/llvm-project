//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class _O, class _T>
// struct out_value_result;

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::out_value_result<A, A>, std::ranges::out_value_result<int, int>>);

struct B {
  B(int);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::out_value_result<B, B>, std::ranges::out_value_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::out_value_result<B, B>, std::ranges::out_value_result<int, int>&>);
static_assert(std::is_constructible_v<std::ranges::out_value_result<B, B>, const std::ranges::out_value_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::out_value_result<B, B>, const std::ranges::out_value_result<int, int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::out_value_result<C, C>, std::ranges::out_value_result<int, int>&>);

// has to be convertible via const&
static_assert(std::is_convertible_v<std::ranges::out_value_result<int, int>&, std::ranges::out_value_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::out_value_result<int, int>&, std::ranges::out_value_result<long, long>>);
static_assert(std::is_convertible_v<std::ranges::out_value_result<int, int>&&, std::ranges::out_value_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::out_value_result<int, int>&&, std::ranges::out_value_result<long, long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::out_value_result<MoveOnly, int>>);
static_assert(std::is_move_constructible_v<std::ranges::out_value_result<int, MoveOnly>>);

// should not be copy constructible with move-only type
static_assert(!std::is_copy_constructible_v<std::ranges::out_value_result<MoveOnly, int>>);
static_assert(!std::is_copy_constructible_v<std::ranges::out_value_result<int, MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::out_value_result<NotConvertible, int>, std::ranges::out_value_result<int, int>>);
static_assert(!std::is_convertible_v<std::ranges::out_value_result<int, NotConvertible>, std::ranges::out_value_result<int, int>>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  // Checks that conversion operations are correct.
  {
    const std::ranges::out_value_result<int, double> res{10, 0.};
    assert(res.out == 10);
    assert(res.value == 0.);
    const std::ranges::out_value_result<ConvertibleFrom<int>, ConvertibleFrom<double>> res2 = res;
    assert(res2.out.content == 10);
    assert(res2.value.content == 0.);
  }

  // Checks that conversions are possible when one of the types is move-only.
  {
    std::ranges::out_value_result<MoveOnly, int> res{MoveOnly{}, 2};
    assert(res.out.get() == 1);
    assert(res.value == 2);
    const auto res2 = static_cast<std::ranges::out_value_result<MoveOnly, int>>(std::move(res));
    assert(res.out.get() == 0);
    assert(res2.out.get() == 1);
    assert(res2.value == 2);
  }

  // Checks that structured bindings get the correct values.
  {
    const auto [out, value] = std::ranges::out_value_result<int, int>{1, 2};
    assert(out == 1);
    assert(value == 2);
  }
  return true;
}

int main() {
  test();
  static_assert(test());
}
