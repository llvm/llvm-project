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
// struct in_fun_result;

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::in_fun_result<A, A>, std::ranges::in_fun_result<int, int>>);

struct B {
  B(int);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::in_fun_result<B, B>, std::ranges::in_fun_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_fun_result<B, B>, std::ranges::in_fun_result<int, int>&>);
static_assert(std::is_constructible_v<std::ranges::in_fun_result<B, B>, const std::ranges::in_fun_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_fun_result<B, B>, const std::ranges::in_fun_result<int, int>&>);

struct C {
  C(int&);
};
// has to be convertible via const&
static_assert(!std::is_constructible_v<std::ranges::in_fun_result<C, C>, std::ranges::in_fun_result<int, int>&>);

static_assert(std::is_convertible_v<std::ranges::in_fun_result<int, int>&, std::ranges::in_fun_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_fun_result<int, int>&, std::ranges::in_fun_result<long, long>>);
static_assert(std::is_convertible_v<std::ranges::in_fun_result<int, int>&&, std::ranges::in_fun_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_fun_result<int, int>&&, std::ranges::in_fun_result<long, long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::in_fun_result<MoveOnly, int>>);
static_assert(std::is_move_constructible_v<std::ranges::in_fun_result<int, MoveOnly>>);

// should not copy constructible with move-only type
static_assert(!std::is_copy_constructible_v<std::ranges::in_fun_result<MoveOnly, int>>);
static_assert(!std::is_copy_constructible_v<std::ranges::in_fun_result<int, MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::in_fun_result<NotConvertible, int>,
                                     std::ranges::in_fun_result<int, int>>);
static_assert(!std::is_convertible_v<std::ranges::in_fun_result<int, NotConvertible>,
                                     std::ranges::in_fun_result<int, int>>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  {
    std::ranges::in_fun_result<int, double> res{10, 0.};
    assert(res.in == 10);
    assert(res.fun == 0.);
    std::ranges::in_fun_result<ConvertibleFrom<int>, ConvertibleFrom<double>> res2 = res;
    assert(res2.in.content == 10);
    assert(res2.fun.content == 0.);
  }
  {
    std::ranges::in_fun_result<MoveOnly, int> res{MoveOnly{}, 2};
    assert(res.in.get() == 1);
    assert(res.fun == 2);
    auto res2 = static_cast<std::ranges::in_fun_result<MoveOnly, int>>(std::move(res));
    assert(res.in.get() == 0);
    assert(res2.in.get() == 1);
    assert(res2.fun == 2);
  }
  {
    auto [in, fun] = std::ranges::in_fun_result<int, int>{1, 2};
    assert(in == 1);
    assert(fun == 2);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
