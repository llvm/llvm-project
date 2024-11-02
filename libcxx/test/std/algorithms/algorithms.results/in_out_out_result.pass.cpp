//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class I1, class O1, class O2>
// struct in_out_out_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::in_out_out_result<A, A, A>, std::ranges::in_out_out_result<int, int, int>>);

struct B {
  B(int);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::in_out_out_result<B, B, B>, std::ranges::in_out_out_result<int, int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_out_out_result<B, B, B>, std::ranges::in_out_out_result<int, int, int>&>);
static_assert(std::is_constructible_v<std::ranges::in_out_out_result<B, B, B>, const std::ranges::in_out_out_result<int, int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_out_out_result<B, B, B>, const std::ranges::in_out_out_result<int, int, int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::in_out_out_result<C, C, C>, std::ranges::in_out_out_result<int, int, int>&>);

// has to be convertible via const&
static_assert(std::is_convertible_v<std::ranges::in_out_out_result<int, int, int>&, std::ranges::in_out_out_result<long, long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_out_result<int, int, int>&, std::ranges::in_out_out_result<long, long, long>>);
static_assert(std::is_convertible_v<std::ranges::in_out_out_result<int, int, int>&&, std::ranges::in_out_out_result<long, long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_out_result<int, int, int>&&, std::ranges::in_out_out_result<long, long, long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::in_out_out_result<MoveOnly, int, int>>);
static_assert(std::is_move_constructible_v<std::ranges::in_out_out_result<int, MoveOnly, int>>);
static_assert(std::is_move_constructible_v<std::ranges::in_out_out_result<int, int, MoveOnly>>);

static_assert(!std::is_copy_constructible_v<std::ranges::in_out_out_result<MoveOnly, int, int>>);
static_assert(!std::is_copy_constructible_v<std::ranges::in_out_out_result<int, MoveOnly, int>>);
static_assert(!std::is_copy_constructible_v<std::ranges::in_out_out_result<int, int, MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::in_out_out_result<NotConvertible, int, int>, std::ranges::in_out_out_result<int, int, int>>);
static_assert(!std::is_convertible_v<std::ranges::in_out_out_result<int, NotConvertible, int>, std::ranges::in_out_out_result<int, int, int>>);
static_assert(!std::is_convertible_v<std::ranges::in_out_out_result<int, int, NotConvertible>, std::ranges::in_out_out_result<int, int, int>>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  {
    std::ranges::in_out_out_result<int, double, float> res{10, 0., 1.f};
    assert(res.in == 10);
    assert(res.out1 == 0.);
    assert(res.out2 == 1.f);
    std::ranges::in_out_out_result<ConvertibleFrom<int>, ConvertibleFrom<double>, ConvertibleFrom<float>> res2 = res;
    assert(res2.in.content == 10);
    assert(res2.out1.content == 0.);
    assert(res2.out2.content == 1.f);
  }
  {
    std::ranges::in_out_out_result<MoveOnly, int, int> res1{MoveOnly{}, 0, 0};
    assert(res1.in.get() == 1);
    auto res2 = static_cast<std::ranges::in_out_out_result<MoveOnly, int, int>>(std::move(res1));
    assert(res1.in.get() == 0);
    assert(res2.in.get() == 1);
  }
  {
    auto [in, out1, out2] = std::ranges::in_out_out_result<int, int, int>{1, 2, 3};
    assert(in == 1);
    assert(out1 == 2);
    assert(out2 == 3);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
