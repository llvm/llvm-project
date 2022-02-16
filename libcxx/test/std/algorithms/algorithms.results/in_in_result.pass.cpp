//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template <class I1, class I2>
// struct in_in_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

template <class T>
struct ConstructibleFrom {
  constexpr ConstructibleFrom(T c) : content{c} {}
  T content;
};

struct A {
  explicit A(int);
};
static_assert(!std::is_constructible_v<std::ranges::in_in_result<A, A>,
                                       std::ranges::in_in_result<int, int>>);

struct B {
  B(const int&);
  B(int&&);
};
static_assert(std::is_constructible_v<std::ranges::in_in_result<B, B>, std::ranges::in_in_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_in_result<B, B>, std::ranges::in_in_result<int, int>&>);
static_assert(std::is_constructible_v<std::ranges::in_in_result<B, B>, const std::ranges::in_in_result<int, int>>);
static_assert(std::is_constructible_v<std::ranges::in_in_result<B, B>, const std::ranges::in_in_result<int, int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::in_in_result<C, C>, std::ranges::in_in_result<int, int>&>);

static_assert(std::is_convertible_v<         std::ranges::in_in_result<int, int>&, std::ranges::in_in_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<std::ranges::in_in_result<int, int>&, std::ranges::in_in_result<long, long>>);
static_assert(std::is_convertible_v<         const std::ranges::in_in_result<int, int>&, std::ranges::in_in_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<const std::ranges::in_in_result<int, int>&, std::ranges::in_in_result<long, long>>);
static_assert(std::is_convertible_v<         std::ranges::in_in_result<int, int>&&, std::ranges::in_in_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<std::ranges::in_in_result<int, int>&&, std::ranges::in_in_result<long, long>>);
static_assert(std::is_convertible_v<         const std::ranges::in_in_result<int, int>&&, std::ranges::in_in_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<const std::ranges::in_in_result<int, int>&&, std::ranges::in_in_result<long, long>>);

struct NotConvertible {};
static_assert(!std::is_convertible_v<std::ranges::in_in_result<NotConvertible, int>,
                                     std::ranges::in_in_result<int, int>>);
static_assert(!std::is_convertible_v<std::ranges::in_in_result<int, NotConvertible>,
                                     std::ranges::in_in_result<int, int>>);

constexpr bool test() {
  {
    std::ranges::in_in_result<int, double> res{10L, 0.};
    assert(res.in1 == 10);
    assert(res.in2 == 0.);
    std::ranges::in_in_result<ConstructibleFrom<int>, ConstructibleFrom<double>> res2 = res;
    assert(res2.in1.content == 10);
    assert(res2.in2.content == 0.);
  }
  {
    std::ranges::in_in_result<MoveOnly, int> res{MoveOnly{}, 0};
    assert(res.in1.get() == 1);
    [[maybe_unused]] auto res2 = static_cast<std::ranges::in_in_result<MoveOnly, int>>(std::move(res));
    assert(res.in1.get() == 0);
  }
  auto [in1, in2] = std::ranges::in_in_result<int, int>{1, 2};
  assert(in1 == 1);
  assert(in2 == 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
