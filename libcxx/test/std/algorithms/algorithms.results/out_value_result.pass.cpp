//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, C++20

//  template <class O, class T>
//    struct out_value_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

//
// Helper structs
//

// only explicit construction
struct IterTypeExplicit {
  explicit IterTypeExplicit(int*);
};

// implicit construction
struct IterTypeImplicit {
  IterTypeImplicit(int*);
};

struct IterTypeImplicitRef {
  IterTypeImplicitRef(int&);
};

struct NotConvertible {};

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

//
constexpr void test_constraints() {
  // requires convertible_to<const _OutIter1&, _OutIter2> && convertible_to<const _ValType1&, _ValType2>
  static_assert(
      std::is_constructible_v<std::ranges::out_value_result<int*, int>, std::ranges::out_value_result<int*, int>>);

  // test failure when implicit conversion isn't allowed
  static_assert(!std::is_constructible_v<std::ranges::out_value_result<IterTypeExplicit, int>,
                                         std::ranges::out_value_result<int*, int>>);

  // test success when implicit conversion is allowed, checking combinations of value, reference, and const
  static_assert(std::is_constructible_v<std::ranges::out_value_result<IterTypeImplicit, int>,
                                        std::ranges::out_value_result<int*, int>>);
  static_assert(std::is_constructible_v<std::ranges::out_value_result<IterTypeImplicit, int>,
                                        std::ranges::out_value_result<int*, int> const>);
  static_assert(std::is_constructible_v<std::ranges::out_value_result<IterTypeImplicit, int>,
                                        std::ranges::out_value_result<int*, int>&>);
  static_assert(std::is_constructible_v<std::ranges::out_value_result<IterTypeImplicit, int>,
                                        std::ranges::out_value_result<int*, int> const&>);

  static_assert(!std::is_constructible_v<std::ranges::out_value_result<IterTypeImplicitRef, int>,
                                         std::ranges::out_value_result<int, int>&>);

  // has to be convertible via const&
  static_assert(
      std::is_convertible_v<std::ranges::out_value_result<int, int>&, std::ranges::out_value_result<long, long>>);
  static_assert(
      std::is_convertible_v<const std::ranges::out_value_result<int, int>&, std::ranges::out_value_result<long, long>>);
  static_assert(
      std::is_convertible_v<std::ranges::out_value_result<int, int>&&, std::ranges::out_value_result<long, long>>);
  static_assert(std::is_convertible_v<const std::ranges::out_value_result<int, int>&&,
                                      std::ranges::out_value_result<long, long>>);

  // should be move constructible
  static_assert(std::is_move_constructible_v<std::ranges::out_value_result<MoveOnly, int>>);
  static_assert(std::is_move_constructible_v<std::ranges::out_value_result<int, MoveOnly>>);

  // conversions should not work if there is no conversion
  static_assert(!std::is_convertible_v<std::ranges::out_value_result<NotConvertible, int>,
                                       std::ranges::out_value_result<int, NotConvertible>>);
  static_assert(!std::is_convertible_v<std::ranges::out_value_result<int, NotConvertible>,
                                       std::ranges::out_value_result<NotConvertible, int>>);
}

// Test results
constexpr bool test() {
  {
    std::ranges::out_value_result<double, int> res{10, 1};
    assert(res.out == 10);
    assert(res.value == 1);
    std::ranges::out_value_result<ConvertibleFrom<double>, ConvertibleFrom<int>> res2 = res;
    assert(res2.out.content == 10);
    assert(res2.value.content == 1);
  }
  {
    std::ranges::out_value_result<MoveOnly, int> res{MoveOnly{}, 10};
    assert(res.out.get() == 1);
    assert(res.value == 10);
    auto res2 = std::move(res);
    assert(res.out.get() == 0);
    assert(res.value == 10);
    assert(res2.out.get() == 1);
    assert(res2.value == 10);
  }
  {
    auto [out, val] = std::ranges::out_value_result<int, int>{1, 2};
    assert(out == 1);
    assert(val == 2);
  }

  return true;
}

int main(int, char**) {
  test_constraints();
  test();
  static_assert(test());
  return 0;
}