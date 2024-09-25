//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class I>
// struct in_found_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

struct A {
  explicit A(int);
};
// no implicit conversion
static_assert(!std::is_constructible_v<std::ranges::in_found_result<A>, std::ranges::in_found_result<int>>);

struct B {
  B(const int&);
  B(int&&);
};
// implicit conversion
static_assert(std::is_constructible_v<std::ranges::in_found_result<B>, std::ranges::in_found_result<int>>);
static_assert(std::is_constructible_v<std::ranges::in_found_result<B>, std::ranges::in_found_result<int>&>);
static_assert(std::is_constructible_v<std::ranges::in_found_result<B>, const std::ranges::in_found_result<int>>);
static_assert(std::is_constructible_v<std::ranges::in_found_result<B>, const std::ranges::in_found_result<int>&>);

struct C {
  C(int&);
};
static_assert(!std::is_constructible_v<std::ranges::in_found_result<C>, std::ranges::in_found_result<int>&>);

// has to be convertible via const&
static_assert(std::is_convertible_v<std::ranges::in_found_result<int>&, std::ranges::in_found_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::in_found_result<int>&, std::ranges::in_found_result<long>>);
static_assert(std::is_convertible_v<std::ranges::in_found_result<int>&&, std::ranges::in_found_result<long>>);
static_assert(std::is_convertible_v<const std::ranges::in_found_result<int>&&, std::ranges::in_found_result<long>>);

// should be move constructible
static_assert(std::is_move_constructible_v<std::ranges::in_found_result<MoveOnly>>);

// should not be copy constructible
static_assert(!std::is_copy_constructible_v<std::ranges::in_found_result<MoveOnly>>);

struct NotConvertible {};
// conversions should not work if there is no conversion
static_assert(!std::is_convertible_v<std::ranges::in_found_result<NotConvertible>, std::ranges::in_found_result<int>>);

static_assert(std::is_same_v<decltype(std::ranges::in_found_result<int>::in), int>);
static_assert(std::is_same_v<decltype(std::ranges::in_found_result<int>::found), bool>);

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

constexpr bool test() {
  // Checks that conversion operations are correct.
  {
    std::ranges::in_found_result<double> res{10, true};
    assert(res.in == 10);
    assert(res.found == true);
    std::ranges::in_found_result<ConvertibleFrom<int>> res2 = res;
    assert(res2.in.content == 10);
    assert(res2.found == true);
  }

  // Checks that conversions are possible when one of the types is move-only.
  {
    std::ranges::in_found_result<MoveOnly> res{MoveOnly{}, false};
    assert(res.in.get() == 1);
    assert(!res.found);
    auto res2 = std::move(res);
    assert(res2.in.get() == 1);
    assert(!res2.found);
    assert(res.in.get() == 0);
    assert(!res.found);
  }

  // Checks that structured bindings get the correct values.
  {
    auto [in, found] = std::ranges::in_found_result<int>{2, false};
    assert(in == 2);
    assert(!found);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
