//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// adjacent_transform_view() = default;

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

#include "helpers.h"

constexpr int buff[] = {1, 2, 3, 4, 5};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + std::ranges::size(buff)) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultCtrView : std::ranges::view_base {
  NoDefaultCtrView() = delete;
  int* begin() const;
  int* end() const;
};

static_assert(
    std::is_default_constructible_v<std::ranges::adjacent_transform_view<DefaultConstructibleView, MakeTuple, 1>>);
static_assert(std::is_default_constructible_v<std::ranges::adjacent_transform_view<DefaultConstructibleView, Tie, 2>>);
static_assert(
    std::is_default_constructible_v<std::ranges::adjacent_transform_view<DefaultConstructibleView, GetFirst, 3>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_transform_view<NoDefaultCtrView, MakeTuple, 1>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_transform_view<NoDefaultCtrView, Tie, 2>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_transform_view<NoDefaultCtrView, GetFirst, 3>>);

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  {
    using View = std::ranges::adjacent_transform_view<DefaultConstructibleView, Fn, N>;
    View v     = View(); // the default constructor is not explicit
    assert(v.size() == std::ranges::size(buff) - (N - 1));
    Validator validator{};
    auto it = v.begin();
    validator(buff, *it, 0);
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test<N, Tie, ValidateTieFromIndex<N>>();
  test<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test<N, Multiply, ValidateMultiplyFromIndex<N>>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
