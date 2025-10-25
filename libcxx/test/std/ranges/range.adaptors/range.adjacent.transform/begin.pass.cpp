//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin();
// constexpr auto begin() const  requires range<const InnerView> &&
//              regular_invocable<const F&, REPEAT(range_reference_t<const V>, N)...>

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "helpers.h"
#include "../range_adaptor_types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

struct NoConstBeginView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct OnlyNonConstFn {
  template <class... T>
    requires((!std::is_const_v<std::remove_reference_t<T>>) && ...)
  int operator()(T&&...) const;
};

template <class Range, class Fn, std::size_t N, class Validator>
constexpr void test_one() {
  using View = std::ranges::adjacent_transform_view<Range, Fn, N>;
  Validator validator{};
  {
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    View v(Range{buffer}, Fn{});

    auto it = v.begin();
    validator(buffer, *it, 0);

    auto cit = std::as_const(v).begin();
    validator(buffer, *cit, 0);
  }

  {
    // empty range
    std::array<int, 0> buffer = {};
    View v(Range{buffer.data(), 0}, Fn{});
    auto it  = v.begin();
    auto cit = std::as_const(v).begin();
    assert(it == v.end());
    assert(cit == std::as_const(v).end());
  }

  if constexpr (N > 2) {
    // N greater than range size
    int buffer[2] = {1, 2};
    View v(Range{buffer}, Fn{});
    auto it  = v.begin();
    auto cit = std::as_const(v).begin();
    assert(it == v.end());
    assert(cit == std::as_const(v).end());
  }
}

template <std::size_t N, class Fn, class Validator>
constexpr void test_simple() {
  test_one<SimpleCommon, Fn, N, Validator>();

  using View = std::ranges::adjacent_transform_view<SimpleCommon, MakeTuple, N>;

  // non-const begin always exists and return iterator<false>
  static_assert(!std::is_same_v<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
}

template <std::size_t N, class Fn, class Validator>
constexpr void test_non_simple() {
  test_one<NonSimpleCommon, Fn, N, Validator>();

  using View = std::ranges::adjacent_transform_view<NonSimpleCommon, MakeTuple, N>;
  static_assert(!std::is_same_v<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
}

template <std::size_t N>
constexpr void test() {
  test_simple<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test_simple<N, Tie, ValidateTieFromIndex<N>>();
  test_simple<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test_simple<N, Multiply, ValidateMultiplyFromIndex<N>>();

  test_non_simple<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test_non_simple<N, Tie, ValidateTieFromIndex<N>>();
  test_non_simple<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test_non_simple<N, Multiply, ValidateMultiplyFromIndex<N>>();

  // Test with view that doesn't support const begin()
  using ViewWithNoConstBegin = std::ranges::adjacent_transform_view<NoConstBeginView, MakeTuple, N>;
  static_assert(HasBegin<ViewWithNoConstBegin>);
  static_assert(!HasConstBegin<ViewWithNoConstBegin>);

  using OnlyNonConstView = std::ranges::adjacent_transform_view<NonSimpleCommon, OnlyNonConstFn, N>;
  static_assert(HasBegin<OnlyNonConstView>);
  static_assert(!HasConstBegin<OnlyNonConstView>);
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
