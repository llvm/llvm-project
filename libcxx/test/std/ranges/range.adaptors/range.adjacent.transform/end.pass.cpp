//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end()
// constexpr auto end() const requires range<const InnerView> &&
//             regular_invocable<const F&, REPEAT(range_reference_t<const V>, N)...>

#include <array>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../range_adaptor_types.h"
#include "helpers.h"

template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

struct NoConstEndView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct OnlyNonConstFn {
  template <class... T>
    requires((!std::is_const_v<std::remove_reference_t<T>>) && ...)
  int operator()(T&&...) const;
};

template <class R, class Fn, std::size_t N>
constexpr void test_one() {
  {
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::adjacent_transform_view<R, Fn, N> v{R{buffer}, Fn{}};

    auto it     = v.begin();
    auto cit    = std::as_const(v).begin();
    auto endIt  = v.end();
    auto cendIt = std::as_const(v).end();
    assert(it != endIt);
    assert(cit != cendIt);
    assert(it + (8 - (N - 1)) == endIt);
    assert(cit + (8 - (N - 1)) == cendIt);
  }
  {
    // empty range
    std::array<int, 0> buffer = {};
    std::ranges::adjacent_transform_view<R, Fn, N> v{R{buffer.data(), 0}, Fn{}};
    auto it     = v.begin();
    auto cit    = std::as_const(v).begin();
    auto endIt  = v.end();
    auto cendIt = std::as_const(v).end();
    assert(it == endIt);
    assert(cit == cendIt);
  }
  if constexpr (N > 2) {
    // N greater than range size
    int buffer[2] = {1, 2};
    std::ranges::adjacent_transform_view<R, Fn, N> v{R{buffer}, Fn{}};
    auto it     = v.begin();
    auto cit    = std::as_const(v).begin();
    auto endIt  = v.end();
    auto cendIt = std::as_const(v).end();
    assert(it == endIt);
    assert(cit == cendIt);
  }
}

template <std::size_t N, class Fn>
constexpr void test_simple_common_types() {
  using NonConstView = std::ranges::adjacent_transform_view<SimpleCommon, Fn, N>;
  using ConstView    = const NonConstView;

  static_assert(std::ranges::common_range<NonConstView>);
  static_assert(std::ranges::common_range<ConstView>);

  // non-const end always exists and returns iterator<false>
  static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

  test_one<SimpleCommon, Fn, N>();
}

template <std::size_t N, class Fn>
constexpr void test_simple_non_common_types() {
  using NonConstView = std::ranges::adjacent_transform_view<SimpleNonCommon, Fn, N>;
  using ConstView    = const NonConstView;

  static_assert(!std::ranges::common_range<NonConstView>);
  static_assert(!std::ranges::common_range<ConstView>);

  // non-const end always exists and returns sentinel<false>
  static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

  test_one<SimpleNonCommon, Fn, N>();
}

template <std::size_t N, class Fn>
constexpr void test_non_simple_common_types() {
  using NonConstView = std::ranges::adjacent_transform_view<NonSimpleCommon, Fn, N>;
  using ConstView    = const NonConstView;

  static_assert(std::ranges::common_range<NonConstView>);
  static_assert(std::ranges::common_range<ConstView>);
  static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

  test_one<NonSimpleCommon, Fn, N>();
}

template <std::size_t N, class Fn>
constexpr void test_non_simple_non_common_types() {
  using NonConstView = std::ranges::adjacent_transform_view<NonSimpleNonCommon, Fn, N>;
  using ConstView    = const NonConstView;

  static_assert(!std::ranges::common_range<NonConstView>);
  static_assert(!std::ranges::common_range<ConstView>);
  static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

  test_one<NonSimpleNonCommon, Fn, N>();
}

template <std::size_t N, class Fn>
constexpr void test_forward_only() {
  using NonConstView = std::ranges::adjacent_transform_view<NonSimpleForwardSizedNonCommon, Fn, N>;
  using ConstView    = const NonConstView;

  static_assert(!std::ranges::common_range<NonConstView>);
  static_assert(!std::ranges::common_range<ConstView>);
  static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

  test_one<NonSimpleForwardSizedNonCommon, Fn, N>();
}

template <std::size_t N, class Fn>
constexpr void test() {
  test_simple_common_types<N, Fn>();
  test_simple_non_common_types<N, Fn>();
  test_non_simple_common_types<N, Fn>();
  test_non_simple_non_common_types<N, Fn>();
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();

  // Test with view that doesn't support const end()
  using ViewWithNoConstBegin = std::ranges::adjacent_transform_view<NoConstEndView, MakeTuple, N>;
  static_assert(HasEnd<ViewWithNoConstBegin>);
  static_assert(!HasConstEnd<ViewWithNoConstBegin>);

  using OnlyNonConstView = std::ranges::adjacent_transform_view<NonSimpleCommon, OnlyNonConstFn, N>;
  static_assert(HasEnd<OnlyNonConstView>);
  static_assert(!HasConstEnd<OnlyNonConstView>);
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
