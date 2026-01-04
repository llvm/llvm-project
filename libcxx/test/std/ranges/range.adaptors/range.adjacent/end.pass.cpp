//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end() requires(!simple-view<_View>)
// constexpr auto end() const requires range<const _View>

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../range_adaptor_types.h"

template <class Underlying, std::size_t N>
constexpr void test_one() {
  {
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::ranges::adjacent_view<Underlying, N> v{Underlying{buffer}};

    auto it     = v.begin();
    auto cit    = std::as_const(v).begin();
    auto endIt  = v.end();
    auto cendIt = std::as_const(v).end();
    assert(it != endIt);
    assert(cit != cendIt);
    std::ranges::advance(it, 8 - (N - 1));
    std::ranges::advance(cit, 8 - (N - 1));
    assert(it == endIt);
    assert(cit == cendIt);
  }
  {
    // empty range
    std::array<int, 0> buffer = {};
    std::ranges::adjacent_view<Underlying, N> v{Underlying{buffer.data(), 0}};
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
    std::ranges::adjacent_view<Underlying, N> v{Underlying{buffer}};
    auto it     = v.begin();
    auto cit    = std::as_const(v).begin();
    auto endIt  = v.end();
    auto cendIt = std::as_const(v).end();
    assert(it == endIt);
    assert(cit == cendIt);
  }
}

template <class Underlying, std::size_t N>
constexpr void test_decrement_back() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::adjacent_view<Underlying, N> v{Underlying{buffer}};
  auto endIt  = v.end();
  auto cendIt = std::as_const(v).end();

  --endIt;
  --cendIt;
  auto tuple  = *endIt;
  auto ctuple = *cendIt;
  assert(std::get<0>(tuple) == buffer[8 - N]);
  assert(std::get<0>(ctuple) == buffer[8 - N]);
  if constexpr (N >= 2) {
    assert(std::get<1>(tuple) == buffer[9 - N]);
    assert(std::get<1>(ctuple) == buffer[9 - N]);
  }
  if constexpr (N >= 3) {
    assert(std::get<2>(tuple) == buffer[10 - N]);
    assert(std::get<2>(ctuple) == buffer[10 - N]);
  }
  if constexpr (N >= 4) {
    assert(std::get<3>(tuple) == buffer[11 - N]);
    assert(std::get<3>(ctuple) == buffer[11 - N]);
  }
  if constexpr (N >= 5) {
    assert(std::get<4>(tuple) == buffer[12 - N]);
    assert(std::get<4>(ctuple) == buffer[12 - N]);
  }
}

template <std::size_t N>
constexpr void test() {
  {
    // simple common
    using NonConstView = std::ranges::adjacent_view<SimpleCommon, N>;
    using ConstView    = const NonConstView;

    static_assert(std::ranges::common_range<NonConstView>);
    static_assert(std::ranges::common_range<ConstView>);
    static_assert(std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

    test_one<SimpleCommon, N>();
    test_decrement_back<SimpleCommon, N>();
  }
  {
    // simple non common

    using NonConstView = std::ranges::adjacent_view<SimpleNonCommon, N>;
    using ConstView    = const NonConstView;

    static_assert(!std::ranges::common_range<NonConstView>);
    static_assert(!std::ranges::common_range<ConstView>);
    static_assert(std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

    test_one<SimpleNonCommon, N>();
  }

  {
    // non simple common
    using NonConstView = std::ranges::adjacent_view<NonSimpleCommon, N>;
    using ConstView    = const NonConstView;

    static_assert(std::ranges::common_range<NonConstView>);
    static_assert(std::ranges::common_range<ConstView>);
    static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

    test_one<NonSimpleCommon, N>();
    test_decrement_back<NonSimpleCommon, N>();
  }
  {
    // non simple non common
    using NonConstView = std::ranges::adjacent_view<NonSimpleNonCommon, N>;
    using ConstView    = const NonConstView;

    static_assert(!std::ranges::common_range<NonConstView>);
    static_assert(!std::ranges::common_range<ConstView>);
    static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

    test_one<NonSimpleNonCommon, N>();
  }
  {
    // forward only
    using NonConstView = std::ranges::adjacent_view<NonSimpleForwardSizedNonCommon, N>;
    using ConstView    = const NonConstView;

    static_assert(!std::ranges::common_range<NonConstView>);
    static_assert(!std::ranges::common_range<ConstView>);
    static_assert(!std::is_same_v<std::ranges::sentinel_t<NonConstView>, std::ranges::sentinel_t<ConstView>>);

    test_one<NonSimpleForwardSizedNonCommon, N>();
    test_one<ForwardSizedView, N>();
  }
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
