//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test std::views::adjacent<N>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "../range_adaptor_types.h"

template <std::size_t N>
constexpr void test_constraints() {
  // needs to be a range
  static_assert(std::is_invocable_v<decltype((std::views::adjacent<N>)), std::ranges::empty_view<int>>);
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent<N>)), int>);

  // underlying needs to be forward_range
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent<N>)), InputCommonView>);
  static_assert(std::is_invocable_v<decltype((std::views::adjacent<N>)), ForwardSizedView>);
}

constexpr void test_pairwise_constraints() {
  // needs to be a range
  static_assert(std::is_invocable_v<decltype((std::views::pairwise)), std::ranges::empty_view<int>>);
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise)), int>);

  // underlying needs to be forward_range
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise)), InputCommonView>);
  static_assert(std::is_invocable_v<decltype((std::views::pairwise)), ForwardSizedView>);
}

constexpr void test_all_constraints() {
  test_pairwise_constraints();
  test_constraints<0>();
  test_constraints<1>();
  test_constraints<2>();
  test_constraints<3>();
  test_constraints<5>();
}

static constexpr void test_zero_case() {
  // N == 0 is a special case that always results in an empty range
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  std::same_as<std::ranges::empty_view<std::tuple<>>> decltype(auto) v =
      std::views::adjacent<0>(ContiguousCommonView{buffer});
  assert(std::ranges::size(v) == 0);
}

struct MoveOnlyView : ForwardSizedView {
  using ForwardSizedView::ForwardSizedView;

  constexpr MoveOnlyView(MoveOnlyView&&)      = default;
  constexpr MoveOnlyView(const MoveOnlyView&) = delete;

  constexpr MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  constexpr MoveOnlyView& operator=(const MoveOnlyView&) = delete;
};

template <std::size_t N>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  const auto validateBegin = [&](auto&& view) {
    auto it    = view.begin();
    auto tuple = *it;

    assert(std::get<0>(tuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buffer[2]);
    if constexpr (N >= 4)
      assert(std::get<3>(tuple) == buffer[3]);
    if constexpr (N >= 5)
      assert(std::get<4>(tuple) == buffer[4]);
  };

  // Test `views::adjacent<N>(r)`
  {
    using View                          = std::ranges::adjacent_view<ContiguousCommonView, N>;
    std::same_as<View> decltype(auto) v = std::views::adjacent<N>(ContiguousCommonView{buffer});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    validateBegin(v);
  }

  // Test `views::adjacent<N>(move only view)`
  {
    using View                          = std::ranges::adjacent_view<MoveOnlyView, N>;
    std::same_as<View> decltype(auto) v = std::views::adjacent<N>(MoveOnlyView{buffer});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    validateBegin(v);
  }

  // Test `r | views::adjacent<N>`
  {
    using View                          = std::ranges::adjacent_view<ContiguousCommonView, N>;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | std::views::adjacent<N>;
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    validateBegin(v);
  }

  // Test `move only view | views::adjacent<N>`
  {
    using View                          = std::ranges::adjacent_view<MoveOnlyView, N>;
    std::same_as<View> decltype(auto) v = MoveOnlyView{buffer} | std::views::adjacent<N>;
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    validateBegin(v);
  }

  // Test adjacent<N> | adjacent<N>
  {
    using View = std::ranges::adjacent_view<std::ranges::adjacent_view<ContiguousCommonView, N>, N>;

    auto twice                          = std::views::adjacent<N> | std::views::adjacent<N>;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | twice;
    assert(std::ranges::size(v) == (N <= 5 ? 10 - 2 * N : 0));

    if (std::ranges::size(v) == 0)
      return;

    auto it          = v.begin();
    auto nestedTuple = *it;

    auto innerFirstTuple = std::get<0>(nestedTuple);

    assert(std::get<0>(innerFirstTuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(innerFirstTuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(innerFirstTuple) == buffer[2]);
    if constexpr (N >= 4)
      assert(std::get<3>(innerFirstTuple) == buffer[3]);
    if constexpr (N >= 5)
      assert(std::get<4>(innerFirstTuple) == buffer[4]);

    if constexpr (N >= 2) {
      auto innerSecondTuple = std::get<1>(nestedTuple);
      assert(std::get<0>(innerSecondTuple) == buffer[1]);
      if constexpr (N >= 3)
        assert(std::get<1>(innerSecondTuple) == buffer[2]);
      if constexpr (N >= 4)
        assert(std::get<2>(innerSecondTuple) == buffer[3]);
      if constexpr (N >= 5)
        assert(std::get<3>(innerSecondTuple) == buffer[4]);
    }
  }
}

constexpr void test_pairwise() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // Test `views::pairwise(r)`
    using View                          = std::ranges::adjacent_view<ContiguousCommonView, 2>;
    std::same_as<View> decltype(auto) v = std::views::pairwise(ContiguousCommonView{buffer});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }

  {
    // Test `views::pairwise(move only view)`
    using View                          = std::ranges::adjacent_view<MoveOnlyView, 2>;
    std::same_as<View> decltype(auto) v = std::views::pairwise(MoveOnlyView{buffer});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test `r | views::pairwise`
    using View                          = std::ranges::adjacent_view<ContiguousCommonView, 2>;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | std::views::pairwise;
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test `move only view | views::pairwise`
    using View                          = std::ranges::adjacent_view<MoveOnlyView, 2>;
    std::same_as<View> decltype(auto) v = MoveOnlyView{buffer} | std::views::pairwise;
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test pairwise | pairwise
    using View = std::ranges::adjacent_view<std::ranges::adjacent_view<ContiguousCommonView, 2>, 2>;

    auto twice                          = std::views::pairwise | std::views::pairwise;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | twice;
    assert(std::ranges::size(v) == 6);

    auto it          = v.begin();
    auto nestedTuple = *it;

    auto innerFirstTuple = std::get<0>(nestedTuple);

    assert(std::get<0>(innerFirstTuple) == buffer[0]);
    assert(std::get<1>(innerFirstTuple) == buffer[1]);

    auto innerSecondTuple = std::get<1>(nestedTuple);
    assert(std::get<0>(innerSecondTuple) == buffer[1]);
    assert(std::get<1>(innerSecondTuple) == buffer[2]);
  }
}

constexpr bool test() {
  test_all_constraints();
  test_zero_case();
  test_pairwise();
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
