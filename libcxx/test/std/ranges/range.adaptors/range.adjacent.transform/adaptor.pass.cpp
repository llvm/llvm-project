//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test std::views::adjacent_transform<N>

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "helpers.h"
#include "../range_adaptor_types.h"

struct Fn {
  int operator()(auto...) const;
};

struct VoidFn {
  void operator()(auto...) const;
};

struct StringFn {
  template <class... Ts>
    requires(sizeof...(Ts) > 0 && (std::is_same_v<Ts, std::string> && ...))
  int operator()(Ts...) const;
};

template <std::size_t N>
constexpr void test_constraints() {
  // needs to be a range
  static_assert(std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), std::ranges::empty_view<int>, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), int, Fn>);

  // underlying needs to be forward_range
  static_assert(std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), ForwardSizedView, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), InputCommonView, Fn>);

  // function needs to be callable with N range references
  static_assert(std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), ForwardSizedView, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), ForwardSizedView, StringFn>);

  // results need to be referenceable
  static_assert(std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), ForwardSizedView, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::adjacent_transform<N>)), ForwardSizedView, VoidFn>);
}

constexpr void test_pairwise_transform_constraints() {
  // needs to be a range
  static_assert(std::is_invocable_v<decltype((std::views::pairwise_transform)), std::ranges::empty_view<int>, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise_transform)), int, Fn>);

  // underlying needs to be forward_range
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise_transform)), InputCommonView, Fn>);
  static_assert(std::is_invocable_v<decltype((std::views::pairwise_transform)), ForwardSizedView, Fn>);

  // function needs to be callable with N range references
  static_assert(std::is_invocable_v<decltype((std::views::pairwise_transform)), ForwardSizedView, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise_transform)), ForwardSizedView, StringFn>);

  // results need to be referenceable
  static_assert(std::is_invocable_v<decltype((std::views::pairwise_transform)), ForwardSizedView, Fn>);
  static_assert(!std::is_invocable_v<decltype((std::views::pairwise_transform)), ForwardSizedView, VoidFn>);
}

constexpr void test_all_constraints() {
  test_pairwise_transform_constraints();
  test_constraints<0>();
  test_constraints<1>();
  test_constraints<2>();
  test_constraints<3>();
  test_constraints<5>();
}

constexpr void test_zero_case() {
  // N == 0 is a special case that always results in an empty range
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  std::same_as<std::ranges::empty_view<int>> decltype(auto) v =
      std::views::adjacent_transform<0>(ContiguousCommonView{buffer}, Fn{});
  assert(std::ranges::size(v) == 0);
}

struct MoveOnlyView : ForwardSizedView {
  using ForwardSizedView::ForwardSizedView;

  constexpr MoveOnlyView(MoveOnlyView&&)      = default;
  constexpr MoveOnlyView(const MoveOnlyView&) = delete;

  constexpr MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  constexpr MoveOnlyView& operator=(const MoveOnlyView&) = delete;
};

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  Validator validator{};

  // Test `views::adjacent_transform<N>(r)`
  {
    using View                          = std::ranges::adjacent_transform_view<ContiguousCommonView, Fn, N>;
    std::same_as<View> decltype(auto) v = std::views::adjacent_transform<N>(ContiguousCommonView{buffer}, Fn{});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    auto it = v.begin();
    validator(buffer, *it, 0);
  }

  // Test `views::adjacent_transform<N>(move only view)`
  {
    using View                          = std::ranges::adjacent_transform_view<MoveOnlyView, Fn, N>;
    std::same_as<View> decltype(auto) v = std::views::adjacent_transform<N>(MoveOnlyView{buffer}, Fn{});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    auto it = v.begin();
    validator(buffer, *it, 0);
  }

  // Test `r | views::adjacent_transform<N>`
  {
    using View                          = std::ranges::adjacent_transform_view<ContiguousCommonView, Fn, N>;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | std::views::adjacent_transform<N>(Fn{});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    auto it = v.begin();
    validator(buffer, *it, 0);
  }

  // Test `move only view | views::adjacent_transform<N>`
  {
    using View                          = std::ranges::adjacent_transform_view<MoveOnlyView, Fn, N>;
    std::same_as<View> decltype(auto) v = MoveOnlyView{buffer} | std::views::adjacent_transform<N>(Fn{});
    assert(std::ranges::size(v) == (N <= 8 ? 9 - N : 0));
    auto it = v.begin();
    validator(buffer, *it, 0);
  }

  // Test adjacent_transform<N> | adjacent_transform<N>
  {
    using View = std::ranges::
        adjacent_transform_view<std::ranges::adjacent_transform_view<ContiguousCommonView, MakeTuple, N>, MakeTuple, N>;

    auto twice = std::views::adjacent_transform<N>(MakeTuple{}) | std::views::adjacent_transform<N>(MakeTuple{});
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

constexpr void test_pairwise_transform() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // Test `views::pairwise_transform(r)`
    using View                          = std::ranges::adjacent_transform_view<ContiguousCommonView, MakeTuple, 2>;
    std::same_as<View> decltype(auto) v = std::views::pairwise_transform(ContiguousCommonView{buffer}, MakeTuple{});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }

  {
    // Test `views::pairwise_transform(move only view)`
    using View                          = std::ranges::adjacent_transform_view<MoveOnlyView, MakeTuple, 2>;
    std::same_as<View> decltype(auto) v = std::views::pairwise_transform(MoveOnlyView{buffer}, MakeTuple{});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test `r | views::pairwise_transform`
    using View                          = std::ranges::adjacent_transform_view<ContiguousCommonView, MakeTuple, 2>;
    std::same_as<View> decltype(auto) v = ContiguousCommonView{buffer} | std::views::pairwise_transform(MakeTuple{});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test `move only view | views::pairwise_transform`
    using View                          = std::ranges::adjacent_transform_view<MoveOnlyView, MakeTuple, 2>;
    std::same_as<View> decltype(auto) v = MoveOnlyView{buffer} | std::views::pairwise_transform(MakeTuple{});
    assert(std::ranges::size(v) == 7);
    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    assert(std::get<1>(tuple) == buffer[1]);
  }
  {
    // Test pairwise_transform | pairwise_transform
    using View = std::ranges::
        adjacent_transform_view<std::ranges::adjacent_transform_view<ContiguousCommonView, MakeTuple, 2>, MakeTuple, 2>;

    auto twice = std::views::pairwise_transform(MakeTuple{}) | std::views::pairwise_transform(MakeTuple{});
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

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test<N, Tie, ValidateTieFromIndex<N>>();
  test<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test<N, Multiply, ValidateMultiplyFromIndex<N>>();
}

constexpr bool test() {
  test_all_constraints();
  test_zero_case();
  test_pairwise_transform();
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
