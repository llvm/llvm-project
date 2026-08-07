//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// std::views::enumerate;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "types.h"

// Concepts

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

template <class Range>
concept CanEnumerate = requires(Range&& range) { std::views::enumerate(std::forward<Range>(range)); };

// Helpers

struct ImmovableReference {
  ImmovableReference(ImmovableReference&&) = delete;

  operator int();
};

struct IteratorWithImmovableReferences {
  using value_type      = int;
  using difference_type = std::ptrdiff_t;

  ImmovableReference operator*() const;
  IteratorWithImmovableReferences& operator++();
  void operator++(int);
  bool operator==(std::default_sentinel_t) const;
};

static_assert(std::input_iterator<IteratorWithImmovableReferences>);

using NonEnumeratableRangeWithImmmovabaleReferences =
    std::ranges::subrange<IteratorWithImmovableReferences, std::default_sentinel_t>;

static_assert(std::ranges::input_range<NonEnumeratableRangeWithImmmovabaleReferences>);
static_assert(!CanEnumerate<NonEnumeratableRangeWithImmmovabaleReferences>);
static_assert(!std::move_constructible<std::ranges::range_reference_t<NonEnumeratableRangeWithImmmovabaleReferences>>);
static_assert(
    !std::move_constructible<std::ranges::range_rvalue_reference_t<NonEnumeratableRangeWithImmmovabaleReferences>>);

template <typename View, typename T>
using ExpectedViewElement = std::tuple<typename std::ranges::iterator_t<View>::difference_type, T>;

// Helpers

template <typename View, typename T = int>
constexpr void compareViews(View v, std::initializer_list<ExpectedViewElement<View, T>> list) {
  assert(std::ranges::equal(v, list));
}

// Test SFINAE friendliness

static_assert(CanBePiped<RangeView, decltype(std::views::enumerate)>);

static_assert(CanEnumerate<RangeView>);

static_assert(!std::is_invocable_v<decltype(std::views::enumerate)>);
static_assert(std::is_invocable_v<decltype(std::views::enumerate), RangeView>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), NotAView>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), NotInvocable>);

static_assert(std::is_same_v<decltype(std::ranges::views::enumerate), decltype(std::views::enumerate)>);

struct MoveOnlyView : std::ranges::view_base {
  constexpr explicit MoveOnlyView(int* b, int* e) : begin_(b), end_(e) {}
  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  constexpr MoveOnlyView(MoveOnlyView&& other) noexcept
      : begin_(std::exchange(other.begin_, nullptr)), end_(std::exchange(other.end_, nullptr)) {}
  MoveOnlyView& operator=(MoveOnlyView&&) = default;

  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_ = nullptr;
  int* end_   = nullptr;
};

static_assert(std::ranges::view<MoveOnlyView>);
static_assert(!std::copy_constructible<MoveOnlyView>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), MoveOnlyView&>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), const MoveOnlyView&>);
static_assert(std::is_invocable_v<decltype(std::views::enumerate), MoveOnlyView>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), const MoveOnlyView>);

constexpr bool test() {
  // Test `views::enumerate_view(v)`
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<RangeView>;
    RangeView const range(buff, buff + 4);

    std::same_as<Result> decltype(auto) result = std::views::enumerate(range);
    compareViews(result, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  }
  {
    std::string_view sv{"babazmt"};
    using Result = std::ranges::enumerate_view<std::string_view>;

    std::same_as<Result> decltype(auto) result = std::views::enumerate(sv);
    compareViews(result, {{0, 'b'}, {1, 'a'}, {2, 'b'}, {3, 'a'}, {4, 'z'}, {5, 'm'}, {6, 't'}});
  }

  // Test `adaptor | views::enumerate`
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<RangeView>;
    RangeView const range(buff, buff + 4);

    std::same_as<Result> decltype(auto) result = range | std::views::enumerate;
    compareViews(result, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  }
  {
    std::string_view sv{"babazmt"};
    using Result = std::ranges::enumerate_view<std::string_view>;

    std::same_as<Result> decltype(auto) result = sv | std::views::enumerate;
    compareViews(result, {{0, 'b'}, {1, 'a'}, {2, 'b'}, {3, 'a'}, {4, 'z'}, {5, 'm'}, {6, 't'}});
  }

  {
    int buff[]{2, 3, 5, 7};
    using Result = std::ranges::enumerate_view<MoveOnlyView>;
    {
      std::same_as<Result> decltype(auto) result = std::views::enumerate(MoveOnlyView(buff, buff + 4));
      compareViews(std::move(result), {{0, 2}, {1, 3}, {2, 5}, {3, 7}});
    }
    {
      std::same_as<Result> decltype(auto) result = MoveOnlyView(buff, buff + 4) | std::views::enumerate;
      compareViews(std::move(result), {{0, 2}, {1, 3}, {2, 5}, {3, 7}});
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
