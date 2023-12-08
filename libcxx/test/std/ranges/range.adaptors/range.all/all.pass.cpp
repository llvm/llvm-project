//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::all;

#include <ranges>

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

int globalBuff[8];

template<bool IsNoexcept>
struct View : std::ranges::view_base {
  int start_ = 0;
  explicit View() noexcept(IsNoexcept) = default;
  constexpr explicit View(int start) : start_(start) {}
  View(View&&) noexcept(IsNoexcept) = default;
  View& operator=(View&&) noexcept(IsNoexcept) = default;
  constexpr int* begin() const { return globalBuff + start_; }
  constexpr int* end() const { return globalBuff + 8; }
};
static_assert(std::ranges::view<View<true>>);
static_assert(std::ranges::view<View<false>>);

template<bool IsNoexcept>
struct CopyableView : std::ranges::view_base {
  int start_ = 0;
  explicit CopyableView() noexcept(IsNoexcept) = default;
  CopyableView(CopyableView const&) noexcept(IsNoexcept) = default;
  CopyableView& operator=(CopyableView const&) noexcept(IsNoexcept) = default;
  constexpr explicit CopyableView(int start) noexcept : start_(start) {}
  constexpr int* begin() const { return globalBuff + start_; }
  constexpr int* end() const { return globalBuff + 8; }
};
static_assert(std::ranges::view<CopyableView<true>>);
static_assert(std::ranges::view<CopyableView<false>>);

struct MoveOnlyView : std::ranges::view_base{
  MoveOnlyView() = default;
  MoveOnlyView(const MoveOnlyView&) = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&) = default;
  MoveOnlyView& operator=(MoveOnlyView&&) = default;

  int* begin() const;
  int* end() const;
};

struct Range {
  int start_;
  constexpr explicit Range(int start) noexcept : start_(start) {}
  constexpr int* begin() const { return globalBuff + start_; }
  constexpr int* end() const { return globalBuff + 8; }
};

struct BorrowableRange {
  int start_;
  constexpr explicit BorrowableRange(int start) noexcept : start_(start) {}
  constexpr int* begin() const { return globalBuff + start_; }
  constexpr int* end() const { return globalBuff + 8; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct RandomAccessRange {
  constexpr auto begin() { return random_access_iterator<int*>(globalBuff); }
  constexpr auto end() { return sized_sentinel(random_access_iterator<int*>(globalBuff + 8)); }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<RandomAccessRange> = true;

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

constexpr bool test() {
  {
    ASSERT_SAME_TYPE(decltype(std::views::all(View<true>())), View<true>);
    static_assert(noexcept(std::views::all(View<true>())));
    static_assert(!noexcept(std::views::all(View<false>())));

    auto viewCopy = std::views::all(View<true>(2));
    ASSERT_SAME_TYPE(decltype(viewCopy), View<true>);
    assert(std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    ASSERT_SAME_TYPE(decltype(std::views::all(std::declval<const CopyableView<true>&>())), CopyableView<true>);
    static_assert(noexcept(std::views::all(CopyableView<true>())));
    static_assert(!noexcept(std::views::all(CopyableView<false>())));

    CopyableView<true> view(2);
    auto viewCopy = std::views::all(view);
    ASSERT_SAME_TYPE(decltype(viewCopy), CopyableView<true>);
    assert(std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    Range range(2);
    auto ref = std::views::all(range);
    ASSERT_SAME_TYPE(decltype(ref), std::ranges::ref_view<Range>);
    assert(std::ranges::begin(ref) == globalBuff + 2);
    assert(std::ranges::end(ref) == globalBuff + 8);

    auto own = std::views::all(std::move(range));
    ASSERT_SAME_TYPE(decltype(own), std::ranges::owning_view<Range>);
    assert(std::ranges::begin(own) == globalBuff + 2);
    assert(std::ranges::end(own) == globalBuff + 8);

    auto cref = std::views::all(std::as_const(range));
    ASSERT_SAME_TYPE(decltype(cref), std::ranges::ref_view<const Range>);
    assert(std::ranges::begin(cref) == globalBuff + 2);
    assert(std::ranges::end(cref) == globalBuff + 8);

    static_assert(!std::is_invocable_v<decltype(std::views::all), const Range&&>);
  }

  {
    auto own = std::views::all(BorrowableRange(2));
    ASSERT_SAME_TYPE(decltype(own), std::ranges::owning_view<BorrowableRange>);
    assert(std::ranges::begin(own) == globalBuff + 2);
    assert(std::ranges::end(own) == globalBuff + 8);
  }

  {
    auto own = std::views::all(RandomAccessRange());
    ASSERT_SAME_TYPE(decltype(own), std::ranges::owning_view<RandomAccessRange>);
    assert(base(std::ranges::begin(own)) == globalBuff);
    assert(base(base(std::ranges::end(own))) == globalBuff + 8);
  }

  // Check SFINAE friendliness of the call operator
  {
    static_assert(!std::is_invocable_v<decltype(std::views::all)>);
    static_assert(!std::is_invocable_v<decltype(std::views::all), RandomAccessRange, RandomAccessRange>);

    // `views::all(v)` is expression equivalent to `decay-copy(v)` if the decayed type
    // of `v` models `view`. If `v` is an lvalue-reference to a move-only view, the
    // expression should be ill-formed because `v` is not copyable
    static_assert(!std::is_invocable_v<decltype(std::views::all), MoveOnlyView&>);
  }

  // Test that std::views::all is a range adaptor
  {
    // Test `v | views::all`
    {
      Range range(0);
      auto result = range | std::views::all;
      ASSERT_SAME_TYPE(decltype(result), std::ranges::ref_view<Range>);
      assert(&result.base() == &range);
    }

    // Test `adaptor | views::all`
    {
      Range range(0);
      auto f = [](int i) { return i; };
      auto const partial = std::views::transform(f) | std::views::all;
      using Result = std::ranges::transform_view<std::ranges::ref_view<Range>, decltype(f)>;
      std::same_as<Result> auto result = partial(range);
      assert(&result.base().base() == &range);
    }

    // Test `views::all | adaptor`
    {
      Range range(0);
      auto f = [](int i) { return i; };
      auto const partial = std::views::all | std::views::transform(f);
      using Result = std::ranges::transform_view<std::ranges::ref_view<Range>, decltype(f)>;
      std::same_as<Result> auto result = partial(range);
      assert(&result.base().base() == &range);
    }

    {
      struct NotAView { };
      static_assert( CanBePiped<Range&,    decltype(std::views::all)>);
      static_assert(!CanBePiped<NotAView,  decltype(std::views::all)>);
    }
  }

  {
    static_assert(std::same_as<decltype(std::views::all), decltype(std::ranges::views::all)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
