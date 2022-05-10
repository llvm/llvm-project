//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::views::drop

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <span>
#include <string_view>
#include <utility>
#include "test_iterators.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct SizedView : std::ranges::view_base {
  int* begin_ = nullptr;
  int* end_ = nullptr;
  constexpr SizedView(int* begin, int* end) : begin_(begin), end_(end) {}

  constexpr auto begin() const { return forward_iterator<int*>(begin_); }
  constexpr auto end() const { return sized_sentinel<forward_iterator<int*>>(forward_iterator<int*>(end_)); }
};
static_assert(std::ranges::forward_range<SizedView>);
static_assert(std::ranges::sized_range<SizedView>);
static_assert(std::ranges::view<SizedView>);

struct SizedViewWithUnsizedSentinel : std::ranges::view_base {
  using iterator = random_access_iterator<int*>;
  using sentinel = sentinel_wrapper<random_access_iterator<int*>>;

  int* begin_ = nullptr;
  int* end_ = nullptr;
  constexpr SizedViewWithUnsizedSentinel(int* begin, int* end) : begin_(begin), end_(end) {}

  constexpr auto begin() const { return iterator(begin_); }
  constexpr auto end() const { return sentinel(iterator(end_)); }
  constexpr size_t size() const { return end_ - begin_; }
};
static_assert(std::ranges::random_access_range<SizedViewWithUnsizedSentinel>);
static_assert(std::ranges::sized_range<SizedViewWithUnsizedSentinel>);
static_assert(!std::sized_sentinel_for<SizedViewWithUnsizedSentinel::sentinel, SizedViewWithUnsizedSentinel::iterator>);
static_assert(std::ranges::view<SizedViewWithUnsizedSentinel>);

template <class T>
constexpr void test_small_range(const T& input) {
  constexpr int N = 100;
  auto size = std::ranges::size(input);
  assert(size < N);

  auto result = input | std::views::drop(N);
  assert(result.empty());
}

constexpr bool test() {
  constexpr int N = 8;
  int buf[N] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test that `std::views::drop` is a range adaptor.
  {
    using SomeView = SizedView;

    // Test `view | views::drop`
    {
      SomeView view(buf, buf + N);
      std::same_as<std::ranges::drop_view<SomeView>> decltype(auto) result = view | std::views::drop(3);
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + N);
      assert(base(result.begin()) == buf + 3);
      assert(base(base(result.end())) == buf + N);
      assert(result.size() == 5);
    }

    // Test `adaptor | views::drop`
    {
      SomeView view(buf, buf + N);
      auto f = [](int i) { return i; };
      auto const partial = std::views::transform(f) | std::views::drop(3);

      using Result = std::ranges::drop_view<std::ranges::transform_view<SomeView, decltype(f)>>;
      std::same_as<Result> decltype(auto) result = partial(view);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(base(result.begin().base()) == buf + 3);
      assert(base(base(result.end().base())) == buf + N);
      assert(result.size() == 5);
    }

    // Test `views::drop | adaptor`
    {
      SomeView view(buf, buf + N);
      auto f = [](int i) { return i; };
      auto const partial = std::views::drop(3) | std::views::transform(f);

      using Result = std::ranges::transform_view<std::ranges::drop_view<SomeView>, decltype(f)>;
      std::same_as<Result> decltype(auto) result = partial(view);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(base(result.begin().base()) == buf + 3);
      assert(base(base(result.end().base())) == buf + N);
      assert(result.size() == 5);
    }

    // Check SFINAE friendliness
    {
      struct NotAView { };
      static_assert(!std::is_invocable_v<decltype(std::views::drop)>);
      static_assert(!std::is_invocable_v<decltype(std::views::drop), NotAView, int>);
      static_assert( CanBePiped<SomeView&,   decltype(std::views::drop(3))>);
      static_assert( CanBePiped<int(&)[10],  decltype(std::views::drop(3))>);
      static_assert(!CanBePiped<int(&&)[10], decltype(std::views::drop(3))>);
      static_assert(!CanBePiped<NotAView,    decltype(std::views::drop(3))>);

      static_assert(!CanBePiped<SomeView&,   decltype(std::views::drop(/*n=*/NotAView{}))>);
    }
  }

  {
    static_assert(std::same_as<decltype(std::views::drop), decltype(std::ranges::views::drop)>);
  }

  // `views::drop(empty_view, n)` returns an `empty_view`.
  {
    using Result = std::ranges::empty_view<int>;
    [[maybe_unused]] std::same_as<Result> decltype(auto) result = std::views::empty<int> | std::views::drop(3);
  }

  // `views::drop(span, n)` returns a `span`.
  {
    std::span<int> s(buf);
    std::same_as<decltype(s)> decltype(auto) result = s | std::views::drop(5);
    assert(result.size() == 3);
  }

  // `views::drop(span, n)` returns a `span` with a dynamic extent, regardless of the input `span`.
  {
    std::span<int, 8> s(buf);
    std::same_as<std::span<int, std::dynamic_extent>> decltype(auto) result = s | std::views::drop(3);
    assert(result.size() == 5);
  }

  // `views::drop(string_view, n)` returns a `string_view`.
  {
    {
      std::string_view sv = "abcdef";
      std::same_as<decltype(sv)> decltype(auto) result = sv | std::views::drop(2);
      assert(result.size() == 4);
    }

    {
      std::u32string_view sv = U"abcdef";
      std::same_as<decltype(sv)> decltype(auto) result = sv | std::views::drop(2);
      assert(result.size() == 4);
    }
  }

  // `views::drop(iota_view, n)` returns an `iota_view`.
  {
    auto iota = std::views::iota(1, 8);
    // The second template argument of the resulting `iota_view` is different because it has to be able to hold
    // the `range_difference_t` of the input `iota_view`.
    using Result = std::ranges::iota_view<int, int>;
    std::same_as<Result> decltype(auto) result = iota | std::views::drop(3);
    assert(result.size() == 4);
    assert(*result.begin() == 4);
    assert(*std::ranges::next(result.begin(), 3) == 7);
  }

  // `views::drop(subrange, n)` returns a `subrange` when `subrange::StoreSize == false`.
  {
    auto subrange = std::ranges::subrange(buf, buf + N);
    LIBCPP_STATIC_ASSERT(!decltype(subrange)::_StoreSize);

    using Result = std::ranges::subrange<int*>;
    std::same_as<Result> decltype(auto) result = subrange | std::views::drop(3);
    assert(result.size() == 5);
  }

  // `views::drop(subrange, n)` returns a `subrange` when `subrange::StoreSize == true`.
  {
    using View = SizedViewWithUnsizedSentinel;
    View view(buf, buf + N);

    using Subrange = std::ranges::subrange<View::iterator, View::sentinel, std::ranges::subrange_kind::sized>;
    auto subrange = Subrange(view.begin(), view.end(), std::ranges::distance(view.begin(), view.end()));
    LIBCPP_STATIC_ASSERT(decltype(subrange)::_StoreSize);

    std::same_as<Subrange> decltype(auto) result = subrange | std::views::drop(3);
    assert(result.size() == 5);
  }

  // `views::drop(subrange, n)` doesn't return a `subrange` if it's not a random access range.
  {
    SizedView v(buf, buf + N);
    auto subrange = std::ranges::subrange(v.begin(), v.end());

    using Result = std::ranges::drop_view<std::ranges::subrange<forward_iterator<int*>,
        sized_sentinel<forward_iterator<int*>>>>;
    std::same_as<Result> decltype(auto) result = subrange | std::views::drop(3);
    assert(result.size() == 5);
  }

  // When the size of the input range `s` is shorter than `n`, an `empty_view` is returned.
  {
    test_small_range(std::span(buf));
    test_small_range(std::string_view("abcdef"));
    test_small_range(std::ranges::subrange(buf, buf + N));
    test_small_range(std::views::iota(1, 8));
  }

  // Test that it's possible to call `std::views::drop` with any single argument as long as the resulting closure is
  // never invoked. There is no good use case for it, but it's valid.
  {
    struct X { };
    [[maybe_unused]] auto partial = std::views::drop(X{});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
