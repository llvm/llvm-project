//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <span>
#include <vector>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

void test() {
  // [range.drop.view]

  {
    std::vector<int> range;

    auto v = std::views::drop(range, 0);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(v).base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).begin();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.end();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.size();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).size();
  }

  // [range.drop.overview]

  { // The `empty_view` case.
    auto emptyView = std::views::empty<int>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(emptyView, 0);
  }

  { // The `span | basic_string_view | iota_view | subrange (StoreSize == false)` case.
    int arr[]{94, 82, 49};
    auto subrange = std::ranges::subrange(arr, arr + 3);
    LIBCPP_STATIC_ASSERT(!decltype(subrange)::_StoreSize);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(subrange, 3);
  }

  { // The the `subrange (StoreSize == true)` case.
    struct SizedViewWithUnsizedSentinel : std::ranges::view_base {
      using iterator = random_access_iterator<int*>;
      using sentinel = sentinel_wrapper<random_access_iterator<int*>>;

      int* begin_ = nullptr;
      int* end_   = nullptr;
      constexpr SizedViewWithUnsizedSentinel(int* begin, int* end) : begin_(begin), end_(end) {}

      constexpr auto begin() const { return iterator(begin_); }
      constexpr auto end() const { return sentinel(iterator(end_)); }
      constexpr std::size_t size() const { return end_ - begin_; }
    };
    static_assert(std::ranges::random_access_range<SizedViewWithUnsizedSentinel>);
    static_assert(std::ranges::sized_range<SizedViewWithUnsizedSentinel>);
    static_assert(
        !std::sized_sentinel_for<SizedViewWithUnsizedSentinel::sentinel, SizedViewWithUnsizedSentinel::iterator>);
    static_assert(std::ranges::view<SizedViewWithUnsizedSentinel>);

    int arr[]{94, 82, 49};

    using View = SizedViewWithUnsizedSentinel;
    View view{arr, arr + 3};

    using Subrange = std::ranges::subrange<View::iterator, View::sentinel, std::ranges::subrange_kind::sized>;
    auto subrange  = Subrange(view.begin(), view.end(), std::ranges::distance(view.begin(), view.end()));
    LIBCPP_STATIC_ASSERT(decltype(subrange)::_StoreSize);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(subrange, 3);
  }

#if TEST_STD_VER >= 23
  { // The `repeat_view` "_RawRange models sized_range" case.
    auto repeatView = std::ranges::repeat_view<int, int>(1, 8);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(repeatView, 3);
  }

  { // The `repeat_view` "otherwise" case.
    auto repeatView = std::ranges::repeat_view<int>(1);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(repeatView, 3);
  }
#endif

  { // The "otherwise" case.
    struct SizedView : std::ranges::view_base {
      int* begin_ = nullptr;
      int* end_   = nullptr;
      constexpr SizedView(int* begin, int* end) : begin_(begin), end_(end) {}

      constexpr auto begin() const { return forward_iterator<int*>(begin_); }
      constexpr auto end() const { return sized_sentinel<forward_iterator<int*>>(forward_iterator<int*>(end_)); }
    };
    static_assert(std::ranges::forward_range<SizedView>);
    static_assert(std::ranges::sized_range<SizedView>);
    static_assert(std::ranges::view<SizedView>);

    int arr[]{94, 82, 49};

    SizedView view{arr, arr + 3};
    auto subrange = std::ranges::subrange(view.begin(), view.end());

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(subrange, 3);
  }

  {
    struct X {};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::drop(X{});
  }
}
