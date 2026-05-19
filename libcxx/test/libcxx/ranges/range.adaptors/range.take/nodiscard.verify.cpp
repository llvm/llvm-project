//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <ranges>

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <vector>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

void test() {
  // [range.take.view]

  {
    std::vector<int> range;

    auto v = std::views::take(range, 0);

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

  // [range.take.sentinel]

  {
    struct MoveOnlyView : std::ranges::view_base {
      int* ptr_;

      constexpr explicit MoveOnlyView(int* ptr) : ptr_(ptr) {}
      MoveOnlyView(MoveOnlyView&&)            = default;
      MoveOnlyView& operator=(MoveOnlyView&&) = default;

      constexpr int* begin() const { return ptr_; }
      constexpr sentinel_wrapper<int*> end() const { return sentinel_wrapper<int*>{ptr_ + 8}; }
    };

    int arr[]{94, 82, 49};
    auto v = std::views::take(MoveOnlyView{arr}, 0);

    auto st = v.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(st).base();
  }

  // [range.take.overview]

  { // The `empty_view` case.
    auto emptyView = std::views::empty<int>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(emptyView, 0);
  }

  { // The `span | basic_string_view | subrange` case.
    int arr[]{94, 82, 49};
    auto subrange = std::ranges::subrange(arr, arr + 3);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(subrange, 3);
  }

  { // The `iota_view` case.
    auto iota = std::views::iota(94, 47);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(iota, 3);
  }

#if TEST_STD_VER >= 23
  { // The `repeat_view` "_RawRange models sized_range" case.
    auto repeat = std::ranges::repeat_view<int, int>(94, 8);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(repeat, 3);
  }

  { // The `repeat_view` "otherwise" case.

    auto repeat = std::ranges::repeat_view<int>(94);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(repeat, 3);
  }
#endif // TEST_STD_VER >= 23

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
    std::views::take(subrange, 3);
  }

  {
    struct X {};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::take(X{});
  }
}
