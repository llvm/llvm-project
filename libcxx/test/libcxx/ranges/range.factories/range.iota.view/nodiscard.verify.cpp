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

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
struct IntSentinelWith {
  using difference_type = std::iter_difference_t<T>;

  T value_;
  constexpr explicit IntSentinelWith(T value = T()) : value_(value) {}

  friend constexpr bool operator==(IntSentinelWith lhs, IntSentinelWith rhs) { return lhs.value_ == rhs.value_; }
  friend constexpr bool operator==(IntSentinelWith lhs, T rhs) { return lhs.value_ == rhs; }
  friend constexpr bool operator==(T lhs, IntSentinelWith rhs) { return lhs == rhs.value_; }

  friend constexpr IntSentinelWith operator+(IntSentinelWith lhs, IntSentinelWith rhs) {
    return IntSentinelWith{lhs.value_ + rhs.value_};
  }
  friend constexpr difference_type operator-(IntSentinelWith lhs, IntSentinelWith rhs) {
    return lhs.value_ - rhs.value_;
  }
  friend constexpr difference_type operator-(IntSentinelWith lhs, T rhs) { return lhs.value_ - rhs; }
  friend constexpr difference_type operator-(T lhs, IntSentinelWith rhs) { return lhs - rhs.value_; }

  constexpr IntSentinelWith& operator++() {
    ++value_;
    return *this;
  }
  constexpr IntSentinelWith operator++(int) {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  constexpr IntSentinelWith operator--() {
    --value_;
    return *this;
  }
};
template <class T>
IntSentinelWith(T) -> IntSentinelWith<T>;
static_assert(std::sized_sentinel_for<IntSentinelWith<random_access_iterator<int*>>, random_access_iterator<int*>>);

void test() {
  {
    // [range.iota.view]

    auto view          = std::views::iota(49, 94);
    auto unboundedView = std::views::iota(82);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    unboundedView.end();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.empty();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.size();

    // [range.iota.iterator]

    auto it = view.begin();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[82];

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;
  }

  {
    // [range.iota.sentinel]

    int buffer[]{94, 82, 47};
    auto outIter = random_access_iterator<int*>(buffer);
    std::ranges::iota_view<random_access_iterator<int*>, IntSentinelWith<random_access_iterator<int*>>> view{
        outIter, IntSentinelWith<random_access_iterator<int*>>(std::ranges::next(outIter, 1))};

    auto it = view.begin();
    auto st = view.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - st;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - it;
  }

  {
    // [range.iota.overview]

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::iota(1);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::iota(1, 10);

#if _LIBCPP_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::indices(5);
#endif
  }
}
