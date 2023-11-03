//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view::iterator

#include "../test.h"
#include "__iterator/concepts.h"
#include "__ranges/access.h"
#include <ranges>
#include <type_traits>

template <class T>
concept is_plus_equalable = requires(T& __t) { __t += 1; };
template <class T>
concept is_minus_equalable = requires(T& __t) { __t -= 1; };

template <class T>
concept is_plusable = requires(T& __t) { __t + 1; };
template <class T>
concept is_minusable = requires(T& __t) { __t - 1; };

template <class T>
concept is_relationally_comparable = requires(T& __t) {
  __t < __t;
  __t > __t;
  __t <= __t;
  __t >= __t;
};

template <class T>
concept is_plus_plusable_post = requires(T& __t) { __t++; };
template <class T>
concept is_plus_plusable_pre = requires(T& __t) { ++__t; };
template <class T>
concept is_minus_minusable_post = requires(T& __t) { __t--; };
template <class T>
concept is_minus_minusable_pre = requires(T& __t) { --__t; };

constexpr bool operator_tests() {
  {
    // What operators are valid for an iterator derived from a stride view
    // over an input view.
    int arr[] = {1, 2, 3};
    auto rav  = InputArrayView<int>(arr, arr + 3);
    auto str  = std::ranges::stride_view<InputArrayView<int>>(rav, 1);

    auto strb = str.begin();

    static_assert(is_plus_plusable_post<decltype(strb)>);
    static_assert(is_plus_plusable_pre<decltype(strb)>);
    static_assert(!is_minus_minusable_post<decltype(strb)>);
    static_assert(!is_minus_minusable_pre<decltype(strb)>);
    static_assert(!is_plus_equalable<decltype(strb)>);
    static_assert(!is_minus_equalable<decltype(strb)>);
    static_assert(!is_plusable<decltype(strb)>);
    static_assert(!is_minusable<decltype(strb)>);
    static_assert(!is_relationally_comparable<decltype(strb)>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a forward  view.
    int arr[] = {1, 2, 3};
    auto rav  = ForwardArrayView<int>(arr, arr + 3);
    auto str  = std::ranges::stride_view<ForwardArrayView<int>>(rav, 1);

    auto strb = str.begin();

    static_assert(is_plus_plusable_post<decltype(strb)>);
    static_assert(is_plus_plusable_pre<decltype(strb)>);
    static_assert(!is_minus_minusable_post<decltype(strb)>);
    static_assert(!is_minus_minusable_pre<decltype(strb)>);
    static_assert(!is_plus_equalable<decltype(strb)>);
    static_assert(!is_minus_equalable<decltype(strb)>);
    static_assert(!is_plusable<decltype(strb)>);
    static_assert(!is_minusable<decltype(strb)>);
    static_assert(!is_relationally_comparable<decltype(strb)>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a bidirectional view.
    int arr[] = {1, 2, 3};
    auto rav  = BidirArrayView<int>(arr, arr + 3);
    auto str  = std::ranges::stride_view<BidirArrayView<int>>(rav, 1);

    auto strb = str.begin();

    static_assert(is_plus_plusable_post<decltype(strb)>);
    static_assert(is_plus_plusable_pre<decltype(strb)>);
    static_assert(is_minus_minusable_post<decltype(strb)>);
    static_assert(is_minus_minusable_pre<decltype(strb)>);
    static_assert(!is_plus_equalable<decltype(strb)>);
    static_assert(!is_minus_equalable<decltype(strb)>);
    static_assert(!is_plusable<decltype(strb)>);
    static_assert(!is_minusable<decltype(strb)>);
    static_assert(!is_relationally_comparable<decltype(strb)>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a random access view.
    int arr[] = {1, 2, 3};
    auto rav  = RandomAccessArrayView<int>(arr, arr + 3);
    auto str  = std::ranges::stride_view<RandomAccessArrayView<int>>(rav, 1);

    auto strb = str.begin();

    static_assert(is_plus_plusable_post<decltype(strb)>);
    static_assert(is_plus_plusable_pre<decltype(strb)>);
    static_assert(is_minus_minusable_post<decltype(strb)>);
    static_assert(is_minus_minusable_pre<decltype(strb)>);
    static_assert(is_plus_equalable<decltype(strb)>);
    static_assert(is_minus_equalable<decltype(strb)>);
    static_assert(is_plusable<decltype(strb)>);
    static_assert(is_minusable<decltype(strb)>);
    static_assert(is_relationally_comparable<decltype(strb)>);
  }
  return true;
}

int main(int, char**) {
  operator_tests();
  static_assert(operator_tests());
  return 0;
}
