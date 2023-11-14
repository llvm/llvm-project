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
#include "__concepts/constructible.h"
#include "__iterator/concepts.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "test_iterators.h"
#include <ranges>
#include <type_traits>
#include <vector>

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

template <class T>
concept can_calculate_distance_between_non_sentinel = requires(T& __t) { __t - __t; };

constexpr bool operator_tests() {
  {
    // What operators are valid for an iterator derived from a stride view
    // over an input view.
    int arr[]  = {1, 2, 3};
    using View = InputView<cpp17_input_iterator<int*>>;
    auto rav   = View(cpp17_input_iterator(arr), cpp17_input_iterator(arr + 3));
    auto str   = std::ranges::stride_view<View>(rav, 1);

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
    int arr[]  = {1, 2, 3};
    using View = InputView<forward_iterator<int*>>;
    auto rav   = View(forward_iterator(arr), forward_iterator(arr + 3));
    auto str   = std::ranges::stride_view<View>(rav, 1);

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
    int arr[]  = {1, 2, 3};
    using View = InputView<bidirectional_iterator<int*>>;
    auto rav   = View(bidirectional_iterator(arr), bidirectional_iterator(arr + 3));
    auto str   = std::ranges::stride_view<View>(rav, 1);

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
    int arr[]  = {1, 2, 3};
    using View = InputView<random_access_iterator<int*>>;
    auto rav   = View(random_access_iterator(arr), random_access_iterator(arr + 3));
    auto str   = std::ranges::stride_view<View>(rav, 1);

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

  {
    // Test the non-forward-range operator- between two iterators (i.e., ceil).
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto rav_zero =
        InputView<SizedInputIterator, SizedInputIterator>(SizedInputIterator(arr), SizedInputIterator(arr + 10));
    auto rav_one =
        InputView<SizedInputIterator, SizedInputIterator>(SizedInputIterator(arr + 1), SizedInputIterator(arr + 10));
    auto stride_zoff = std::ranges::stride_view(rav_zero, 3);
    auto stride_ooff = std::ranges::stride_view(rav_one, 3);

    auto stride_zoff_base = stride_zoff.begin();
    auto stride_ooff_base = stride_ooff.begin();

    auto stride_zoff_one   = stride_zoff_base;
    auto stride_zoff_four  = ++stride_zoff_base;
    auto stride_zoff_seven = ++stride_zoff_base;

    auto stride_ooff_two  = stride_ooff_base;
    auto stride_ooff_five = ++stride_ooff_base;

    static_assert(std::sized_sentinel_for<decltype(std::move(stride_zoff_base).base()),
                                          decltype(std::move(stride_zoff_base).base())>);
    static_assert(can_calculate_distance_between_non_sentinel<decltype(stride_zoff_base)>);

    assert(*stride_zoff_one == 1);
    assert(*stride_zoff_four == 4);
    assert(*stride_zoff_seven == 7);

    assert(*stride_ooff_two == 2);
    assert(*stride_ooff_five == 5);

    // Check positive __n with exact multiple of left's stride.
    assert(stride_zoff_four - stride_zoff_one == 1);
    assert(stride_zoff_seven - stride_zoff_one == 2);
    // Check positive __n with non-exact multiple of left's stride.
    assert(stride_ooff_two - stride_zoff_one == 1);
    assert(stride_ooff_five - stride_zoff_one == 2);

    // Check negative __n with exact multiple of left's stride.
    assert(stride_zoff_one - stride_zoff_four == -1);
    assert(stride_zoff_one - stride_zoff_seven == -2);
    // Check negative __n with non-exact multiple of left's stride.
    assert(stride_zoff_one - stride_ooff_two == -1);
    assert(stride_zoff_one - stride_ooff_five == -2);
  }
  return true;
}

int main(int, char**) {
  operator_tests();
  static_assert(operator_tests());
  return 0;
}
