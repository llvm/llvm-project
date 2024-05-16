//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <algorithm> functions are marked [[nodiscard]]

// clang-format off

#include <algorithm>
#include <functional>
#include <iterator>

#include "test_macros.h"

struct P {
  bool operator()(int) const { return false; }
};

void test() {
  int arr[1] = { 1 };

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::all_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1, std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::clamp(2, 1, 3);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::clamp(2, 1, 3, std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr),
             std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
             std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if_not(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_partitioned(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr), std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr), std::greater<int>());

#if TEST_STD_VER >= 20
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare_three_way(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare_three_way(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr), std::compare_three_way());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr),
                std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::none_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remove_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remove(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
              std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr),
              std::default_searcher(std::begin(arr), std::end(arr)));
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::unique(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::unique(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

#if TEST_STD_VER >= 20
  int range[1];
  int* iter = range;
  auto pred = [](auto...) { return true; };
  std::ranges::adjacent_find(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::adjacent_find(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::all_of(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::all_of(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::any_of(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::any_of(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::binary_search(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::binary_search(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::clamp(1, 2, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::count_if(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::count_if(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::count(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::count(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::equal_range(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::equal_range(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::equal(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::equal(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_end(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_end(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_first_of(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_first_of(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_if_not(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_if_not(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_if(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find_if(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::find(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::includes(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::includes(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_heap_until(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_heap_until(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_heap(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_heap(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_partitioned(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_partitioned(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_permutation(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_permutation(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_sorted_until(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_sorted_until(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_sorted(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::is_sorted(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::lexicographical_compare(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::lexicographical_compare(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::lower_bound(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::lower_bound(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::max_element(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::max_element(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::max(1, 2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::max({1, 2, 3}); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::max(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::minmax_element(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::minmax_element(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::minmax(1, 2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::minmax({1, 2, 3}); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::minmax(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::mismatch(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::mismatch(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::none_of(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::none_of(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::remove_if(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::remove_if(iter, iter, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::remove(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::remove(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::search_n(range, 1, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::search_n(iter, iter, 1, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::search(range, range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::search(iter, iter, iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::unique(range); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::unique(iter, iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::upper_bound(range, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::upper_bound(iter, iter, 1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 23
  std::ranges::contains(range, 1);
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::contains(iter, iter, 1);
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::contains_subrange(range, range);
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::contains_subrange(iter, iter, iter, iter);
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::fold_left(range, 0, std::plus());
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::fold_left(iter, iter, 0, std::plus());
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::fold_left_with_iter(range, 0, std::plus());
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::fold_left_with_iter(iter, iter, 0, std::plus());
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
