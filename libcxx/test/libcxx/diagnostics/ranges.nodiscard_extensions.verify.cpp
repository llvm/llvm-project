//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that ranges algorithms are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>

#include "test_macros.h"

void test() {
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

#if TEST_STD_VER >= 23
  std::ranges::contains(range, 1);
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::contains(iter, iter, 1);
  // expected-warning@-1{{ignoring return value of function declared with 'nodiscard' attribute}}
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
