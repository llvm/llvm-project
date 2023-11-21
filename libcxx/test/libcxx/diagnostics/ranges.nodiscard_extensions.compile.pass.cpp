//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that ranges algorithms aren't marked [[nodiscard]] when
// _LIBCPP_DISBALE_NODISCARD_EXT is defined

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

#include <algorithm>

void test() {
  int range[1];
  int* iter = range;
  auto pred = [](auto...) { return true; };
  std::ranges::adjacent_find(range);
  std::ranges::adjacent_find(iter, iter);
  std::ranges::all_of(range, pred);
  std::ranges::all_of(iter, iter, pred);
  std::ranges::any_of(range, pred);
  std::ranges::any_of(iter, iter, pred);
  std::ranges::binary_search(range, 1);
  std::ranges::binary_search(iter, iter, 1);
  std::ranges::clamp(1, 2, 3);
  std::ranges::count_if(range, pred);
  std::ranges::count_if(iter, iter, pred);
  std::ranges::count(range, 1);
  std::ranges::count(iter, iter, 1);
  std::ranges::equal_range(range, 1);
  std::ranges::equal_range(iter, iter, 1);
  std::ranges::equal(range, range);
  std::ranges::equal(iter, iter, iter, iter);
  std::ranges::find_end(range, range);
  std::ranges::find_end(iter, iter, iter, iter);
  std::ranges::find_first_of(range, range);
  std::ranges::find_first_of(iter, iter, iter, iter);
  std::ranges::find_if_not(range, pred);
  std::ranges::find_if_not(iter, iter, pred);
  std::ranges::find_if(range, pred);
  std::ranges::find_if(iter, iter, pred);
  std::ranges::find(range, 1);
  std::ranges::find(iter, iter, 1);
  std::ranges::includes(range, range);
  std::ranges::includes(iter, iter, iter, iter);
  std::ranges::is_heap_until(range);
  std::ranges::is_heap_until(iter, iter);
  std::ranges::is_heap(range);
  std::ranges::is_heap(iter, iter);
  std::ranges::is_partitioned(range, pred);
  std::ranges::is_partitioned(iter, iter, pred);
  std::ranges::is_permutation(range, range);
  std::ranges::is_permutation(iter, iter, iter, iter);
  std::ranges::is_sorted_until(range);
  std::ranges::is_sorted_until(iter, iter);
  std::ranges::is_sorted(range);
  std::ranges::is_sorted(iter, iter);
  std::ranges::lexicographical_compare(range, range);
  std::ranges::lexicographical_compare(iter, iter, iter, iter);
  std::ranges::lower_bound(range, 1);
  std::ranges::lower_bound(iter, iter, 1);
  std::ranges::max_element(range);
  std::ranges::max_element(iter, iter);
  std::ranges::max(1, 2);
  std::ranges::max({1, 2, 3});
  std::ranges::max(range);
  std::ranges::minmax_element(range);
  std::ranges::minmax_element(iter, iter);
  std::ranges::minmax(1, 2);
  std::ranges::minmax({1, 2, 3});
  std::ranges::minmax(range);
  std::ranges::mismatch(range, range);
  std::ranges::mismatch(iter, iter, iter, iter);
  std::ranges::none_of(range, pred);
  std::ranges::none_of(iter, iter, pred);
  std::ranges::remove_if(range, pred);
  std::ranges::remove_if(iter, iter, pred);
  std::ranges::remove(range, 1);
  std::ranges::remove(iter, iter, 1);
  std::ranges::search_n(range, 1, 1);
  std::ranges::search_n(iter, iter, 1, 1);
  std::ranges::search(range, range);
  std::ranges::search(iter, iter, iter, iter);
  std::ranges::unique(range);
  std::ranges::unique(iter, iter);
  std::ranges::upper_bound(range, 1);
  std::ranges::upper_bound(iter, iter, 1);
}
