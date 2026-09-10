//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that the algorithms can be instantiated with volatile pointers.
//
// A pointer to volatile is a valid Cpp17InputIterator, so algorithms that only require an input iterator are
// required by the Standard to work with one. It is however not a valid Cpp17ForwardIterator, since its
// reference type (volatile T&) is neither value_type& nor const value_type&. Supporting the algorithms that
// require a forward iterator or better is an extension, which we provide on a best-effort basis.

// UNSUPPORTED: c++03

#include <algorithm>
#include <functional>
#include <numeric>

#include "test_macros.h"

#if TEST_STD_VER >= 20
#  include <ranges>
#  include <span>
#endif

// Named functions instead of lambdas, to keep the diagnostics readable.
inline bool is_positive(int x) { return x > 0; }
inline int identity(int x) { return x; }
inline int plus(int x, int y) { return x + y; }
inline int zero() { return 0; }
inline void observe(int) {}

using It = int volatile*;

void test(It first,
          It mid,
          It last,
          It first2,
          It last2,
          int* pfirst,
          int* plast,
          int value,
          int volatile& volatile_value,
          char volatile* cfirst,
          char volatile* clast,
          char* pcfirst,
          char* pclast,
          char cvalue) {
  // Reading from a volatile range.
  (void)std::find(first, last, value);
  (void)std::find(first, last, volatile_value);
  (void)std::find(pfirst, plast, volatile_value); // plain range, volatile value
  (void)std::find_if(first, last, is_positive);
  (void)std::find_if_not(first, last, is_positive);
  (void)std::count(first, last, value);
  (void)std::count_if(first, last, is_positive);
  (void)std::all_of(first, last, is_positive);
  (void)std::any_of(first, last, is_positive);
  (void)std::none_of(first, last, is_positive);
  (void)std::for_each(first, last, observe);
  (void)std::adjacent_find(first, last);
  (void)std::search(first, last, first2, last2);
  (void)std::search_n(first, last, 2, value);
  (void)std::find_end(first, last, first2, last2);
  (void)std::find_first_of(first, last, first2, last2);
  (void)std::min_element(first, last);
  (void)std::max_element(first, last);
  (void)std::minmax_element(first, last);
  (void)std::is_sorted(first, last);
  (void)std::is_sorted_until(first, last);
  (void)std::is_partitioned(first, last, is_positive);
  (void)std::partition_point(first, last, is_positive);
  (void)std::lower_bound(first, last, value);
  (void)std::upper_bound(first, last, value);
  (void)std::equal_range(first, last, value);
  (void)std::binary_search(first, last, value);
  (void)std::is_heap(first, last);
  (void)std::is_heap_until(first, last);
  (void)std::equal(first, last, first2);                          // volatile against volatile
  (void)std::equal(first, last, pfirst);                          // volatile against plain
  (void)std::mismatch(first, last, pfirst);                       //
  (void)std::lexicographical_compare(first, last, first2, last2); //
  (void)std::lexicographical_compare(first, last, pfirst, plast); //
  (void)std::is_permutation(first, last, first2);
  (void)std::includes(first, last, first2, last2);
#if TEST_STD_VER >= 14
  (void)std::equal(first, last, first2, last2);
  (void)std::equal(first, last, pfirst, plast);
  (void)std::mismatch(first, last, pfirst, plast);
#endif

  // Writing into a volatile range.
  (void)std::copy(pfirst, plast, first);
  (void)std::copy_n(pfirst, 3, first);
  (void)std::copy_backward(pfirst, plast, last);
  (void)std::copy_if(pfirst, plast, first, is_positive);
  (void)std::move(pfirst, plast, first);
  (void)std::move_backward(pfirst, plast, last);
  (void)std::fill(first, last, value);
  (void)std::fill_n(first, 3, value);
  (void)std::generate(first, last, zero);
  (void)std::generate_n(first, 3, zero);
  (void)std::transform(pfirst, plast, first, identity);
  (void)std::transform(pfirst, plast, pfirst, first, plus);
  (void)std::replace_copy(pfirst, plast, first, value, value);
  (void)std::replace_copy_if(pfirst, plast, first, is_positive, value);
  (void)std::remove_copy(pfirst, plast, first, value);
  (void)std::remove_copy_if(pfirst, plast, first, is_positive);
  (void)std::unique_copy(pfirst, plast, first);
  (void)std::reverse_copy(pfirst, plast, first);
  (void)std::rotate_copy(pfirst, pfirst + 1, plast, first);
  (void)std::merge(pfirst, plast, pfirst, plast, first);
  (void)std::set_union(pfirst, plast, pfirst, plast, first);
  (void)std::set_intersection(pfirst, plast, pfirst, plast, first);
  (void)std::set_difference(pfirst, plast, pfirst, plast, first);
  (void)std::set_symmetric_difference(pfirst, plast, pfirst, plast, first);
  (void)std::partition_copy(pfirst, plast, first, first, is_positive);
  (void)std::partial_sort_copy(pfirst, plast, first, last);

  // Reading a volatile range into a non-volatile one.
  (void)std::copy(first, last, pfirst);
  (void)std::copy_backward(first, last, plast);
  (void)std::copy_if(first, last, pfirst, is_positive);
  (void)std::move(first, last, pfirst);
  (void)std::transform(first, last, pfirst, identity);
  (void)std::remove_copy(first, last, pfirst, value);
  (void)std::unique_copy(first, last, pfirst);
  (void)std::reverse_copy(first, last, pfirst);
  (void)std::merge(first, last, first, last, pfirst);

  // Modifying a volatile range in place.
  (void)std::copy(first, last, first);
  (void)std::transform(first, last, first, identity);
  (void)std::replace(first, last, value, value);
  (void)std::replace_if(first, last, is_positive, value);

  // Permuting a volatile range in place.
  (void)std::swap_ranges(first, mid, mid);
  (void)std::iter_swap(first, mid);
  (void)std::reverse(first, last);
  (void)std::rotate(first, mid, last);
  (void)std::remove(first, last, value);
  (void)std::remove_if(first, last, is_positive);
  (void)std::unique(first, last);
  (void)std::partition(first, last, is_positive);
  (void)std::stable_partition(first, last, is_positive);
  (void)std::next_permutation(first, last);
  (void)std::prev_permutation(first, last);

  // Sorting and heaps.
  (void)std::sort(first, last);
  (void)std::stable_sort(first, last);
  (void)std::partial_sort(first, mid, last);
  (void)std::nth_element(first, mid, last);
  (void)std::inplace_merge(first, mid, last);
  (void)std::make_heap(first, last);
  (void)std::push_heap(first, last);
  (void)std::pop_heap(first, last);
  (void)std::sort_heap(first, last);
#if TEST_STD_VER >= 14
  (void)std::sort(first, last, std::greater<>());
#endif

  // Exercise the memchr/memcmp based implementations.
  (void)std::find(cfirst, clast, cvalue);
  (void)std::count(cfirst, clast, cvalue);
  (void)std::equal(cfirst, clast, pcfirst);
  (void)std::mismatch(cfirst, clast, pcfirst);
  (void)std::lexicographical_compare(cfirst, clast, pcfirst, pclast);
  (void)std::copy(pcfirst, pclast, cfirst);
  (void)std::copy(cfirst, clast, pcfirst);
  (void)std::fill(cfirst, clast, cvalue);

  // <numeric>
  (void)std::accumulate(first, last, 0, plus);
  (void)std::inner_product(first, last, first, 0);
  (void)std::inner_product(first, last, pfirst, 0);
  (void)std::iota(first, last, 0);
  (void)std::partial_sum(first, last, pfirst);
  (void)std::partial_sum(pfirst, plast, first);
  (void)std::adjacent_difference(first, last, pfirst);
  (void)std::adjacent_difference(pfirst, plast, first);
#if TEST_STD_VER >= 17
  (void)std::reduce(first, last, 0);
  (void)std::transform_reduce(first, last, 0, plus, identity);
  (void)std::transform_reduce(first, last, pfirst, 0);
  (void)std::inclusive_scan(first, last, pfirst);
  (void)std::inclusive_scan(pfirst, plast, first);
  (void)std::exclusive_scan(first, last, pfirst, 0);
  (void)std::exclusive_scan(pfirst, plast, first, 0);
  (void)std::transform_inclusive_scan(first, last, pfirst, plus, identity);
  (void)std::transform_exclusive_scan(first, last, pfirst, 0, plus, identity);
#endif

#if TEST_STD_VER >= 20
  namespace ranges = std::ranges;

  (void)ranges::find(first, last, value);
  (void)ranges::find_if(first, last, is_positive);
  (void)ranges::count(first, last, value);
  (void)ranges::for_each(first, last, observe);
  (void)ranges::equal(first, last, first, last);
  (void)ranges::mismatch(first, last, pfirst, plast);
  (void)ranges::min_element(first, last);
  (void)ranges::lower_bound(first, last, value);
  (void)ranges::fill(first, last, value);
  (void)ranges::copy(pfirst, plast, first);
  (void)ranges::copy(first, last, pfirst);
  (void)ranges::transform(first, last, first, identity);
  (void)ranges::remove(first, last, value);
  (void)ranges::unique(first, last);
  (void)ranges::make_heap(first, last);

  (void)ranges::swap_ranges(first, mid, mid, last);
  (void)ranges::reverse(first, last);
  (void)ranges::rotate(first, mid, last);
  (void)ranges::partition(first, last, is_positive);
  (void)ranges::sort(first, last);
  (void)ranges::stable_sort(first, last);
  (void)ranges::nth_element(first, mid, last);

  ranges::subrange range(first, last);
  static_assert(ranges::contiguous_range<decltype(range)>);
  (void)ranges::count(range, value);
  (void)ranges::fill(range, value);
  (void)ranges::for_each(range | std::views::take(2), observe);
  (void)ranges::distance(range);
  (void)ranges::sort(range);

  std::span<int volatile> span(first, last);
  static_assert(ranges::contiguous_range<decltype(span)>);
  (void)ranges::count(span, value);
  (void)ranges::sort(span);
#endif
}
