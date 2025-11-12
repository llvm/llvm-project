//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>
//
// Algorithms that take predicates should support predicates that return a non-boolean value as long as the
// returned type is implicitly convertible to bool.

#include <algorithm>

#include <initializer_list>

#include "boolean_testable.h"

using Value    = StrictComparable<int>;
using Iterator = StrictBooleanIterator<Value*>;
auto pred1     = StrictUnaryPredicate;
auto pred2     = StrictBinaryPredicate;

void f(Iterator it, Iterator out, std::size_t n, Value const& val, std::initializer_list<Value> ilist) {
  (void)std::any_of(it, it, pred1);
  (void)std::all_of(it, it, pred1);
  (void)std::none_of(it, it, pred1);
  (void)std::find_if(it, it, pred1);
  (void)std::find_if_not(it, it, pred1);
  (void)std::find_first_of(it, it, it, it);
  (void)std::find_first_of(it, it, it, it, pred2);
  (void)std::adjacent_find(it, it);
  (void)std::adjacent_find(it, it, pred2);
  (void)std::mismatch(it, it, it, it);
  (void)std::mismatch(it, it, it, it, pred2);
  (void)std::mismatch(it, it, it);
  (void)std::mismatch(it, it, it);
  (void)std::mismatch(it, it, it, pred2);
  (void)std::equal(it, it, it, it);
  (void)std::equal(it, it, it, it, pred2);
  (void)std::equal(it, it, it);
  (void)std::equal(it, it, it, pred2);
  (void)std::lexicographical_compare(it, it, it, it);
  (void)std::lexicographical_compare(it, it, it, it, pred2);
  (void)std::partition_point(it, it, pred1);
  (void)std::lower_bound(it, it, val);
  (void)std::lower_bound(it, it, val, pred2);
  (void)std::upper_bound(it, it, val);
  (void)std::upper_bound(it, it, val, pred2);
  (void)std::equal_range(it, it, val);
  (void)std::equal_range(it, it, val, pred2);
  (void)std::binary_search(it, it, val);
  (void)std::binary_search(it, it, val, pred2);
  (void)std::min(val, val);
  (void)std::min(val, val, pred2);
  (void)std::min(ilist);
  (void)std::min(ilist, pred2);
  (void)std::max(val, val);
  (void)std::max(val, val, pred2);
  (void)std::max(ilist);
  (void)std::max(ilist, pred2);
  (void)std::minmax(val, val);
  (void)std::minmax(val, val, pred2);
  (void)std::minmax(ilist);
  (void)std::minmax(ilist, pred2);
  (void)std::min_element(it, it);
  (void)std::min_element(it, it, pred2);
  (void)std::max_element(it, it);
  (void)std::max_element(it, it, pred2);
  (void)std::minmax_element(it, it);
  (void)std::minmax_element(it, it, pred2);
  (void)std::count_if(it, it, pred1);
  (void)std::search(it, it, it, it);
  (void)std::search(it, it, it, it, pred2);
  (void)std::search_n(it, it, n, val);
  (void)std::search_n(it, it, n, val, pred2);
  (void)std::is_partitioned(it, it, pred1);
  (void)std::is_sorted(it, it);
  (void)std::is_sorted(it, it, pred2);
  (void)std::is_sorted_until(it, it);
  (void)std::is_sorted_until(it, it, pred2);
  (void)std::is_heap(it, it);
  (void)std::is_heap(it, it, pred2);
  (void)std::is_heap_until(it, it);
  (void)std::is_heap_until(it, it, pred2);
  (void)std::clamp(val, val, val);
  (void)std::clamp(val, val, val, pred2);
  (void)std::is_permutation(it, it, it, it);
  (void)std::is_permutation(it, it, it, it, pred2);
  (void)std::copy_if(it, it, out, pred1);
  (void)std::remove_copy_if(it, it, out, pred1);
  (void)std::remove_copy(it, it, out, val);
  (void)std::replace(it, it, val, val);
  (void)std::replace_if(it, it, pred1, val);
  (void)std::replace_copy_if(it, it, out, pred1, val);
  (void)std::replace_copy(it, it, out, val, val);
  (void)std::unique_copy(it, it, out, pred2);
  (void)std::partition_copy(it, it, out, out, pred1);
  (void)std::partial_sort_copy(it, it, it, it, pred2);
  (void)std::merge(it, it, it, it, out);
  (void)std::merge(it, it, it, it, out, pred2);
  (void)std::set_difference(it, it, it, it, out, pred2);
  (void)std::set_intersection(it, it, it, it, out, pred2);
  (void)std::set_symmetric_difference(it, it, it, it, out, pred2);
  (void)std::set_union(it, it, it, it, out, pred2);
  (void)std::remove_if(it, it, pred1);
  (void)std::remove(it, it, val);
  (void)std::unique(it, it, pred2);
  (void)std::partition(it, it, pred1);
  (void)std::stable_partition(it, it, pred1);
  (void)std::sort(it, it);
  (void)std::sort(it, it, pred2);
  (void)std::stable_sort(it, it);
  (void)std::stable_sort(it, it, pred2);
  (void)std::partial_sort(it, it, it);
  (void)std::partial_sort(it, it, it, pred2);
  (void)std::nth_element(it, it, it);
  (void)std::nth_element(it, it, it, pred2);
  (void)std::inplace_merge(it, it, it);
  (void)std::inplace_merge(it, it, it, pred2);
  (void)std::make_heap(it, it);
  (void)std::make_heap(it, it, pred2);
  (void)std::push_heap(it, it);
  (void)std::push_heap(it, it, pred2);
  (void)std::pop_heap(it, it);
  (void)std::pop_heap(it, it, pred2);
  (void)std::sort_heap(it, it);
  (void)std::sort_heap(it, it, pred2);
  (void)std::prev_permutation(it, it);
  (void)std::prev_permutation(it, it, pred2);
  (void)std::next_permutation(it, it);
  (void)std::next_permutation(it, it, pred2);
}
