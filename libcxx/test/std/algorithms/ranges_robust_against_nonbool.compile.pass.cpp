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
// Range algorithms that take predicates should support predicates that return a non-boolean value as long as the
// returned type is implicitly convertible to bool.

#include <algorithm>

#include <initializer_list>
#include <ranges>

#include "boolean_testable.h"
#include "test_macros.h"

using Value     = StrictComparable<int>;
using Iterator  = StrictBooleanIterator<Value*>;
using Range     = std::ranges::subrange<Iterator>;
auto pred1      = StrictUnaryPredicate;
auto pred2      = StrictBinaryPredicate;
auto projection = [](Value const& val) -> Value { return val; };

void f(Iterator it, Range in, Iterator out, std::size_t n, Value const& val, std::initializer_list<Value> ilist) {
  // Functions of the form (in, pred)
  auto in_pred = [&](auto func, auto pred) {
    (void)func(it, it, pred);
    (void)func(in, pred);
    (void)func(it, it, pred, projection);
    (void)func(in, pred, projection);
  };

  // Functions of the form (in, in, pred)
  auto in_in_pred = [&](auto func, auto pred) {
    (void)func(it, it, it, it, pred);
    (void)func(in, in, pred);
    (void)func(it, it, it, it, pred, projection);
    (void)func(in, in, pred, projection);
  };

  in_pred(std::ranges::any_of, pred1);
  in_pred(std::ranges::all_of, pred1);
#if TEST_STD_VER >= 23
  in_in_pred(std::ranges::ends_with, pred2);
#endif
  in_pred(std::ranges::none_of, pred1);
  in_pred(std::ranges::find_if, pred1);
  in_pred(std::ranges::find_if_not, pred1);
#if TEST_STD_VER >= 23
  in_pred(std::ranges::find_last_if, pred1);
  in_pred(std::ranges::find_last_if_not, pred1);
#endif
  in_in_pred(std::ranges::find_first_of, pred2);
  in_pred(std::ranges::adjacent_find, pred2);
  in_in_pred(std::ranges::mismatch, pred2);
  in_in_pred(std::ranges::equal, pred2);
  in_in_pred(std::ranges::lexicographical_compare, pred2);
  in_pred(std::ranges::partition_point, pred1);
  // lower_bound
  {
    (void)std::ranges::lower_bound(it, it, val, pred2);
    (void)std::ranges::lower_bound(in, val, pred2);
    (void)std::ranges::lower_bound(it, it, val, pred2, projection);
    (void)std::ranges::lower_bound(in, val, pred2, projection);
  }
  // upper_bound
  {
    (void)std::ranges::upper_bound(it, it, val, pred2);
    (void)std::ranges::upper_bound(in, val, pred2);
    (void)std::ranges::upper_bound(it, it, val, pred2, projection);
    (void)std::ranges::upper_bound(in, val, pred2, projection);
  }
  // equal_range
  {
    (void)std::ranges::equal_range(it, it, val, pred2);
    (void)std::ranges::equal_range(in, val, pred2);
    (void)std::ranges::equal_range(it, it, val, pred2, projection);
    (void)std::ranges::equal_range(in, val, pred2, projection);
  }
  // binary_search
  {
    (void)std::ranges::binary_search(it, it, val, pred2);
    (void)std::ranges::binary_search(in, val, pred2);
    (void)std::ranges::binary_search(it, it, val, pred2, projection);
    (void)std::ranges::binary_search(in, val, pred2, projection);
  }
  // min
  {
    (void)std::ranges::min(val, val, pred2);
    (void)std::ranges::min(val, val, pred2, projection);
    (void)std::ranges::min(ilist, pred2);
    (void)std::ranges::min(ilist, pred2, projection);
    (void)std::ranges::min(in, pred2);
    (void)std::ranges::min(in, pred2, projection);
  }
  // max
  {
    (void)std::ranges::max(val, val, pred2);
    (void)std::ranges::max(val, val, pred2, projection);
    (void)std::ranges::max(ilist, pred2);
    (void)std::ranges::max(ilist, pred2, projection);
    (void)std::ranges::max(in, pred2);
    (void)std::ranges::max(in, pred2, projection);
  }
  // minmax
  {
    (void)std::ranges::minmax(val, val, pred2);
    (void)std::ranges::minmax(val, val, pred2, projection);
    (void)std::ranges::minmax(ilist, pred2);
    (void)std::ranges::minmax(ilist, pred2, projection);
    (void)std::ranges::minmax(in, pred2);
    (void)std::ranges::minmax(in, pred2, projection);
  }

  in_pred(std::ranges::min_element, pred2);
  in_pred(std::ranges::max_element, pred2);
  in_pred(std::ranges::minmax_element, pred2);
  in_pred(std::ranges::count_if, pred1);
  in_in_pred(std::ranges::search, pred2);
  // search_n
  {
    (void)std::ranges::search_n(it, it, n, val, pred2);
    (void)std::ranges::search_n(in, n, val, pred2);
    (void)std::ranges::search_n(it, it, n, val, pred2, projection);
    (void)std::ranges::search_n(in, n, val, pred2, projection);
  }
  in_in_pred(std::ranges::find_end, pred2);
  in_pred(std::ranges::is_partitioned, pred1);
  in_pred(std::ranges::is_sorted, pred2);
  in_pred(std::ranges::is_sorted_until, pred2);
  in_in_pred(std::ranges::includes, pred2);
  in_pred(std::ranges::is_heap, pred2);
  in_pred(std::ranges::is_heap_until, pred2);
  // clamp
  {
    (void)std::ranges::clamp(val, val, val);
    (void)std::ranges::clamp(val, val, val, pred2);
    (void)std::ranges::clamp(val, val, val, pred2, projection);
  }
  in_in_pred(std::ranges::is_permutation, pred2);
  // copy_if
  {
    (void)std::ranges::copy_if(it, it, out, pred1);
    (void)std::ranges::copy_if(in, out, pred1);
    (void)std::ranges::copy_if(it, it, out, pred1, projection);
    (void)std::ranges::copy_if(in, out, pred1, projection);
  }
  {
    (void)std::ranges::remove_copy_if(it, it, out, pred1);
    (void)std::ranges::remove_copy_if(in, out, pred1);
    (void)std::ranges::remove_copy_if(it, it, out, pred1, projection);
    (void)std::ranges::remove_copy_if(in, out, pred1, projection);
  }
  // remove_copy
  {
    (void)std::ranges::remove_copy(it, it, out, val);
    (void)std::ranges::remove_copy(in, out, val);
    (void)std::ranges::remove_copy(it, it, out, val, projection);
    (void)std::ranges::remove_copy(in, out, val, projection);
  }
  // replace
  {
    (void)std::ranges::replace(it, it, val, val);
    (void)std::ranges::replace(in, val, val);
    (void)std::ranges::replace(it, it, val, val, projection);
    (void)std::ranges::replace(in, val, val, projection);
  }
  // replace_if
  {
    (void)std::ranges::replace_if(it, it, pred1, val);
    (void)std::ranges::replace_if(in, pred1, val);
    (void)std::ranges::replace_if(it, it, pred1, val, projection);
    (void)std::ranges::replace_if(in, pred1, val, projection);
  }
  // replace_copy_if
  {
    (void)std::ranges::replace_copy_if(it, it, out, pred1, val);
    (void)std::ranges::replace_copy_if(in, out, pred1, val);
    (void)std::ranges::replace_copy_if(it, it, out, pred1, val, projection);
    (void)std::ranges::replace_copy_if(in, out, pred1, val, projection);
  }
  // replace_copy
  {
    (void)std::ranges::replace_copy(it, it, out, val, val);
    (void)std::ranges::replace_copy(in, out, val, val);
    (void)std::ranges::replace_copy(it, it, out, val, val, projection);
    (void)std::ranges::replace_copy(in, out, val, val, projection);
  }
  // unique_copy
  {
    (void)std::ranges::unique_copy(it, it, out, pred2);
    (void)std::ranges::unique_copy(in, out, pred2);
    (void)std::ranges::unique_copy(it, it, out, pred2, projection);
    (void)std::ranges::unique_copy(in, out, pred2, projection);
  }
  // partition_copy
  {
    (void)std::ranges::partition_copy(it, it, out, out, pred1);
    (void)std::ranges::partition_copy(in, out, out, pred1);
    (void)std::ranges::partition_copy(it, it, out, out, pred1, projection);
    (void)std::ranges::partition_copy(in, out, out, pred1, projection);
  }
  in_in_pred(std::ranges::partial_sort_copy, pred2);
#if TEST_STD_VER > 20
  in_in_pred(std::ranges::starts_with, pred2);
#endif
  // merge
  {
    (void)std::ranges::merge(it, it, it, it, out, pred2);
    (void)std::ranges::merge(in, in, out, pred2);
    (void)std::ranges::merge(it, it, it, it, out, pred2, projection, projection);
    (void)std::ranges::merge(in, in, out, pred2, projection, projection);
  }
  // set_difference
  {
    (void)std::ranges::set_difference(it, it, it, it, out, pred2);
    (void)std::ranges::set_difference(in, in, out, pred2);
    (void)std::ranges::set_difference(it, it, it, it, out, pred2, projection, projection);
    (void)std::ranges::set_difference(in, in, out, pred2, projection, projection);
  }
  // set_intersection
  {
    (void)std::ranges::set_intersection(it, it, it, it, out, pred2);
    (void)std::ranges::set_intersection(in, in, out, pred2);
    (void)std::ranges::set_intersection(it, it, it, it, out, pred2, projection, projection);
    (void)std::ranges::set_intersection(in, in, out, pred2, projection, projection);
  }
  // set_symmetric_difference
  {
    (void)std::ranges::set_symmetric_difference(it, it, it, it, out, pred2);
    (void)std::ranges::set_symmetric_difference(in, in, out, pred2);
    (void)std::ranges::set_symmetric_difference(it, it, it, it, out, pred2, projection, projection);
    (void)std::ranges::set_symmetric_difference(in, in, out, pred2, projection, projection);
  }
  // set_union
  {
    (void)std::ranges::set_union(it, it, it, it, out, pred2);
    (void)std::ranges::set_union(in, in, out, pred2);
    (void)std::ranges::set_union(it, it, it, it, out, pred2, projection, projection);
    (void)std::ranges::set_union(in, in, out, pred2, projection, projection);
  }
  in_pred(std::ranges::remove_if, pred1);
  // remove
  {
    (void)std::ranges::remove(it, it, val);
    (void)std::ranges::remove(it, it, val, projection);
    (void)std::ranges::remove(in, val);
    (void)std::ranges::remove(in, val, projection);
  }
  in_pred(std::ranges::unique, pred2);
  in_pred(std::ranges::partition, pred1);
  in_pred(std::ranges::stable_partition, pred1);
  in_pred(std::ranges::sort, pred2);
  in_pred(std::ranges::stable_sort, pred2);
  // partial_sort
  {
    (void)std::ranges::partial_sort(it, it, it, pred2);
    (void)std::ranges::partial_sort(in, it, pred2);
    (void)std::ranges::partial_sort(it, it, it, pred2, projection);
    (void)std::ranges::partial_sort(in, it, pred2, projection);
  }
  // nth_element
  {
    (void)std::ranges::nth_element(it, it, it, pred2);
    (void)std::ranges::nth_element(in, it, pred2);
    (void)std::ranges::nth_element(it, it, it, pred2, projection);
    (void)std::ranges::nth_element(in, it, pred2, projection);
  }
  // inplace_merge
  {
    (void)std::ranges::inplace_merge(it, it, it, pred2);
    (void)std::ranges::inplace_merge(in, it, pred2);
    (void)std::ranges::inplace_merge(it, it, it, pred2, projection);
    (void)std::ranges::inplace_merge(in, it, pred2, projection);
  }
  in_pred(std::ranges::make_heap, pred2);
  in_pred(std::ranges::push_heap, pred2);
  in_pred(std::ranges::pop_heap, pred2);
  in_pred(std::ranges::sort_heap, pred2);
  in_pred(std::ranges::prev_permutation, pred2);
  in_pred(std::ranges::next_permutation, pred2);
}
