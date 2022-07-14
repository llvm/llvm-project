//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>
//
// Range algorithms that take predicates should support predicates that return a non-boolean value as long as the
// returned type is implicitly convertible to bool.

#include <algorithm>

#include <array>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <ranges>

#include "boolean_testable.h"

auto unary_pred = [](int i) { return BooleanTestable(i > 0); };
static_assert(!std::same_as<decltype(unary_pred(1)), bool>);
static_assert(std::convertible_to<decltype(unary_pred(1)), bool>);

auto binary_pred = [](int i, int j) { return BooleanTestable(i < j); };
static_assert(!std::same_as<decltype(binary_pred(1, 2)), bool>);
static_assert(std::convertible_to<decltype(binary_pred(1, 2)), bool>);

// Invokes both the (iterator, sentinel, ...) and the (range, ...) overloads of the given niebloid.

// (in, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test(Func&& func, Input& in, Args&& ...args) {
  func(in.begin(), in.end(), std::forward<Args>(args)...);
  func(in, std::forward<Args>(args)...);
}

// (in1, in2, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test(Func&& func, Input& in1, Input& in2, Args&& ...args) {
  func(in1.begin(), in1.end(), in2.begin(), in2.end(), std::forward<Args>(args)...);
  func(in1, in2, std::forward<Args>(args)...);
}

// (in, mid, ...)
template <class Func, std::ranges::range Input, class ...Args>
constexpr void test_mid(Func&& func, Input& in, std::ranges::iterator_t<Input> mid, Args&& ...args) {
  func(in.begin(), mid, in.end(), std::forward<Args>(args)...);
  func(in, mid, std::forward<Args>(args)...);
}

constexpr bool test_all() {
  std::array in = {1, 2, 3};
  std::array in2 = {4, 5, 6};
  auto mid = in.begin() + 1;

  std::array output = {7, 8, 9, 10, 11, 12};
  auto out = output.begin();
  auto out2 = output.begin() + 1;

  int x = 2;
  int count = 1;

  test(std::ranges::any_of, in, unary_pred);
  test(std::ranges::all_of, in, unary_pred);
  test(std::ranges::none_of, in, unary_pred);
  test(std::ranges::find_if, in, unary_pred);
  test(std::ranges::find_if_not, in, unary_pred);
  test(std::ranges::find_first_of, in, in2, binary_pred);
  test(std::ranges::adjacent_find, in, binary_pred);
  test(std::ranges::mismatch, in, in2, binary_pred);
  test(std::ranges::equal, in, in2, binary_pred);
  test(std::ranges::lexicographical_compare, in, in2, binary_pred);
  test(std::ranges::partition_point, in, unary_pred);
  test(std::ranges::lower_bound, in, x, binary_pred);
  test(std::ranges::upper_bound, in, x, binary_pred);
  test(std::ranges::equal_range, in, x, binary_pred);
  test(std::ranges::binary_search, in, x, binary_pred);

  // min
  std::ranges::min(1, 2, binary_pred);
  std::ranges::min(std::initializer_list<int>{1, 2}, binary_pred);
  std::ranges::min(in, binary_pred);
  // max
  std::ranges::max(1, 2, binary_pred);
  std::ranges::max(std::initializer_list<int>{1, 2}, binary_pred);
  std::ranges::max(in, binary_pred);
  // minmax
  std::ranges::minmax(1, 2, binary_pred);
  std::ranges::minmax(std::initializer_list<int>{1, 2}, binary_pred);
  std::ranges::minmax(in, binary_pred);

  test(std::ranges::min_element, in, binary_pred);
  test(std::ranges::max_element, in, binary_pred);
  test(std::ranges::minmax_element, in, binary_pred);
  test(std::ranges::count_if, in, unary_pred);
  test(std::ranges::search, in, in2, binary_pred);
  test(std::ranges::search_n, in, count, x, binary_pred);
  test(std::ranges::find_end, in, in2, binary_pred);
  test(std::ranges::is_partitioned, in, unary_pred);
  test(std::ranges::is_sorted, in, binary_pred);
  test(std::ranges::is_sorted_until, in, binary_pred);
  //test(std::ranges::includes, in, in2, binary_pred);
  //test(std::ranges::is_heap, in, binary_pred);
  //test(std::ranges::is_heap_until, in, binary_pred);
  //std::ranges::clamp(2, 1, 3, binary_pred);
  //test(std::ranges::is_permutation, in, in2, binary_pred);
  test(std::ranges::copy_if, in, out, unary_pred);
  //test(std::ranges::remove_copy_if, in, out, unary_pred);
  test(std::ranges::replace_if, in, unary_pred, x);
  //test(std::ranges::replace_copy_if, in, out, unary_pred, x);
  //test(std::ranges::unique_copy, in, out, binary_pred);
  test(std::ranges::partition_copy, in, out, out2, unary_pred);
  //test(std::ranges::partial_sort_copy, in, in2, binary_pred);
  test(std::ranges::merge, in, in2, out, binary_pred);
  test(std::ranges::set_difference, in, in2, out, binary_pred);
  test(std::ranges::set_intersection, in, in2, out, binary_pred);
  test(std::ranges::set_symmetric_difference, in, in2, out, binary_pred);
  test(std::ranges::set_union, in, in2, out, binary_pred);
  test(std::ranges::remove_if, in, unary_pred);
  //test(std::ranges::unique, in, binary_pred);
  test(std::ranges::partition, in, unary_pred);
  if (!std::is_constant_evaluated())
    test(std::ranges::stable_partition, in, unary_pred);
  test(std::ranges::sort, in, binary_pred);
  if (!std::is_constant_evaluated())
    test(std::ranges::stable_sort, in, binary_pred);
  test_mid(std::ranges::partial_sort, in, mid, binary_pred);
  test_mid(std::ranges::nth_element, in, mid, binary_pred);
  //test_mid(std::ranges::inplace_merge, in, mid, binary_pred);
  test(std::ranges::make_heap, in, binary_pred);
  test(std::ranges::push_heap, in, binary_pred);
  test(std::ranges::pop_heap, in, binary_pred);
  test(std::ranges::sort_heap, in, binary_pred);
  //test(std::ranges::prev_permutation, in, binary_pred);
  //test(std::ranges::next_permutation, in, binary_pred);

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
