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
// Range algorithms should work with proxy iterators. For example, the implementations should use `iter_swap` (which is
// a customization point) rather than plain `swap` (which might not work with certain valid iterators).

#include <algorithm>

#include <array>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <ranges>

#include "test_iterators.h"

template <class Func, std::ranges::range Input, class ...Args>
constexpr void test(Func&& func, Input& in, Args&& ...args) {
  func(in.begin(), in.end(), std::forward<Args>(args)...);
  func(in, std::forward<Args>(args)...);
}

template <class Func, std::ranges::range Range1, std::ranges::range Range2, class ...Args>
constexpr void test(Func&& func, Range1& r1, Range2& r2, Args&& ...args) {
  func(r1.begin(), r1.end(), r2.begin(), r2.end(), std::forward<Args>(args)...);
  func(r1, r2, std::forward<Args>(args)...);
}

template <class Func, std::ranges::range Input, class ...Args>
constexpr void test_mid(Func&& func, Input& in, std::ranges::iterator_t<Input> mid, Args&& ...args) {
  func(in.begin(), mid, in.end(), std::forward<Args>(args)...);
  func(in, mid, std::forward<Args>(args)...);
}

constexpr bool test_all() {
  std::array input = {1, 2, 3};
  ProxyRange in{input};
  std::array input2 = {4, 5, 6};
  ProxyRange in2{input2};

  auto mid = in.begin() + 1;

  std::array output = {4, 5, 6, 7, 8, 9};
  ProxyIterator out{output.begin()};
  //auto out2 = output.begin() + 1;

  int num = 2;
  Proxy<int&> x{num};
  int count = 1;

  auto unary_pred = [](const Proxy<int>& val) { return val.getData() > 0; };
  //auto binary_pred = [](const Proxy<int>& lhs, const Proxy<int>& rhs) { return lhs.getData() == rhs.getData(); };
  auto binary_func = [](const Proxy<int>& lhs, const Proxy<int>&) -> Proxy<int> { return lhs.getData(); };
  //auto gen = [] { return Proxy<int>(42); };
  //std::mt19937 rand_gen;

  test(std::ranges::any_of, in, unary_pred);
  test(std::ranges::all_of, in, unary_pred);
  test(std::ranges::none_of, in, unary_pred);
  test(std::ranges::find, in, x);
  test(std::ranges::find_if, in, unary_pred);
  test(std::ranges::find_if_not, in, unary_pred);
  test(std::ranges::find_first_of, in, in2);
  test(std::ranges::adjacent_find, in);
  test(std::ranges::mismatch, in, in2);
  test(std::ranges::equal, in, in2);
  test(std::ranges::lexicographical_compare, in, in2);
  //test(std::ranges::partition_point, unary_pred);
  test(std::ranges::lower_bound, in, x);
  test(std::ranges::upper_bound, in, x);
  //test(std::ranges::equal_range, in, x);
  test(std::ranges::binary_search, in, x);

  test(std::ranges::min_element, in);
  test(std::ranges::max_element, in);
  test(std::ranges::minmax_element, in);
  test(std::ranges::count, in, x);
  test(std::ranges::count_if, in, unary_pred);
  test(std::ranges::search, in, in2);
  test(std::ranges::search_n, in, count, x);
  test(std::ranges::find_end, in, in2);
  test(std::ranges::is_partitioned, in, unary_pred);
  test(std::ranges::is_sorted, in);
  test(std::ranges::is_sorted_until, in);
  //test(std::ranges::includes, in, in2);
  //test(std::ranges::is_heap, in);
  //test(std::ranges::is_heap_until, in);
  //test(std::ranges::is_permutation, in, in2);
  test(std::ranges::for_each, in, std::identity{});
  std::ranges::for_each_n(in.begin(), count, std::identity{});
  test(std::ranges::copy, in, out);
  std::ranges::copy_n(in.begin(), count, out);
  test(std::ranges::copy_if, in, out, unary_pred);
  // TODO: uncomment `copy_backward` once https://reviews.llvm.org/D128864 lands.
  //test(std::ranges::copy_backward, in, out);
  test(std::ranges::move, in, out);
  // TODO: uncomment `move_backward` once https://reviews.llvm.org/D128864 lands.
  // test(std::ranges::move_backward, in, out);
  test(std::ranges::fill, in, x);
  std::ranges::fill_n(in.begin(), count, x);
  test(std::ranges::transform, in, out, std::identity{});
  test(std::ranges::transform, in, in2, out, binary_func);
  //test(std::ranges::generate, in, gen);
  //std::ranges::generate_n(in.begin(), count, gen);
  //test(std::ranges::remove_copy, in, out, x);
  //test(std::ranges::remove_copy_if, in, out, unary_pred);
  test(std::ranges::replace, in, x, x);
  test(std::ranges::replace_if, in, unary_pred, x);
  //test(std::ranges::replace_copy, in, out, x, x);
  //test(std::ranges::replace_copy_if, in, out, unary_pred, x);
  //test(std::ranges::swap_ranges, in, in2);
  test(std::ranges::reverse_copy, in, out);
  test_mid(std::ranges::rotate_copy, in, mid, out);
  //test(std::ranges::sample, in, out, count, rand_gen);
  //test(std::ranges::unique_copy, in, out);
  //test(std::ranges::partition_copy, in, out, out2, unary_pred);
  //test_mid(std::ranges::partial_sort_copy, in, in2);
  test(std::ranges::merge, in, in2, out);
  test(std::ranges::set_difference, in, in2, out);
  test(std::ranges::set_intersection, in, in2, out);
  test(std::ranges::set_symmetric_difference, in, in2, out);
  test(std::ranges::set_union, in, in2, out);
  test(std::ranges::remove, in, x);
  test(std::ranges::remove_if, in, unary_pred);
  test(std::ranges::reverse, in);
  //test_mid(std::ranges::rotate, in, mid);
  //test(std::ranges::shuffle, in, rand_gen);
  //test(std::ranges::unique, in);
  //test(std::ranges::partition, in, binary_pred);
  //if (!std::is_constant_evaluated())
  //  test(std::ranges::stable_partition, in, binary_pred);
  test(std::ranges::sort, in);
  // TODO: `stable_sort` requires `ranges::rotate` to be implemented.
  //if (!std::is_constant_evaluated())
  //  test(std::ranges::stable_sort, in);
  //test_mid(std::ranges::partial_sort, in, mid);
  test_mid(std::ranges::nth_element, in, mid);
  //if (!std::is_constant_evaluated())
  //  test_mid(std::ranges::inplace_merge, in, mid);
  test(std::ranges::make_heap, in);
  test(std::ranges::push_heap, in);
  test(std::ranges::pop_heap, in);
  test(std::ranges::sort_heap, in);
  //test(std::ranges::prev_permutation, in);
  //test(std::ranges::next_permutation, in);

  // The algorithms that work on uninitialized memory have constraints that prevent proxy iterators from being used with
  // them.

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
