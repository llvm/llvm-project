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
// Range algorithms should return `std::ranges::dangling` when given a dangling range.

#include <algorithm>

#include <array>
#include <concepts>
#include <iterator>
#include <ranges>
#include <random>

#include "test_iterators.h"

struct NonBorrowedRange {
  using Iter = int*;
  using Sent = sentinel_wrapper<Iter>;

  int* data_;
  size_t size_;

  template <size_t N>
  constexpr explicit NonBorrowedRange(std::array<int, N>& arr) : data_{arr.data()}, size_{arr.size()} {}

  constexpr Iter begin() const { return data_; };
  constexpr Sent end() const { return Sent{data_ + size_}; };
};

using R = NonBorrowedRange;

// (dangling_in, ...)
template <class ExpectedT = std::ranges::dangling, class Func, std::ranges::range Input, class ...Args>
constexpr void dangling_1st(Func&& func, Input& in, Args&& ...args) {
  decltype(auto) result = func(R(in), std::forward<Args>(args)...);
  static_assert(std::same_as<decltype(result), ExpectedT>);
}

// (in, dangling_in, ...)
template <class ExpectedT = std::ranges::dangling, class Func, std::ranges::range Input, class ...Args>
constexpr void dangling_2nd(Func&& func, Input& in1, Input& in2, Args&& ...args) {
  decltype(auto) result = func(in1, R(in2), std::forward<Args>(args)...);
  static_assert(std::same_as<decltype(result), ExpectedT>);
}

// (dangling_in1, dangling_in2, ...)
template <class ExpectedT = std::ranges::dangling, class Func, std::ranges::range Input, class ...Args>
constexpr void dangling_both(Func&& func, Input& in1, Input& in2, Args&& ...args) {
  decltype(auto) result = func(R(in1), R(in2), std::forward<Args>(args)...);
  static_assert(std::same_as<decltype(result), ExpectedT>);
}

std::mt19937 rand_gen() { return std::mt19937(); }

// TODO: also check the iterator values for algorithms that return `*_result` types.
constexpr bool test_all() {
  using std::ranges::dangling;

  using std::ranges::binary_transform_result;
  using std::ranges::copy_result;
  using std::ranges::copy_backward_result;
  using std::ranges::copy_if_result;
  using std::ranges::for_each_result;
  using std::ranges::merge_result;
  using std::ranges::minmax_result;
  using std::ranges::mismatch_result;
  using std::ranges::move_result;
  using std::ranges::move_backward_result;
  using std::ranges::partial_sort_copy_result;
  using std::ranges::partition_copy_result;
  using std::ranges::remove_copy_result;
  using std::ranges::remove_copy_if_result;
  using std::ranges::replace_copy_result;
  using std::ranges::replace_copy_if_result;
  using std::ranges::reverse_copy_result;
  using std::ranges::rotate_copy_result;
  using std::ranges::set_difference_result;
  using std::ranges::set_intersection_result;
  using std::ranges::set_symmetric_difference_result;
  using std::ranges::set_union_result;
  using std::ranges::swap_ranges_result;
  using std::ranges::unary_transform_result;
  using std::ranges::unique_copy_result;

  auto unary_pred = [](int i) { return i > 0; };
  auto binary_pred = [](int i, int j) { return i < j; };
  auto gen = [] { return 42; };

  std::array in = {1, 2, 3};
  std::array in2 = {4, 5, 6};

  auto mid = in.begin() + 1;

  std::array output = {7, 8, 9, 10, 11, 12};
  auto out = output.begin();
  auto out2 = output.begin() + 1;

  int x = 2;
  size_t count = 1;

  dangling_1st(std::ranges::find, in, x);
  dangling_1st(std::ranges::find_if, in, unary_pred);
  dangling_1st(std::ranges::find_if_not, in, unary_pred);
  dangling_1st(std::ranges::find_first_of, in, in2);
  dangling_1st(std::ranges::adjacent_find, in);
  dangling_1st<mismatch_result<dangling, int*>>(std::ranges::mismatch, in, in2);
  dangling_2nd<mismatch_result<int*, dangling>>(std::ranges::mismatch, in, in2);
  dangling_both<mismatch_result<dangling, dangling>>(std::ranges::mismatch, in, in2);
  dangling_1st(std::ranges::partition_point, in, unary_pred);
  dangling_1st(std::ranges::lower_bound, in, x);
  dangling_1st(std::ranges::upper_bound, in, x);
  dangling_1st(std::ranges::equal_range, in, x);
  dangling_1st(std::ranges::min_element, in);
  dangling_1st(std::ranges::max_element, in);
  dangling_1st<minmax_result<dangling>>(std::ranges::minmax_element, in);
  dangling_1st(std::ranges::search, in, in2);
  dangling_1st(std::ranges::search_n, in, count, x);
  dangling_1st(std::ranges::find_end, in, in2);
  dangling_1st(std::ranges::is_sorted_until, in);
  dangling_1st(std::ranges::is_heap_until, in);
  dangling_1st<for_each_result<dangling, decltype(unary_pred)>>(std::ranges::for_each, in, unary_pred);
  dangling_1st<copy_result<dangling, int*>>(std::ranges::copy, in, out);
  dangling_1st<copy_backward_result<dangling, int*>>(std::ranges::copy_backward, in, output.end());
  dangling_1st<copy_if_result<dangling, int*>>(std::ranges::copy_if, in, out, unary_pred);
  dangling_1st<move_result<dangling, int*>>(std::ranges::move, in, out);
  dangling_1st<move_backward_result<dangling, int*>>(std::ranges::move_backward, in, output.end());
  dangling_1st(std::ranges::fill, in, x);
  { // transform
    std::array out_transform = {false, true, true};
    dangling_1st<unary_transform_result<dangling, bool*>>(std::ranges::transform, in, out_transform.begin(), unary_pred);
    dangling_1st<binary_transform_result<dangling, int*, bool*>>(
        std::ranges::transform, in, in2, out_transform.begin(), binary_pred);
    dangling_2nd<binary_transform_result<int*, dangling, bool*>>(
        std::ranges::transform, in, in2, out_transform.begin(), binary_pred);
    dangling_both<binary_transform_result<dangling, dangling, bool*>>(
        std::ranges::transform, in, in2, out_transform.begin(), binary_pred);
  }
  dangling_1st(std::ranges::generate, in, gen);
  dangling_1st<remove_copy_result<dangling, int*>>(std::ranges::remove_copy, in, out, x);
  dangling_1st<remove_copy_if_result<dangling, int*>>(std::ranges::remove_copy_if, in, out, unary_pred);
  dangling_1st(std::ranges::replace, in, x, x);
  dangling_1st(std::ranges::replace_if, in, std::identity{}, x);
  dangling_1st<replace_copy_result<dangling, int*>>(std::ranges::replace_copy, in, out, x, x);
  dangling_1st<replace_copy_if_result<dangling, int*>>(std::ranges::replace_copy_if, in, out, unary_pred, x);
  dangling_1st<swap_ranges_result<dangling, int*>>(std::ranges::swap_ranges, in, in2);
  dangling_2nd<swap_ranges_result<int*, dangling>>(std::ranges::swap_ranges, in, in2);
  dangling_both<swap_ranges_result<dangling, dangling>>(std::ranges::swap_ranges, in, in2);
  dangling_1st<reverse_copy_result<dangling, int*>>(std::ranges::reverse_copy, in, out);
  dangling_1st<rotate_copy_result<dangling, int*>>(std::ranges::rotate_copy, in, mid, out);
  dangling_1st<unique_copy_result<dangling, int*>>(std::ranges::unique_copy, in, out);
  dangling_1st<partition_copy_result<dangling, int*, int*>>(std::ranges::partition_copy, in, out, out2, unary_pred);
  dangling_1st<partial_sort_copy_result<dangling, int*>>(std::ranges::partial_sort_copy, in, in2);
  dangling_2nd<partial_sort_copy_result<int*, dangling>>(std::ranges::partial_sort_copy, in, in2);
  dangling_both<partial_sort_copy_result<dangling, dangling>>(std::ranges::partial_sort_copy, in, in2);
  dangling_1st<merge_result<dangling, int*, int*>>(std::ranges::merge, in, in2, out);
  dangling_2nd<merge_result<int*, dangling, int*>>(std::ranges::merge, in, in2, out);
  dangling_both<merge_result<dangling, dangling, int*>>(std::ranges::merge, in, in2, out);
  dangling_1st<set_difference_result<dangling, int*>>(std::ranges::set_difference, in, in2, out);
  dangling_1st<set_intersection_result<dangling, int*, int*>>(std::ranges::set_intersection, in, in2, out);
  dangling_2nd<set_intersection_result<int*, dangling, int*>>(std::ranges::set_intersection, in, in2, out);
  dangling_both<set_intersection_result<dangling, dangling, int*>>(std::ranges::set_intersection, in, in2, out);
  dangling_1st<set_symmetric_difference_result<dangling, int*, int*>>(
      std::ranges::set_symmetric_difference, in, in2, out);
  dangling_2nd<set_symmetric_difference_result<int*, dangling, int*>>(
      std::ranges::set_symmetric_difference, in, in2, out);
  dangling_both<set_symmetric_difference_result<dangling, dangling, int*>>(
      std::ranges::set_symmetric_difference, in, in2, out);
  dangling_1st<set_union_result<dangling, int*, int*>>(std::ranges::set_union, in, in2, out);
  dangling_2nd<set_union_result<int*, dangling, int*>>(std::ranges::set_union, in, in2, out);
  dangling_both<set_union_result<dangling, dangling, int*>>(std::ranges::set_union, in, in2, out);
  dangling_1st(std::ranges::remove, in, x);
  dangling_1st(std::ranges::remove_if, in, unary_pred);
  dangling_1st(std::ranges::reverse, in);
  dangling_1st(std::ranges::rotate, in, mid);
  if (!std::is_constant_evaluated()) // `shuffle` isn't `constexpr`.
    dangling_1st(std::ranges::shuffle, in, rand_gen());
  dangling_1st(std::ranges::unique, in);
  dangling_1st(std::ranges::partition, in, unary_pred);
  if (!std::is_constant_evaluated())
    dangling_1st(std::ranges::stable_partition, in, unary_pred);
  dangling_1st(std::ranges::sort, in);
  if (!std::is_constant_evaluated())
    dangling_1st(std::ranges::stable_sort, in);
  dangling_1st(std::ranges::partial_sort, in, mid);
  dangling_1st(std::ranges::nth_element, in, mid);
  if (!std::is_constant_evaluated())
    dangling_1st(std::ranges::inplace_merge, in, mid);
  dangling_1st(std::ranges::make_heap, in);
  dangling_1st(std::ranges::push_heap, in);
  dangling_1st(std::ranges::pop_heap, in);
  dangling_1st(std::ranges::sort_heap, in);
  //dangling_1st(std::ranges::prev_permutation, in);
  //dangling_1st(std::ranges::next_permutation, in);

  return true;
}

int main(int, char**) {
  test_all();
  static_assert(test_all());

  return 0;
}
