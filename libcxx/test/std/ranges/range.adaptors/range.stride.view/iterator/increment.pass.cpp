//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr __iterator& operator++()
// constexpr void operator++(int)
// constexpr __iterator operator++(int)

#include <iterator>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

template <class T>
concept CanPostIncrementVoid = std::is_same_v<void, decltype(std::declval<T>()++)> && requires(T& t) { t++; };
template <class T>
concept CanPostIncrementIterator = std::is_same_v<T, decltype(std::declval<T>()++)> && requires(T& t) { t = t++; };
template <class T>
concept CanPreIncrementIterator = std::is_same_v<T&, decltype(++(std::declval<T>()))> && requires(T& t) { t = ++t; };

// A stride view with a base that is a non forward range returns void from operator++
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(CanPostIncrementVoid<StrideViewOverInputViewIterator>);
static_assert(!CanPostIncrementIterator<StrideViewOverInputViewIterator>);
static_assert(CanPreIncrementIterator<StrideViewOverInputViewIterator>);

// A stride view with a base that is a forward range returns void from operator++
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;
static_assert(!CanPostIncrementVoid<StrideViewOverForwardViewIterator>);
static_assert(CanPostIncrementIterator<StrideViewOverForwardViewIterator>);
static_assert(CanPreIncrementIterator<StrideViewOverForwardViewIterator>);

template <typename Iter>
  requires std::sized_sentinel_for<Iter, Iter> && (!std::forward_iterator<Iter>)
constexpr bool test_non_forward_operator_increment(Iter zero_begin, Iter three_begin, Iter end) {
  using Base               = BasicTestView<Iter, Iter>;
  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;
  static_assert(std::weakly_incrementable<StrideViewIterator>);
  static_assert(!std::ranges::forward_range<Base>);

  auto base_view_offset_zero              = Base(zero_begin, end);
  auto base_view_offset_three             = Base(three_begin, end);
  auto stride_view_over_base_zero_offset  = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_three_offset = std::ranges::stride_view(base_view_offset_three, 3);

  auto sv_zero_offset_begin  = stride_view_over_base_zero_offset.begin();
  auto sv_three_offset_begin = stride_view_over_base_three_offset.begin();

  auto sv_zero_offset_third_index = sv_zero_offset_begin; // With a stride of 3, so ++ moves 3 indexes.
  ++sv_zero_offset_third_index;
  assert(*sv_three_offset_begin == *sv_zero_offset_third_index);

  sv_zero_offset_third_index = sv_zero_offset_begin;
  sv_zero_offset_third_index++;
  assert(*sv_three_offset_begin == *sv_zero_offset_third_index);

  // See if both get to the end (with pre-increment).
  auto sv_zero_offset_incremented_to_end = sv_zero_offset_begin;
  ++sv_zero_offset_incremented_to_end; // 3
  ++sv_zero_offset_incremented_to_end; // 6
  ++sv_zero_offset_incremented_to_end; // 9
  ++sv_zero_offset_incremented_to_end; // End

  auto sv_three_offset_incremented_to_end = sv_three_offset_begin; // With a stride of 3, so ++ moves 3 indexes.
  ++sv_three_offset_incremented_to_end;                            // 6
  ++sv_three_offset_incremented_to_end;                            // 9
  ++sv_three_offset_incremented_to_end;                            // End

  assert(sv_three_offset_incremented_to_end == sv_zero_offset_incremented_to_end);
  assert(sv_three_offset_incremented_to_end == stride_view_over_base_three_offset.end());
  assert(sv_zero_offset_incremented_to_end == stride_view_over_base_zero_offset.end());

  // See if both get to the end (with post-increment).
  sv_zero_offset_incremented_to_end = sv_zero_offset_begin;
  sv_zero_offset_incremented_to_end++; // 3
  sv_zero_offset_incremented_to_end++; // 6
  sv_zero_offset_incremented_to_end++; // 9
  sv_zero_offset_incremented_to_end++; // End

  sv_three_offset_incremented_to_end = sv_three_offset_begin; // With a stride of 3, so ++ moves 3 indexes.
  sv_three_offset_incremented_to_end++;                       // 6
  sv_three_offset_incremented_to_end++;                       // 9
  sv_three_offset_incremented_to_end++;                       // End

  assert(sv_three_offset_incremented_to_end == sv_zero_offset_incremented_to_end);
  assert(sv_three_offset_incremented_to_end == stride_view_over_base_three_offset.end());
  assert(sv_zero_offset_incremented_to_end == stride_view_over_base_zero_offset.end());

  return true;
}

template <std::forward_iterator Iter>
constexpr bool test_forward_operator_increment(Iter begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;

  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;
  static_assert(std::ranges::forward_range<Base>);
  static_assert(std::weakly_incrementable<StrideViewIterator>);

  auto base_view_offset_zero             = Base(begin, end);
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto sv_zero_offset_begin              = stride_view_over_base_zero_offset.begin();

  // Create a ground truth for comparison.
  auto sv_zero_offset_third_index_key = stride_view_over_base_zero_offset.begin();
  sv_zero_offset_third_index_key++;

  auto sv_zero_offset_third_index = ++sv_zero_offset_begin;
  assert(*sv_zero_offset_third_index == *sv_zero_offset_begin);
  assert(*sv_zero_offset_third_index == *sv_zero_offset_third_index_key);

  sv_zero_offset_begin       = stride_view_over_base_zero_offset.begin();
  sv_zero_offset_third_index = sv_zero_offset_begin;
  sv_zero_offset_third_index++;
  assert(*sv_zero_offset_third_index == *sv_zero_offset_third_index_key);

  sv_zero_offset_begin                   = stride_view_over_base_zero_offset.begin();
  auto sv_zero_offset_incremented_to_end = sv_zero_offset_begin;
  ++sv_zero_offset_incremented_to_end; // 3
  ++sv_zero_offset_incremented_to_end; // 6
  ++sv_zero_offset_incremented_to_end; // 9
  ++sv_zero_offset_incremented_to_end; // End

  auto sv_zero_offset_incremented_to_end_reset = sv_zero_offset_begin; // With a stride of 3, so ++ moves 3 indexes.
  sv_zero_offset_incremented_to_end_reset      = ++sv_zero_offset_incremented_to_end_reset; // 3
  sv_zero_offset_incremented_to_end_reset      = ++sv_zero_offset_incremented_to_end_reset; // 6
  sv_zero_offset_incremented_to_end_reset      = ++sv_zero_offset_incremented_to_end_reset; // 9
  sv_zero_offset_incremented_to_end_reset      = ++sv_zero_offset_incremented_to_end_reset; // End

  assert(sv_zero_offset_incremented_to_end == sv_zero_offset_incremented_to_end_reset);
  assert(sv_zero_offset_incremented_to_end == stride_view_over_base_zero_offset.end());

  sv_zero_offset_incremented_to_end = sv_zero_offset_begin;
  sv_zero_offset_incremented_to_end++; // 3
  sv_zero_offset_incremented_to_end++; // 6
  sv_zero_offset_incremented_to_end++; // 9
  sv_zero_offset_incremented_to_end++; // End
  assert(sv_zero_offset_incremented_to_end == stride_view_over_base_zero_offset.end());

  return true;
}

constexpr bool test_properly_handling_missing() {
  // Check whether __missing_ gets handled properly.
  using Base = BasicTestView<int*, int*>;
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto base    = Base(arr, arr + 10);
  auto strider = std::ranges::stride_view<Base>(base, 7);

  auto strider_iter = strider.end();

  strider_iter--;
  assert(*strider_iter == 8);

  // Now that we are back among the valid, we should
  // have a normal stride length back (i.e., __missing_
  // should be equal to 0).
  strider_iter--;
  assert(*strider_iter == 1);

  strider_iter++;
  assert(*strider_iter == 8);

  // By striding past the end, we are going to generate
  // another __missing_ != 0 value.
  strider_iter++;
  assert(strider_iter == strider.end());

  // Make sure that all sentinel operations work!
  assert(strider.end() == std::default_sentinel);
  assert(std::default_sentinel == strider.end());

  assert(strider_iter - std::default_sentinel == 0);
  assert(std::default_sentinel - strider.end() == 0);
  assert(std::default_sentinel - strider_iter == 0);

  // Let's make sure that the newly regenerated __missing__ gets used.
  strider_iter += -2;
  assert(*strider_iter == 1);

  return true;
}

int main(int, char**) {
  {
    constexpr int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    test_forward_operator_increment(arr, arr + 11);
    test_forward_operator_increment(vec.begin(), vec.end());
  }

  {
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    test_non_forward_operator_increment(
        SizedInputIterator(arr), SizedInputIterator(arr + 3), SizedInputIterator(arr + 10));
  }

  test_properly_handling_missing();
  static_assert(test_properly_handling_missing());
  return 0;
}
